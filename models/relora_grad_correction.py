import os
import math
import json
from typing import List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
# import bitsandbytes as bnb
# import bitsandbytes.functional as bnbF

# from transformers import AutoModelForCausalLM, AutoConfig

# from loguru import logger


@dataclass
class ReLoRaConfig:
    r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: List[str]
    correct_coef: float
    keep_original_weights: bool
    lora_only: bool = False
    trainable_scaling: bool = False
    # quantize: str = None
    # use_double_quant: bool = False


# def merge_and_reinit_functional(module):
#     if not isinstance(module, ReLoRaLinear):
#         return

#     # if module.quantize is not None:
#     #     # Look below in merge_and_reinint method for the inspiration on how to implement this
#     #     raise NotImplementedError("merge_and_reinit_functional for quantized models is not implemented yet. Use non-functional implementation")

#     _delta = module.lora_B.weight @ module.lora_A.weight
#     _delta = _delta * module._post_lora_scale()
#     module.weight.data += _delta
#     nn.init.kaiming_uniform_(module.lora_A.weight, a=math.sqrt(5))
#     ## TODO: init by SVD of full grad

#     nn.init.zeros_(module.lora_B.weight)
#     if module.trainable_scaling:
#         nn.init.zeros_(module.scaling)


class ReLoRaModel(torch.nn.Module):
    def __init__(
        self,
        model,
        *,
        target_modules,
        r=128,
        lora_alpha=32,
        lora_dropout=0.1,
        keep_original_weights=True,
        lora_only=False,
        trainable_scaling=False,
        correct_coef=0.05,
        # quantize=None,
        # use_double_quant=False,
    ):
        if r <= 0:
            raise ValueError("r must be positive. If you want r == 0, use the original model.")

        super().__init__()
        self.wrapped_model: nn.Module = model
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules
        self.keep_original_weights = keep_original_weights
        self.lora_only = lora_only
        self.trainable_scaling = trainable_scaling
        self.correct_coef = correct_coef

        self._config = ReLoRaConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            correct_coef=correct_coef,
            keep_original_weights=keep_original_weights,
            # quantize=quantize,
            # use_double_quant=use_double_quant,
        )

        # patch methods
        self.forward = self.wrapped_model.forward

        target_modules_list = target_modules
        if isinstance(target_modules_list, str):
            target_modules_list = [target_modules_list]

        for module_name, module in self.wrapped_model.named_modules():
            print(module_name)
            if not isinstance(module, nn.Linear):
                continue

            if not any(target_key in module_name for target_key in target_modules_list):
                continue
            
            print(f"ReLoRa'ing {module_name}")

            weight_data = module.weight.data if keep_original_weights else None
            bias_data = None
            if module.bias is not None:
                bias_data = module.bias.data if keep_original_weights else None

            new_module = ReLoRaLinear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                r=self.r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                lora_only=self.lora_only,
                trainable_scaling=self.trainable_scaling,
                # quantize=quantize,
                weight_data=weight_data,
                bias_data=bias_data,
                correct_coef=self.correct_coef,
                # bnb_4bit_use_double_quant=use_double_quant,
            )
            if self.keep_original_weights:
                # make lora'ed network to be exacty the same as the original network at initialization
                nn.init.zeros_(new_module.lora_A.weight)
                assert new_module.lora_A.bias is None
                assert new_module.lora_B.bias is None

            if self.lora_only:
                assert not self.keep_original_weights
                module.weight = None

            del module

            parent = self._get_parent(module_name)
            module_suffix = module_name.split(".")[-1]
            setattr(parent, module_suffix, new_module)

        torch.cuda.empty_cache()

    def _get_parent(self, module_name):
        module_names_list = module_name.split(".")
        parent_name = ".".join(module_names_list[:-1])
        parent = self.wrapped_model.get_submodule(parent_name)
        return parent

    def merge(self):
        for module in self.modules():
            if isinstance(module, ReLoRaLinear):
                module.merge()
                module.lora_A.weight.data = 0 * module.lora_A.weight.data
                module.lora_B.weight.data = 0 * module.lora_B.weight.data
                # compute weight grad for reinit
                module.weight.requires_grad = True
                module.lora_A.weight.requires_grad = False
                module.lora_B.weight.requires_grad = False

    def reinit(self, eps=1e-6):
        for module in self.modules():
            if isinstance(module, ReLoRaLinear):
                grad = module.weight.grad.detach()
                module.full_grad = grad
                U, _, V = torch.linalg.svd(grad)
                # print(grad.shape, U.shape, V.shape)
                module.reinit(init_U=U[:, :self.r], init_V=V[:self.r, :], eps=eps)
                # print(module.lora_A.weight.shape, V[:self.r, :].shape,  module.lora_B.weight.shape, U[:, :self.r].shape)
                module.weight.requires_grad = False
                module.lora_A.weight.requires_grad = True
                module.lora_B.weight.requires_grad = True

    def correct_grad(self):
        for module in self.modules():
            if isinstance(module, ReLoRaLinear):
                module.correct_grad()

    def save_pretrained(self, path):
        self.wrapped_model.save_pretrained(path)
        with open(os.path.join(path, "relora_config.json"), "w") as f:
            json.dump(self._config.__dict__, f, indent=4)

    # @classmethod
    # def from_pretrained(cls, path):
    #     with open(os.path.join(path, "relora_config.json"), "r") as f:
    #         relora_config = json.load(f)

    #     config = AutoConfig.from_pretrained(path)

    #     base_model = AutoModelForCausalLM.from_config(config)
    #     if "keep_original" in relora_config:
    #         print("WARNING: keep_original is deprecated. Use lora_only instead.")
    #         print(f"keep_original: {relora_config['keep_original']}")
    #         relora_config["lora_only"] = not relora_config.pop("keep_original")
    #         relora_config["keep_original_weights"] = not relora_config["lora_only"]

    #     if "trainable_scaling" not in relora_config:
    #         relora_config["trainable_scaling"] = False

    #     model = cls(base_model, **relora_config)

    #     with open(os.path.join(path, "pytorch_model.bin"), "rb") as f:
    #         state_dict = torch.load(f, map_location="cpu")

    #     model.wrapped_model.load_state_dict(state_dict, strict=True)
    #     return model


# The code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
class ReLoRaLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int,
        *,
        lora_alpha: int = 1,
        lora_dropout: float = 0.1,
        lora_only: bool = False,
        weight_data=None,
        bias_data=None,
        trainable_scaling: bool = False,
        correct_coef: float = 0.05,
        bias=True,
        device=None,
        dtype=None,
        # quantize=False,
        # bnb_4bit_use_double_quant=False,
        # bnb_4bit_quant_type="nf4",
    ):
        """Wraps linear layer x W into x W + x W_a @ W_b * lora_alpha / r
        
        Notice that scale = lora_alpha / r.
        """
        nn.Module.__init__(self)
        if r <= 0:
            raise ValueError("r must be positive. If you want r == 0, use the original model.")

        if lora_only:
            self.weight = None
            self.bias = None
        else:
            # if full model weight + lora weight
            if bias_data is None:
                bias_data = torch.zeros(out_features, device=device, dtype=dtype, requires_grad=True) if bias else None
            self.bias = nn.Parameter(bias_data) if bias else None

            if weight_data is None:
                # note that our trainable weight are W_a and W_b
                weight_data = torch.zeros(out_features, in_features, device=device, dtype=dtype, requires_grad=False)

            # if quantize is None:
            self.weight = nn.Parameter(weight_data, requires_grad=False)
            self.full_grad = torch.zeros_like(weight_data, requires_grad=False)
            # elif quantize == "4bit":
            #     self.weight = bnb.nn.Params4bit(
            #         weight_data,
            #         requires_grad=False,
            #         compress_statistics=bnb_4bit_use_double_quant,
            #         quant_type=bnb_4bit_quant_type,
            #     )
            # elif quantize == "8bit":
            #     logger.warning("Int8 currently does not support merge_and_reinit! It will fail")
            #     self.weight = bnb.nn.Int8Params(
            #         weight_data,
            #         requires_grad=False,
            #     )
            # else:
            #     raise ValueError(f"Unknown quantize type: {quantize}")

        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout)
        self.lora_only = lora_only
        self.trainable_scaling = trainable_scaling
        self.correct_coef = correct_coef
        # self.quantize = quantize

        if r > 0:
            self.lora_A = nn.Linear(in_features, r, bias=False)
            # nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            # initialize A by SVD of full grad
            # self.lora_A.weight.data = math.sqrt(5)*init_U
            self.lora_B = nn.Linear(r, out_features, bias=False)
            nn.init.zeros_(self.lora_A.weight)
            nn.init.zeros_(self.lora_B.weight)

            if trainable_scaling:
                self.scaling = nn.Parameter(torch.tensor([1.]), requires_grad=True)
            else:
                self.scaling = self.lora_alpha / self.r

            # Freezing the pre-trained weight matrix
            if not self.lora_only:
                self.weight.requires_grad = False
    
    def _post_lora_scale(self):
        if self.trainable_scaling:
            return self.scaling.tanh()

        return self.scaling

    @torch.no_grad()
    def merge(self):
        if self.lora_only:
            print("WARNING: Skipping merge and reinit, because only lora parameters are used")
            return

        # if not self.quantize:
        self.weight.data += self.lora_B.weight @ self.lora_A.weight * self._post_lora_scale()

    @torch.no_grad()
    def reinit(self, init_U, init_V, eps=1e-6):
        self.lora_B.weight.data = 0 * init_U
        ## TODO: test initialization of B: zero or eps?
        # self.lora_A.weight.data = init_V * math.sqrt(2/self.lora_A.weight.shape[1])
        self.lora_A.weight.data = init_V * math.sqrt(2/self.r)
        if self.trainable_scaling:
            nn.init.zeros_(self.scaling)

    @torch.no_grad()
    def correct_grad(self):
        self.lora_A.weight.grad.data = self.correct_coef * self.lora_B.weight.grad.data.T @ self.full_grad + (1-self.correct_coef) * self.lora_A.weight.grad.data
        self.lora_B.weight.grad.data = self.correct_coef * self.full_grad @ self.lora_A.weight.grad.data.T + (1-self.correct_coef) * self.lora_B.weight.grad.data


    def forward(self, x: torch.Tensor):
        if self.lora_only:
            # just lora
            return self.lora_B(self.lora_A(self.lora_dropout(x))) * self._post_lora_scale()

        # if self.quantize == "4bit":
        #     result = bnb.matmul_4bit(x, self.weight.t(), bias=self.bias, quant_state=self.weight.quant_state)
        # elif self.quantize == "8bit":
        #     result = bnb.matmul(x, self.weight.t(), bias=self.bias, quant_state=self.weight.quant_state)
        # else:
        # print(self.weight.shape, x.shape)
        result = F.linear(x, self.weight, bias=self.bias)

        if self.r > 0:
            result += self.lora_B(self.lora_A(self.lora_dropout(x))) * self._post_lora_scale()
        return result
