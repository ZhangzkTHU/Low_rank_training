import torch
# from torch import Tensor
from torch.optim import Adam
from torch.optim.optimizer import _dispatch_sqrt
# Define custom optimizer 
class Adam_proj(Adam): 
	def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), weight_decay=0):
		super().__init__(params, lr=lr, betas=betas)
		self.weight_decay = weight_decay
		self.layer = 0 # a marker for which layer the param belongs to
		self.block_idx = 0
	def step(self, projs):
		for group in self.param_groups: 
			for p in group['params']:
				if p.grad is None: 
					continue
				grad = p.grad.data 
				# print(p.size())
				if grad.is_sparse:
					raise RuntimeError("Adam does not support sparse gradients")

				proj = projs[self.block_idx]
				if p.size() in [torch.Size([1536, 512]), torch.Size([512, 512])]:
					self.block_idx = (self.block_idx+1) % 16

				# project only large grads
				if p.size() in [torch.Size([1536, 512]), torch.Size([512, 512])]:
					grad = grad.mm(proj.T) # grad: (dim_out, dim) @ proj.T: (dim, r) -> (dim_out, r)
				
				state = self.state[p]
				
				# State initialization
				if len(state) == 0:
					state["step"] = 0
					# Exponential moving average of gradient values 
					state["exp_avg"] = torch.zeros_like(grad) 
					# Exponential moving average of squared gradient values 
					state["exp_avg_sq"] = torch.zeros_like(grad) 
				
				exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"] 
				beta1, beta2 = group["betas"] 
				
				state["step"] += 1
		
				if self.weight_decay != 0: 
					# grad = grad.add(p.data, alpha=self.weight_decay)
					pass
					# TODO: implement projected weight decay

				# Decay the first and second moment running average coefficient 
				exp_avg.lerp_(grad, 1 - beta1)
				exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
		
				denom = exp_avg_sq.sqrt().add_(group["eps"]) 
				bias_correction1 = 1 - beta1 ** state["step"] 
				bias_correction2 = 1 - beta2 ** state["step"] 

				bias_correction2_sqrt = _dispatch_sqrt(bias_correction2)
				denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(group["eps"])
				step_size = group["lr"] / bias_correction1 
				# step_size = group["lr"] * _dispatch_sqrt(bias_correction2) / bias_correction1 

				# project back and update
				if p.size() in [torch.Size([1536, 512]), torch.Size([512, 512])]:
					p.data.add_(torch.div(exp_avg, denom).mm(proj), alpha=-step_size)
				# # if p.size() in [torch.Size([3072, 768]), torch.Size([2304, 768]), torch.Size([768, 768])]:
				# # 	p.data.add_(exp_avg.div_(denom).mm(proj), alpha=-step_size)
				# # elif p.size() == torch.Size([768, 3072]):
				# # 	p.data.add_(proj.mm(exp_avg.div_(denom)), alpha=-step_size)
				else:
					p.data.addcdiv_(exp_avg, denom, value=-step_size)
				
				# p.data.addcdiv_(exp_avg, denom, value=-step_size)