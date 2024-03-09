import torch
import torch.nn as nn


# from models.vit_factorized import ViT
# model = ViT(
# image_size = 32,
# patch_size = 4,
# num_classes = 10,
# dim = 512,
# depth = 6,
# heads = 8,
# mlp_dim = 3072,)

# from models.vit_factorized import ViT_factorized
# model = ViT_factorized(
# image_size = 32,
# patch_size = 4,
# num_classes = 10,
# dim = 512,
# depth = 6,
# heads = 8,
# mlp_dim = 3072,
# rs = [None] * 5 + [40]
# )

from models.vit_factorized import ViT_overparametrized
model = ViT_overparametrized(
image_size = 32,
patch_size = 4,
num_classes = 10,
dim = 512,
depth = 6,
heads = 8,
mlp_dim = 3072,
)
# test inference

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
from torchstat import stat
# params
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("#Params:", params)

# FLOPs
inputs = torch.randn(128,3, 32, 32).to(device)
from torchprofile import profile_macs
# stat(model, (128, 28, 28))
macs = profile_macs(model, inputs)
print("FLOPs", macs)

# memory
from torch.cuda import memory_allocated
loss = nn.CrossEntropyLoss()
loss = model(inputs)
max_memory_allocated = torch.cuda.max_memory_allocated(device)

print(f"Memory: {max_memory_allocated / 1024**2:.2f} MB")