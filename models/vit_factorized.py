# from: https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit.py

import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange

####################################### basic blocks #######################################

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(patches, temperature = 10000, dtype = torch.float32):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :] 
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
    return pe.type(dtype)

# classes
    # class FeedForward_op(nn.Module):
    #     def __init__(self, dim, hidden_dim, r=None):
    #         super().__init__()
    #         self.net = nn.Sequential(
    #             nn.LayerNorm(dim),
    #             nn.Linear(dim, hidden_dim),
    #             nn.GELU(),
    #             nn.Linear(hidden_dim, hidden_dim),
    #             nn.Linear(hidden_dim, hidden_dim),
    #             nn.Linear(hidden_dim, dim),
    #         )

    #     def forward(self, x):
    #         return self.net(x)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, r=None, deep=False):
        super().__init__()
        if isinstance(r, int) and r != 0:
            if deep:
                self.net = nn.Sequential(
                    nn.Linear(dim, r),
                    nn.Linear(r, r),
                    nn.Linear(r, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, r),
                    nn.Linear(r, r),
                    nn.Linear(r, dim),
                )
            else:
                self.net = nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, r),
                    nn.Linear(r, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, r),
                    nn.Linear(r, dim),
                )
        else:
            self.net = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, dim),
            )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, r=None, deep=False):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        if isinstance(r, int):
            if deep:
                self.to_q = nn.Sequential(nn.Linear(dim, r, bias = False),nn.Linear(r, r, bias=False), nn.Linear(r, inner_dim, bias = False),)
                self.to_k = nn.Sequential(nn.Linear(dim, r, bias = False),nn.Linear(r, r, bias=False), nn.Linear(r, inner_dim, bias = False),)
                self.to_v = nn.Sequential(nn.Linear(dim, r, bias = False),nn.Linear(r, r, bias=False), nn.Linear(r, inner_dim, bias = False),)
            else:
                self.to_q = nn.Sequential(nn.Linear(dim, r, bias = False), nn.Linear(r, inner_dim, bias = False),)
                self.to_k = nn.Sequential(nn.Linear(dim, r, bias = False), nn.Linear(r, inner_dim, bias = False),)
                self.to_v = nn.Sequential(nn.Linear(dim, r, bias = False), nn.Linear(r, inner_dim, bias = False),)
        else:
            self.to_q = nn.Linear(dim, inner_dim, bias = False)
            self.to_k = nn.Linear(dim, inner_dim, bias = False)
            self.to_v = nn.Linear(dim, inner_dim, bias = False)
        # self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = (self.to_q(x), self.to_k(x), self.to_v(x))
        # qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# class FeedForward_factorized(nn.Module):
#     def __init__(self, dim, hidden_dim, r):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.LayerNorm(dim),
#             nn.Linear(dim, r),
#             nn.Linear(r, hidden_dim),
#             # nn.Linear(dim, hidden_dim),
#             nn.GELU(),
#             nn.Linear(hidden_dim, r),
#             nn.Linear(r, dim),
#         )
#     def forward(self, x):
#         return self.net(x)

# class Attention_Factorized(nn.Module):
#     def __init__(self, dim, r, heads = 8, dim_head = 64):
#         super().__init__()
#         inner_dim = dim_head *  heads
#         self.heads = heads
#         self.scale = dim_head ** -0.5
#         self.norm = nn.LayerNorm(dim)

#         self.attend = nn.Softmax(dim = -1)

#         # self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
#         self.to_q = nn.Sequential(nn.Linear(dim, r), nn.Linear(r, inner_dim),)
#         self.to_k = nn.Sequential(nn.Linear(dim, r), nn.Linear(r, inner_dim),)
#         self.to_v = nn.Sequential(nn.Linear(dim, r), nn.Linear(r, inner_dim),)
#         # self.to_qkv = nn.Sequential(nn.Linear(dim, r), nn.Linear(r, inner_dim * 3),)
#         # self.to_out = nn.Sequential(nn.Linear(inner_dim, r), nn.Linear(r, dim),)
#         self.to_out = nn.Linear(inner_dim, dim, bias = False)

#     def forward(self, x):
#         x = self.norm(x)

#         # qkv = self.to_qkv(x).chunk(3, dim = -1)
#         qkv = [self.to_q(x), self.to_k(x), self.to_v(x)]
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

#         dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

#         attn = self.attend(dots)

#         out = torch.matmul(attn, v)
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, rs_ff, rs_attn, deep):
        super().__init__()
        self.layers = nn.ModuleList([])

        for l in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, r = rs_attn[l], deep=deep),
                FeedForward(dim, mlp_dim, r = rs_ff[l], deep=deep)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class Transformer_op(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])

        for l in range(depth-1):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
        self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward_op(dim, mlp_dim)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

########################################## ViTs ##########################################
class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, init_scale=0.1):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, [None] * depth)

        self.to_latent = nn.Identity()
        self.linear_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        # orthogonal initialization of all linear layer
        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.orthogonal_(m.weight, gain=init_scale)

    def forward(self, img):
        *_, h, w, dtype = *img.shape, img.dtype

        x = self.to_patch_embedding(img)
        pe = posemb_sincos_2d(x)
        x = rearrange(x, 'b ... d -> b (...) d') + pe

        x = self.transformer(x)
        # x = x.mean(dim = 1)
        x = x[:, 0]

        x = self.to_latent(x)
        return self.linear_head(x)

class ViT_factorized(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, init_scale=0.1, rs_ff=None, rs_attn=None, deep=False):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, rs_ff, rs_attn, deep)

        self.to_latent = nn.Identity()
        self.linear_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.orthogonal_(m.weight, gain=init_scale)

    def forward(self, img):
        *_, h, w, dtype = *img.shape, img.dtype

        x = self.to_patch_embedding(img)
        pe = posemb_sincos_2d(x)
        x = rearrange(x, 'b ... d -> b (...) d') + pe

        x = self.transformer(x)
        # x = x.mean(dim = 1)
        x = x[:, 0]

        x = self.to_latent(x)
        return self.linear_head(x)

class ViT_overparametrized(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, rs = None, init_scale=0.1):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.transformer = Transformer_op(dim, depth, heads, dim_head, mlp_dim)

        self.to_latent = nn.Identity()
        self.linear_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.orthogonal_(m.weight, gain=init_scale)

    def forward(self, img):
        *_, h, w, dtype = *img.shape, img.dtype

        x = self.to_patch_embedding(img)
        pe = posemb_sincos_2d(x)
        x = rearrange(x, 'b ... d -> b (...) d') + pe

        x = self.transformer(x)
        # x = x.mean(dim = 1)
        x = x[:, 0]

        x = self.to_latent(x)
        return self.linear_head(x)
