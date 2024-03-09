# https://github.com/lucidrains/mlp-mixer-pytorch/blob/main/mlp_mixer_pytorch/mlp_mixer_pytorch.py
from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce

pair = lambda x: x if isinstance(x, tuple) else (x, x)

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout)
    )

def FeedForward_factorized(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear, rank = None, deep=False):
    inner_dim = int(dim * expansion_factor)
    if rank is None:
        return nn.Sequential(
            dense(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            dense(inner_dim, dim),
            nn.Dropout(dropout)
        )
    elif not deep:
        return nn.Sequential(
            dense(dim, rank),
            dense(rank, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            dense(inner_dim, rank),
            dense(rank, dim),
            nn.Dropout(dropout)
        )
    else:
        return nn.Sequential(
            dense(dim, rank),
            dense(rank, rank),
            dense(rank, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            dense(inner_dim, rank),
            dense(rank, rank),
            dense(rank, dim),
            nn.Dropout(dropout)
        )

        


def MLPMixer(*, image_size, channels, patch_size, dim, depth, num_classes, expansion_factor = 4, expansion_factor_token = 0.5, dropout = 0.):
    image_h, image_w = pair(image_size)
    assert (image_h % patch_size) == 0 and (image_w % patch_size) == 0, 'image must be divisible by patch size'
    num_patches = (image_h // patch_size) * (image_w // patch_size)
    chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear

    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
        nn.Linear((patch_size ** 2) * channels, dim),
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)), # patch mixing
            PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, chan_last)) # channel mixing
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        Reduce('b n c -> b c', 'mean'),
        nn.Linear(dim, num_classes)
    )

def MLPMixer_factorized(*, image_size, channels, patch_size, dim, depth, num_classes, expansion_factor = 4, expansion_factor_token = 0.5, dropout = 0., rs_patch = None, rs_channel = None, deep=False):
    image_h, image_w = pair(image_size)
    assert (image_h % patch_size) == 0 and (image_w % patch_size) == 0, 'image must be divisible by patch size'
    num_patches = (image_h // patch_size) * (image_w // patch_size)
    chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear

    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
        nn.Linear((patch_size ** 2) * channels, dim),
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward_factorized(num_patches, expansion_factor, dropout, chan_first, rs_patch[l], deep)), # patch mixing
            PreNormResidual(dim, FeedForward_factorized(dim, expansion_factor_token, dropout, chan_last, rs_channel[l], deep)) # channel mixing
        ) for l in range(depth)],
        nn.LayerNorm(dim),
        Reduce('b n c -> b c', 'mean'),
        nn.Linear(dim, num_classes)
    )