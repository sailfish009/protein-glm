import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import EinMix

# helper functions

def exists(val):
    return val is not None

# attention

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.to_q = EinMix('b n d -> b h n dh', weight_shape = 'h d dh', d = dim, dh = dim_head, h = heads)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        """
        einstein notation:
        b - batch
        h - heads
        d - feature dimension
        n, i, j - sequence (main, source, target)
        """

        q = self.to_q(x)
        k, v = self.to_kv(x).chunk(2, dim = -1)

        q = q * self.scale
        sim = einsum('b h i d, b j d -> b h i j', q, k)

        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# classes

class ProteinGLM(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(PreNormResidual(dim, Attention(dim = dim, dim_head = dim_head, heads = heads)))

    def forward(self, x):
        for attn in self.layers:
            x = attn(x)

        return x
