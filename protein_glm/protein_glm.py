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

def FeedForward(dim, mult = 4):
    inner_dim = int(mult * dim)
    return nn.Sequential(
        nn.Linear(dim, inner_dim),
        nn.GELU(),
        nn.Linear(inner_dim, dim)
    )

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

    def forward(
        self,
        x,
        causal_boundary_indices = None
    ):
        """
        einstein notation:
        b - batch
        h - heads
        d - feature dimension
        n, i, j - sequence (main, source, target)
        """
        n, device = x.shape[1], x.device

        q = self.to_q(x)
        k, v = self.to_kv(x).chunk(2, dim = -1)

        q = q * self.scale
        sim = einsum('b h i d, b j d -> b h i j', q, k)

        i, j = sim.shape[-2:]
        max_neg_value = -torch.finfo(sim.dtype).max
        causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)

        if exists(causal_boundary_indices):
            seq_range = torch.arange(n, device = device)
            causal_mask_after = rearrange(seq_range, 'j -> 1 1 1 j') >= rearrange(causal_boundary_indices, 'b -> b 1 1 1')
            causal_mask = causal_mask & causal_mask_after

        sim = sim.masked_fill(causal_mask, max_neg_value)

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
        num_tokens,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
    ):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNormResidual(dim, Attention(dim = dim, dim_head = dim_head, heads = heads)),
                PreNormResidual(dim, FeedForward(dim = dim, mult = ff_mult)),
            ]))

        self.norm = nn.LayerNorm(dim)
        self.to_logits = nn.Linear(dim, num_tokens, bias = False)

    def forward(
        self,
        x,
        causal_boundary_indices = None # the sequence indices, per batch, at which autoregressive mask starts (before the index would be full bidirectional attention)
    ):
        for attn, ff in self.layers:
            x = attn(x, causal_boundary_indices = causal_boundary_indices)
            x = ff(x)

        x = self.norm(x)
        return self.to_logits(x)
