import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Sequence, Optional, Tuple, Any
from einops import rearrange, repeat, reduce

class Quantization(nn.Module):
    def __init__(self, codes: int, dim: int):
        super().__init__()
        self.codes = codes
        self.dim = dim

        self.temperature = 1.0
        self.norm = nn.LayerNorm(self.dim)
        self.embedding = torch.zeros(self.codes, self.dim)
        nn.init.orthogonal_(self.embedding)

    def forward(self, x, return_codes=False):
        x = self.norm(x)
        flattened = x.reshape(-1, self.dim)
        distances = flattened.pow(2).sum(1, keepdim=True)
            - 2 * flattened @ self.embedding.T
            + self.embedding.pow(2).sum(1, keepdim=True).T
        indices = F.gumbel_softmax(-distances, tau=self.temperature, hard=True, dim=-1)
        if return_codes:
            return indices.reshape(x.shape[:-1])
        onehot = F.onehot(indices, self.codes)
        onehot = onehot.reshape(x.shape[:-1] + (self.codes,))
        quantized = F.embedding(indices, self.embedding)
        quantized = quantized.reshape(x.shape)
        return quantized

class Residual(nn.Module):
    def __init__(self, fns: Sequential[Any]):
        """
        In the constructor we stash way the modules that'll be called along
        the residual branch. This is just for convenience.
        """
        super().__init__()
        self.fns = fns

    def forward(self, x):
        return x + sum(fn(x) for fn in self.fns)

class Rotary(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        inv_freq = 1. / torch.logspace(1.0, 100.0, dim // 2)
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, x):
        n = x.shape[-3] # [... sequence, head, dim]
        t = torch.linspace(-1, 1, n).type_as(self.inv_freq)
        freqs = einsum('n , c -> n c', t, self.inv_freq) # c = d / 2
        posemb = repeat(freqs, "n c -> () n () (2 c)")
        out = x * posemb.cos()
        odds, evens = rearrange(x, '... (j c) -> ... j c', j = 2).unbind(dim = -2)
        rotated = torch.cat((-evens, odds), dim = -1)
        out += rotated * posemb.sin()
        return out

class Block(nn.Module):
    def __init__(self, heads: int, head_dim: int, rank: int, compression: Sequence[int], axis: int = 1):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.hidden_dim = self.heads * self.head_dim
        self.rank = rank
        self.compression = compression

        self.ln = nn.LayerNorm(self.hidden_dim)
        self.in_proj = nn.Linear(self.hidden_dim, self.hidden_dim * 7, False)
        self.out_proj = nn.Linear(self.hidden_dim * 5, self.hidden_dim, True)
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        x = self.ln(x)
        x = self.in_proj(x)
        q, k, v, p = torch.split(x, [
                                   self.hidden_dim,
                                   self.hidden_dim,
                                   self.hidden_dim,
                                   self.hidden_dim * 4,
                                   ], -1)
        (q, p) = map(lambda x: F.interpolate(x, scale_factor=self.compression, mode='linear'), (q, p))
        (q, k, v) = map(lambda x: x.transpose(self.axis, -2), (q, k, v))
        (q, k, v) = map(lambda x: rearrange(x, "... (h d) -> ... h d", h=self.heads), (q, k, v))
        (q, k) = map(lambda x: self.rotary(x), (q, k))
        a = einsum("... i h d, ... j h d -> ... h i j", q, k) * (self.head_dim ** -0.5)
        a = F.softmax(a, dim=-1)
        o = einsum("... h i j, ... j h d -> ... i h d", a, v)
        o = rearrange(o, "... h d -> ... (h d)")
        o = o.transpose(self.axis, -2)
        p = F.gelu(p)
        o = torch.cat([o, p], -1)
        x = self.out_proj(o)
        return x

class Encoder(nn.Module):
    def __init__(self, depth: int, rank: int, compression: Sequence[int]):
        assert len(compression) <= 3, "Only inputs up to 3D are supported"
        super().__init__()
        scales = [compression for depth in range(depth)] # This will not work. Need a strategy
        self.net = nn.Sequential([Residual([Block(8, 64, scale, axis=j) for j in range(rank)]) for (i, scale) in enumerate(scales)])

    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, depth: int, rank: int, decompression: Sequence[int]):
        assert len(decompression) <= 3, "Only outputs up to 3D are supported"
        super().__init__()
        scales = [decompression for depth in range(depth)] # This will not work. Need a strategy
        self.net = nn.Sequential([Residual([Block(8, 64, scale, axis=j) for j in range(rank)]) for (i, scale) in enumerate(scales)])

    def forward(self, x):
        return self.net(x)