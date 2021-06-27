import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Sequence, Optional, Tuple, Any
from einops import rearrange, repeat, reduce

class Quantize(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass

class Residual(nn.Module):
    def __init__(self, residuals: Sequential[Any]):
        """
        In the constructor we stash way the modules that'll be called along
        the residual branch. This is just for convenience.
        """
        super(Residual, self).__init__()
        self.residuals = residuals

    def forward(self, x):
        return x + sum(residual(x) for residual in self.residuals)

class Rotary(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass

class Block(nn.Module):
    def __init__(self, rank: int, compression: Sequence[int], axis: int = 1):
        super(Block, self).__init__()
        self.heads = config.heads
        self.head_dim = config.head_dim
        self.hidden_dim = self.heads * self.head_dim
        self.rank = rank
        self.compression = compression

        self.ln = nn.LayerNorm(self.hidden_dim)
        self.in_proj = nn.Linear(self.hidden_dim, self.hidden_dim * 7, False)
        self.out_proj = nn.Linear(self.hidden_dim * 5, self.hidden_dim, True)
        self.rotary = Rotary()

    def forward(self, x):
        # This function needs some work to make it agnostic to dimension
        x = self.ln(x)
        x = self.in_proj(x)
        q, k, v, p = torch.split(x, [
                                   self.hidden_dim,
                                   self.hidden_dim,
                                   self.hidden_dim,
                                   self.hidden_dim * 4,
                                   ], -1)
        (k, v, p) = map(lambda x: F.interpolate(x, scale_factor=self.compression, mode='linear'), (k, v, p))
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
    def __init__(self, depth: int, compression: Sequence[int]):
        assert len(compression) <= 3, "Only inputs up to 3D are supported"
        self.depth = depth
        pass

    def forward(self, x):
        pass

class Decoder(nn.Module):
    def __init__(self, depth: int, decompression: Sequence[int]):
        assert len(decompression) <= 3, "Only outputs up to 3D are supported"
        self.depth = depth
        pass

    def forward(self, x):
        pass
