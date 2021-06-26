import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Sequence, Optional, Tuple
from einops import rearrange, repeat, reduce

class Quantize(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass

class Residual(nn.Module):
    def __init__(self, residual):
        """
        In the constructor we stash way the module that'll be called along
        the residual branch. This is just for convenience.
        """
        super(Residual, self).__init__()
        self.residual = residual

    def forward(self, x):
        return x + self.residual(x)

class Rotary(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass

class Block(nn.Module):
    def __init__(self, rank: int, compression: int = 1, decompression: int = 1, axis: int = 1):
        super(Block, self).__init__()
        self.heads = config.heads
        self.head_dim = config.head_dim
        self.hidden_dim = self.heads * self.head_dim
        self.rank = rank
        self.compression = compression
        self.decompression = decompression 

        self.ln = nn.LayerNorm(self.hidden_dim)
        self.in_proj = nn.Linear(self.hidden_dim, self.hidden_dim * 3, False)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim, True)
        self.rotary = Rotary()

    def forward(self, x):
        # This function needs some work to make it agnostic to dimension
        x = x.transpose(self.axis, -2)
        x = self.ln(x)
        x = self.in_proj(x)
        q, k, v = torch.split(x, [
                                   self.hidden_dim,
                                   self.hidden_dim,
                                   self.hidden_dim,
                                   ], -1)
        (q, k, v) = map(lambda x: rearrange(x, "... (h d) -> ... h d", h=self.heads), (q, k, v))
        # Use ConvNd to compress or decompress kv
        (q, k) = map(lambda x: self.rotary(x), (q, k))
        a = einsum("... i h d, ... j h d -> ... h i j", q, k) * (self.head_dim ** -0.5)
        a = F.softmax(a, dim=-1)
        o = einsum("... h i j, ... j h d -> ... i h d", a, v)
        o = rearrange(o, "... h d -> ... (h d)")
        x = self.out_proj(o)
        x = x.transpose(self.axis, -2)
        return x

class Encoder(nn.Module):
    def __init__(self, compression: Sequence[int]):
        assert len(compression) <= 3, "Only inputs up to 3D are supported"
        pass

    def forward(self, x):
        pass

class Decoder(nn.Module):
    def __init__(self, decompression: Sequence[int]):
        assert len(decompression) <= 3, "Only outputs up to 3D are supported"
        pass

    def forward(self, x):
        pass
