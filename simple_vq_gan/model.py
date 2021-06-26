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
    def __init__(self):
        pass

    def forward(self, x):
        pass

class Rotary(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass

class Block(nn.Module):
    def __init__(self, compression: int = 1, decompression: int = 1):
        pass

    def forward(self, x):
        pass

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
