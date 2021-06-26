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

class Block(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass

class Encoder(nn.Module):
    def __init__(self, compression: Sequence[int]):
        pass

    def forward(self, x):
        pass

class Decoder(nn.Module):
    def __init__(self, decompression: Sequence[int]):
        pass

    def forward(self, x):
        pass
