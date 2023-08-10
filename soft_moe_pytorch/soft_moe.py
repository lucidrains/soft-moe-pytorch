import torch
from torch.nn import Module
import torch.nn.functional as F
from torch import nn, einsum, Tensor

from einops import rearrange

class SoftMoE(Module):
    def __init__(
        self,
        dim
    ):
        super().__init__()

    def forward(self, x):
        return x
