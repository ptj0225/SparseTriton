from re import S
import torch.nn as nn
from sparsetriton import SparseTensor
from sparsetriton.nn.functional import sparse_pooling, sparse_upsample, sparse_downsample
from typing import *

class SparsePooling(nn.Module):
    """
    Placeholder for sparse pooling module.
    """
    def __init__(self, kernel_size:int, mode: Literal["avg", "max"] = "avg", stride:int=1, padding:int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.mode = mode
        self.padding = padding

    def forward(self, x: SparseTensor) -> SparseTensor:
        return sparse_pooling(x, self.kernel_size, self.padding ,self.stride, self.mode)

class SparseUpsample(nn.Module):
    """
    Placeholder for sparse upsample module.
    """
    def __init__(self, size=None, scale_factor=None, align_corners=None):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.align_corners = align_corners
        # TODO: Initialize any parameters if needed

    def forward(self, input: SparseTensor):
        return sparse_upsample(input, self.size, self.scale_factor)

class SparseDownsample(nn.Module): 
    """
    Placeholder for sparse downsample module.
    """
    def __init__(self, size=None, scale_factor=None, align_corners=None):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.align_corners = align_corners
        # TODO: Initialize any parameters if needed

    def forward(self, input: SparseTensor):
        return sparse_downsample(input, self.scale_factor)
