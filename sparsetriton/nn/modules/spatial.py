from re import S
import torch.nn as nn
from sparsetriton import SparseTensor
from sparsetriton.nn.functional import sparse_pooling, sparse_upsample, sparse_downsample
from typing import *

class SparsePooling(nn.Module):
    """
    Placeholder for sparse pooling module.
    """
    def __init__(self, kernel_size:int, mode: Literal["avg", "max"] = "avg", stride:int=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.mode = mode

    def forward(self, x: SparseTensor) -> SparseTensor:
        return sparse_pooling(x, self.kernel_size, self.stride, self.padding, self.mode)

class SparseUpsample(nn.Module):
    """
    Placeholder for sparse upsample module.
    """
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        # TODO: Initialize any parameters if needed

    def forward(self, input: SparseTensor):
        return sparse_upsample(input, self.size, self.scale_factor, self.mode, self.align_corners)

class SparseDownsample(nn.Module):
    """
    Placeholder for sparse downsample module.
    """
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        # TODO: Initialize any parameters if needed

    def forward(self, input: SparseTensor):
        return sparse_downsample(input, self.size, self.scale_factor, self.mode, self.align_corners)
