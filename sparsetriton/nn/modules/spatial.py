from ast import mod
from re import S
import torch.nn as nn
from sparsetriton import SparseTensor
from sparsetriton.nn.functional import sparse_pooling, sparse_upsample
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
    def __init__(self, scale_factor=None):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, input: SparseTensor):
        return sparse_upsample(input, self.scale_factor)

class SparseDownsample(SparsePooling): 
    """
    Placeholder for sparse downsample module.
    """
    def __init__(self, scale_factor=None, mode=Literal['max', 'avg']):
        super().__init__(kernel_size=scale_factor, stride=scale_factor, mode=mode, padding=0)
    
    def forward(self, input: SparseTensor):
        return super().forward(input)

