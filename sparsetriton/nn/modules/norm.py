import torch
from torch import nn
from sparsetriton import SparseTensor
from sparsetriton.nn.functional.norm import sparse_batch_norm

__all__ = ["SparseBatchNorm", "SparseLayerNorm"]

class SparseBatchNorm(nn.BatchNorm1d):
    def forward(self, input: SparseTensor) -> SparseTensor:
        return input.replace((input.F, super().forward))

class SparseLayerNorm(nn.LayerNorm):
    def forward(self, input: SparseTensor) -> SparseTensor:
        if isinstance(input, SparseTensor):
            return input.replace(super().forward(input.F))
        return super().forward(input)
