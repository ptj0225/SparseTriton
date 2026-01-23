from torch import nn
from sparsetriton import SparseTensor

__all__ = ["SparseLinear"]

class SparseLinear(nn.Linear):
    def forward(self, input: SparseTensor) -> SparseTensor:
        if isinstance(input, SparseTensor):
            return input.replace(super().forward(input.F))
        return super().forward(input)
