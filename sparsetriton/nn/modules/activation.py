from torch import nn
from sparsetriton import SparseTensor

__all__ = ["ReLU", "LeakyReLU", "SiLU", "GELU", "Sigmoid", "Tanh"]

class ReLU(nn.ReLU):
    def forward(self, input: SparseTensor) -> SparseTensor:
        if isinstance(input, SparseTensor):
            return input.replace(super().forward(input.F))
        return super().forward(input)

class LeakyReLU(nn.LeakyReLU):
    def forward(self, input: SparseTensor) -> SparseTensor:
        if isinstance(input, SparseTensor):
            return input.replace(super().forward(input.F))
        return super().forward(input)

class SiLU(nn.SiLU):
    def forward(self, input: SparseTensor) -> SparseTensor:
        if isinstance(input, SparseTensor):
            return input.replace(super().forward(input.F))
        return super().forward(input)

class GELU(nn.GELU):
    def forward(self, input: SparseTensor) -> SparseTensor:
        if isinstance(input, SparseTensor):
            return input.replace(super().forward(input.F))
        return super().forward(input)

class Sigmoid(nn.Sigmoid):
    def forward(self, input: SparseTensor) -> SparseTensor:
        if isinstance(input, SparseTensor):
            return input.replace(super().forward(input.F))
        return super().forward(input)

class Tanh(nn.Tanh):
    def forward(self, input: SparseTensor) -> SparseTensor:
        if isinstance(input, SparseTensor):
            return input.replace(super().forward(input.F))
        return super().forward(input)
    
    