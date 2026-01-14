import torch
from tensor import SparseTensor



class SubmSparseConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight = torch.nn.Parameter(torch.randn(kernel_size, kernel_size, kernel_size, in_channels, out_channels))

    def forward(self, input: SparseTensor) -> SparseTensor:
        # Placeholder for actual implementation
        out_feats = torch.randn(input.feats.shape[0], self.out_channels)
        return SparseTensor(out_feats, input.coords, stride=input.s)
    


class SparseConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding