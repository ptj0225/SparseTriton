"""
Building blocks for sparse neural networks.

This module provides common building blocks used in sparse convolution networks
for autonomous driving and 3D point cloud processing.
"""

import numpy as np
from typing import Union, List, Tuple
import torch
from torch import nn

from sparsetriton import SparseTensor
from sparsetriton.nn.modules import SparseConv3D, SubMConv3D, SparseConvTransposed3D
from sparsetriton.nn.modules.norm import SparseBatchNorm
from sparsetriton.nn.modules.activation import ReLU

__all__ = ["SparseConvBlock", "SparseConvTransposeBlock", "SparseResBlock"]


class SparseConvBlock(nn.Sequential):
    """Sparse convolution block with batch normalization and ReLU activation.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of convolution kernel
        stride: Stride for downsampling
        dilation: Dilation factor
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, List[int], Tuple[int, ...]],
        stride: Union[int, List[int], Tuple[int, ...]] = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__(
            SubMConv3D(
                in_channels, out_channels, kernel_size, stride=stride, dilation=dilation
            ),
            SparseBatchNorm(out_channels),
            ReLU(inplace=True),
        )


class SparseConvTransposeBlock(nn.Sequential):
    """Sparse transposed convolution block with batch normalization and ReLU activation.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of convolution kernel
        stride: Stride for upsampling
        dilation: Dilation factor
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, List[int], Tuple[int, ...]],
        stride: Union[int, List[int], Tuple[int, ...]] = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__(
            SparseConvTransposed3D(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                dilation=dilation,
            ),
            SparseBatchNorm(out_channels),
            ReLU(inplace=True),
        )


class SparseResBlock(nn.Module):
    """Sparse residual block.

    Implements a residual connection for sparse convolution networks.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of convolution kernel
        stride: Stride for downsampling
        dilation: Dilation factor
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, List[int], Tuple[int, ...]],
        stride: Union[int, List[int], Tuple[int, ...]] = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.main = nn.Sequential(
            SubMConv3D(
                in_channels, out_channels, kernel_size, dilation=dilation, stride=stride
            ),
            SparseBatchNorm(out_channels),
            ReLU(inplace=True),
            SubMConv3D(out_channels, out_channels, kernel_size, dilation=dilation),
            SparseBatchNorm(out_channels),
        )

        if in_channels != out_channels or np.prod(stride) != 1:
            self.shortcut = nn.Sequential(
                SubMConv3D(in_channels, out_channels, 1, stride=stride),
                SparseBatchNorm(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = ReLU(inplace=True)

    def forward(self, x: SparseTensor) -> SparseTensor:
        return self.relu(self.main(x) + self.shortcut(x))
