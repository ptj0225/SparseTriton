"""
Sparse ResNet architectures for 3D point cloud processing.

This module implements sparse ResNet architectures commonly used in
autonomous driving for LiDAR point cloud processing.
"""

from typing import List, Tuple, Union

from torch import nn

from sparsetriton import SparseTensor
from sparsetriton.nn.modules.blocks import SparseConvBlock, SparseResBlock

__all__ = ["SparseResNet21D"]


class SparseResNet(nn.ModuleList):
    """Base class for sparse ResNet architectures.

    Args:
        blocks: List of (num_blocks, out_channels, kernel_size, stride) tuples
        in_channels: Number of input channels
        width_multiplier: Multiplier for output channels
    """

    def __init__(
        self,
        blocks: List[
            Tuple[int, int, Union[int, Tuple[int, ...]], Union[int, Tuple[int, ...]]]
        ],
        *,
        in_channels: int = 4,
        width_multiplier: float = 1.0,
    ) -> None:
        super().__init__()
        self.blocks = blocks
        self.in_channels = in_channels
        self.width_multiplier = width_multiplier

        for num_blocks, out_channels, kernel_size, stride in blocks:
            out_channels = int(out_channels * width_multiplier)
            blocks = []
            for index in range(num_blocks):
                if index == 0:
                    blocks.append(
                        SparseConvBlock(
                            in_channels,
                            out_channels,
                            kernel_size,
                            stride=stride,
                        )
                    )
                else:
                    blocks.append(
                        SparseResBlock(
                            in_channels,
                            out_channels,
                            kernel_size,
                        )
                    )
                in_channels = out_channels
            self.append(nn.Sequential(*blocks))

    def forward(self, x: SparseTensor) -> List[SparseTensor]:
        """Forward pass.

        Args:
            x: Input sparse tensor

        Returns:
            List of sparse tensors from each stage
        """
        outputs = []
        for module in self:
            x = module(x)
            outputs.append(x)
        return outputs


class SparseResNet21D(SparseResNet):
    """SparseResNet21D backbone for autonomous driving.

    This is a 21-layer ResNet architecture commonly used in LiDAR-based
    3D object detection models like SECOND, PointPillars, etc.

    Architecture:
        - Stage 1: 3 blocks, 16 channels, kernel_size=3, stride=1
        - Stage 2: 3 blocks, 32 channels, kernel_size=3, stride=2
        - Stage 3: 3 blocks, 64 channels, kernel_size=3, stride=2
        - Stage 4: 3 blocks, 128 channels, kernel_size=3, stride=2
        - Stage 5: 1 block, 128 channels, kernel_size=(1,3,1), stride=(1,2,1)

    Args:
        in_channels: Number of input channels (default: 4 for x,y,z,intensity)
        width_multiplier: Multiplier for output channels (default: 1.0)
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(
            blocks=[
                (3, 16, 3, 1),
                (3, 32, 3, 2),
                (3, 64, 3, 2),
                (3, 128, 3, 2),
                (1, 128, 3, 2),
            ],
            **kwargs,
        )
