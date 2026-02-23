"""Spatial operation modules for sparse tensors.

This module provides sparse-compatible pooling, upsampling, and
downsampling layers that work with SparseTensor inputs.

Example:
    >>> import torch
    >>> from sparsetriton import randn
    >>> from sparsetriton.nn.modules import SparsePooling, SparseUpsample
    >>> sp_tensor = randn((16, 16, 16), batch_size=2, nnz=10, channels=4, device="cpu")
    >>> pool = SparsePooling(kernel_size=2, mode="max", stride=2)
    >>> output = pool(sp_tensor)
"""

from typing import Literal
import torch.nn as nn
from sparsetriton import SparseTensor
from sparsetriton.nn.functional import sparse_pooling, sparse_upsample

__all__ = ["SparsePooling", "SparseUpsample", "SparseDownsample"]


class SparsePooling(nn.Module):
    """Sparse pooling layer for downsampling.

    Applies average or max pooling to sparse tensor features.
    Operates on the sparse tensor structure directly.

    Attributes:
        kernel_size: Size of the pooling window
        stride: Stride for the pooling operation
        mode: Pooling mode - "avg" for average pooling, "max" for max pooling
        padding: Padding added to all sides

    Example:
        >>> import torch
        >>> from sparsetriton import randn
        >>> from sparsetriton.nn.modules import SparsePooling
        >>> sp_tensor = randn((16, 16, 16), batch_size=2, nnz=10, channels=4, device="cpu")
        >>> pool = SparsePooling(kernel_size=2, mode="max", stride=2, padding=0)
        >>> output = pool(sp_tensor)
        >>> isinstance(output, SparseTensor)
        True
    """

    def __init__(
        self,
        kernel_size: int,
        mode: Literal["avg", "max"] = "avg",
        stride: int = 1,
        padding: int = 0,
    ):
        """Initialize a sparse pooling layer.

        Args:
            kernel_size: Size of the pooling window (int)
            mode: Pooling mode - "avg" for average pooling, "max" for max pooling
            stride: Stride for the pooling operation
            padding: Padding added to all sides

        Raises:
            ValueError: If mode is not "avg" or "max"

        Example:
            >>> from sparsetriton.nn.modules import SparsePooling
            >>> pool = SparsePooling(kernel_size=2, mode="max", stride=2)
            >>> pool.kernel_size
            2
            >>> pool.mode
            'max'
        """
        super().__init__()
        if mode not in ["avg", "max"]:
            raise ValueError(f"mode must be 'avg' or 'max', got '{mode}'")
        self.kernel_size = kernel_size
        self.stride = stride
        self.mode = mode
        self.padding = padding

    def forward(self, x: SparseTensor) -> SparseTensor:
        """Forward pass of sparse pooling.

        Args:
            x: Input sparse tensor

        Returns:
            SparseTensor: Output sparse tensor after pooling

        Example:
            >>> import torch
            >>> from sparsetriton import randn
            >>> from sparsetriton.nn.modules import SparsePooling
            >>> sp_tensor = randn((16, 16, 16), batch_size=2, nnz=5, channels=4, device="cpu")
            >>> pool = SparsePooling(kernel_size=2, mode="avg")
            >>> output = pool(sp_tensor)
            >>> isinstance(output, SparseTensor)
            True
        """
        return sparse_pooling(x, self.kernel_size, self.padding, self.stride, self.mode)


class SparseUpsample(nn.Module):
    """Sparse upsampling layer.

    Upsamples sparse tensor features by duplicating or interpolating
    coordinates and features.

    Attributes:
        scale_factor: Factor by which to upsample (e.g., 2 for 2x upsampling)

    Example:
        >>> import torch
        >>> from sparsetriton import randn
        >>> from sparsetriton.nn.modules import SparseUpsample
        >>> sp_tensor = randn((16, 16, 16), batch_size=2, nnz=10, channels=4, device="cpu")
        >>> upsample = SparseUpsample(scale_factor=2)
        >>> output = upsample(sp_tensor)
        >>> isinstance(output, SparseTensor)
        True
    """

    def __init__(self, scale_factor: int = None):
        """Initialize a sparse upsampling layer.

        Args:
            scale_factor: Factor by which to upsample

        Example:
            >>> from sparsetriton.nn.modules import SparseUpsample
            >>> upsample = SparseUpsample(scale_factor=2)
            >>> upsample.scale_factor
            2
        """
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, input: SparseTensor) -> SparseTensor:
        """Forward pass of sparse upsampling.

        Args:
            input: Input sparse tensor

        Returns:
            SparseTensor: Output sparse tensor after upsampling

        Example:
            >>> import torch
            >>> from sparsetriton import randn
            >>> from sparsetriton.nn.modules import SparseUpsample
            >>> sp_tensor = randn((16, 16, 16), batch_size=2, nnz=5, channels=4, device="cpu")
            >>> upsample = SparseUpsample(scale_factor=2)
            >>> output = upsample(sp_tensor)
            >>> isinstance(output, SparseTensor)
            True
        """
        return sparse_upsample(input, self.scale_factor)


class SparseDownsample(SparsePooling):
    """Sparse downsampling layer (alias for pooling).

    Convenience alias for pooling with stride=kernel_size and no padding.

    Attributes:
        scale_factor: Downsampling factor (equal to kernel_size and stride)
        mode: Pooling mode - "max" for max pooling, "avg" for average pooling

    Example:
        >>> import torch
        >>> from sparsetriton import randn
        >>> from sparsetriton.nn.modules import SparseDownsample
        >>> sp_tensor = randn((16, 16, 16), batch_size=2, nnz=10, channels=4, device="cpu")
        >>> downsample = SparseDownsample(scale_factor=2, mode="max")
        >>> output = downsample(sp_tensor)
        >>> isinstance(output, SparseTensor)
        True
    """

    def __init__(self, scale_factor: int = None, mode: Literal["max", "avg"] = "max"):
        """Initialize a sparse downsampling layer.

        Args:
            scale_factor: Downsampling factor (used as both kernel_size and stride)
            mode: Pooling mode - "max" for max pooling, "avg" for average pooling

        Example:
            >>> from sparsetriton.nn.modules import SparseDownsample
            >>> downsample = SparseDownsample(scale_factor=2, mode="max")
            >>> downsample.kernel_size
            2
            >>> downsample.stride
            2
            >>> downsample.padding
            0
        """
        super().__init__(kernel_size=scale_factor, stride=scale_factor, mode=mode, padding=0)

    def forward(self, input: SparseTensor) -> SparseTensor:
        """Forward pass of sparse downsampling.

        Args:
            input: Input sparse tensor

        Returns:
            SparseTensor: Output sparse tensor after downsampling

        Example:
            >>> import torch
            >>> from sparsetriton import randn
            >>> from sparsetriton.nn.modules import SparseDownsample
            >>> sp_tensor = randn((16, 16, 16), batch_size=2, nnz=5, channels=4, device="cpu")
            >>> downsample = SparseDownsample(scale_factor=2, mode="max")
            >>> output = downsample(sp_tensor)
            >>> isinstance(output, SparseTensor)
            True
        """
        return super().forward(input)
