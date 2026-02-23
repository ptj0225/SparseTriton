"""Normalization layers for sparse tensors.

This module provides sparse-compatible normalization layers including
batch normalization and layer normalization.

Example:
    >>> import torch
    >>> from sparsetriton import randn
    >>> from sparsetriton.nn.modules import SparseBatchNorm, SparseLayerNorm
    >>> sp_tensor = randn((16, 16, 16), batch_size=2, nnz=10, channels=4, device="cpu")
    >>> bn = SparseBatchNorm(num_features=4)
    >>> output = bn(sp_tensor)
    >>> print(output.feats.shape[1])
    4
"""

import torch
from torch import nn
from sparsetriton import SparseTensor
from sparsetriton.nn.functional.norm import sparse_batch_norm

__all__ = ["SparseBatchNorm", "SparseLayerNorm"]


class SparseBatchNorm(nn.BatchNorm1d):
    """Sparse-compatible batch normalization layer.

    Applies batch normalization to the features of a SparseTensor
    while preserving the coordinate structure. Normalizes across
    the batch dimension.

    Attributes:
        num_features: Number of features (channels)
        eps: Small value for numerical stability
        momentum: Momentum value for running stats
        affine: Whether to learn affine parameters (scale and shift)
        track_running_stats: Whether to track running statistics

    Example:
        >>> import torch
        >>> from sparsetriton import randn
        >>> from sparsetriton.nn.modules import SparseBatchNorm
        >>> sp_tensor = randn((16, 16, 16), batch_size=2, nnz=10, channels=4, device="cpu")
        >>> bn = SparseBatchNorm(num_features=4)
        >>> output = bn(sp_tensor)
        >>> output.feats.shape == sp_tensor.feats.shape
        True
        >>> torch.allclose(output.coords, sp_tensor.coords)
        tensor(True)
    """

    def forward(self, input: SparseTensor) -> SparseTensor:
        """Forward pass of sparse batch normalization.

        Args:
            input: Input sparse tensor

        Returns:
            SparseTensor: Output sparse tensor with normalized features

        Example:
            >>> import torch
            >>> from sparsetriton import randn
            >>> from sparsetriton.nn.modules import SparseBatchNorm
            >>> sp_tensor = randn((16, 16, 16), batch_size=2, nnz=5, channels=4, device="cpu")
            >>> bn = SparseBatchNorm(num_features=4)
            >>> output = bn(sp_tensor)
            >>> output.feats.shape == sp_tensor.feats.shape
            True
        """
        # Call parent forward with features, wrap result in tuple for functional API
        return input.replace(super().forward(input.F))


class SparseLayerNorm(nn.LayerNorm):
    """Sparse-compatible layer normalization layer.

    Applies layer normalization to the features of a SparseTensor
    while preserving the coordinate structure. Normalizes across
    the feature dimension.

    Attributes:
        normalized_shape: Shape of input (without batch dim)
        eps: Small value for numerical stability
        elementwise_affine: Whether to learn elementwise affine parameters

    Example:
        >>> import torch
        >>> from sparsetriton import randn
        >>> from sparsetriton.nn.modules import SparseLayerNorm
        >>> sp_tensor = randn((16, 16, 16), batch_size=2, nnz=10, channels=4, device="cpu")
        >>> ln = SparseLayerNorm(normalized_shape=4)
        >>> output = ln(sp_tensor)
        >>> output.feats.shape == sp_tensor.feats.shape
        True
    """

    def forward(self, input: SparseTensor) -> SparseTensor:
        """Forward pass of sparse layer normalization.

        Args:
            input: Input sparse tensor

        Returns:
            SparseTensor: Output sparse tensor with normalized features

        Example:
            >>> import torch
            >>> from sparsetriton import randn
            >>> from sparsetriton.nn.modules import SparseLayerNorm
            >>> sp_tensor = randn((16, 16, 16), batch_size=2, nnz=5, channels=4, device="cpu")
            >>> ln = SparseLayerNorm(normalized_shape=4)
            >>> output = ln(sp_tensor)
            >>> output.feats.shape == sp_tensor.feats.shape
            True
        """
        if isinstance(input, SparseTensor):
            return input.replace(super().forward(input.F))
        return super().forward(input)
