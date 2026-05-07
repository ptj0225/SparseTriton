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


class SparseBatchNorm(nn.Module):
    """Sparse-compatible batch normalization layer.

    Applies batch normalization to the features of a SparseTensor
    while preserving the coordinate structure. Normalizes across
    the batch dimension.

    Unlike nn.BatchNorm1d, this correctly computes per-batch statistics
    using the batch indices from coordinates rather than treating each
    sparse point as an independent sample.

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

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        if track_running_stats:
            self.register_buffer("running_mean", torch.zeros(num_features))
            self.register_buffer("running_var", torch.ones(num_features))
            self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.ones_()
            self.bias.data.zeros_()

    def _check_input_dim(self, input):
        if input.F.dim() != 2:
            raise ValueError(f"Expected 2D input, got {input.F.dim()}D")

    def forward(self, input: SparseTensor) -> SparseTensor:
        self._check_input_dim(input)

        if self.training:
            result = sparse_batch_norm(
                input,
                weight=self.weight,
                bias=self.bias,
                running_mean=self.running_mean,
                running_var=self.running_var,
                training=True,
                eps=self.eps,
                momentum=self.momentum,
            )
            if self.track_running_stats:
                self.num_batches_tracked += 1
        else:
            result = sparse_batch_norm(
                input,
                weight=self.weight,
                bias=self.bias,
                running_mean=self.running_mean,
                running_var=self.running_var,
                training=False,
                eps=self.eps,
            )

        return result


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
