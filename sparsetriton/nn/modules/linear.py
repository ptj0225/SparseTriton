"""Linear layers for sparse tensors.

This module provides sparse-compatible linear (fully connected) layers
that operate on SparseTensor inputs.

Example:
    >>> import torch
    >>> from sparsetriton import randn
    >>> from sparsetriton.nn.modules import SparseLinear
    >>> sp_tensor = randn((16, 16, 16), batch_size=2, nnz=10, channels=4, device="cpu")
    >>> linear = SparseLinear(in_features=4, out_features=8)
    >>> output = linear(sp_tensor)
    >>> print(output.feats.shape[1])
    8
"""

from torch import nn
from sparsetriton import SparseTensor

__all__ = ["SparseLinear"]


class SparseLinear(nn.Linear):
    """Sparse-compatible linear (fully connected) layer.

    Applies a linear transformation to the features of a SparseTensor
    while preserving the coordinate structure. Inherits from nn.Linear
    and overrides the forward method to handle SparseTensor inputs.

    Attributes:
        in_features: Size of each input sample
        out_features: Size of each output sample
        weight: Learnable weights of shape (out_features, in_features)
        bias: Optional learnable bias of shape (out_features,)

    Example:
        >>> import torch
        >>> from sparsetriton import randn
        >>> from sparsetriton.nn.modules import SparseLinear
        >>> sp_tensor = randn((16, 16, 16), batch_size=2, nnz=10, channels=4, device="cpu")
        >>> linear = SparseLinear(in_features=4, out_features=8)
        >>> output = linear(sp_tensor)
        >>> print(output.feats.shape)
        torch.Size([10, 8])
        >>> # Coordinates are preserved
        >>> torch.allclose(output.coords, sp_tensor.coords)
        tensor(True)
    """

    def forward(self, input: SparseTensor) -> SparseTensor:
        """Forward pass of sparse linear layer.

        Applies linear transformation to features, preserves coordinates.

        Args:
            input: Input sparse tensor

        Returns:
            SparseTensor: Output sparse tensor with transformed features

        Example:
            >>> import torch
            >>> from sparsetriton import randn
            >>> from sparsetriton.nn.modules import SparseLinear
            >>> sp_tensor = randn((16, 16, 16), batch_size=2, nnz=5, channels=4, device="cpu")
            >>> linear = SparseLinear(in_features=4, out_features=8)
            >>> output = linear(sp_tensor)
            >>> output.feats.shape == torch.Size([5, 8])
            True
        """
        if isinstance(input, SparseTensor):
            return input.replace(super().forward(input.F))
        return super().forward(input)
