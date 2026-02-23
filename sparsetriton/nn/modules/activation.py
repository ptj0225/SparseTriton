"""Activation functions for sparse tensors.

This module provides sparse-compatible activation function layers
that apply element-wise non-linearities to SparseTensor features.

Example:
    >>> import torch
    >>> from sparsetriton import randn
    >>> from sparsetriton.nn.modules import ReLU, LeakyReLU, GELU
    >>> sp_tensor = randn((16, 16, 16), batch_size=2, nnz=10, channels=4, device="cpu")
    >>> relu = ReLU()
    >>> output = relu(sp_tensor)
    >>> torch.all(output.feats >= 0)
    tensor(True)
"""

from torch import nn
from sparsetriton import SparseTensor

__all__ = ["ReLU", "LeakyReLU", "SiLU", "GELU", "Sigmoid", "Tanh"]


class ReLU(nn.ReLU):
    """Sparse-compatible ReLU activation.

    Applies the Rectified Linear Unit activation: max(0, x).

    Attributes:
        inplace: Whether to modify input in-place

    Example:
        >>> import torch
        >>> from sparsetriton import randn
        >>> from sparsetriton.nn.modules import ReLU
        >>> sp_tensor = randn((16, 16, 16), batch_size=2, nnz=10, channels=4, device="cpu")
        >>> relu = ReLU()
        >>> output = relu(sp_tensor)
        >>> torch.all(output.feats >= 0)
        tensor(True)
        >>> torch.allclose(output.coords, sp_tensor.coords)
        tensor(True)
    """

    def forward(self, input: SparseTensor) -> SparseTensor:
        """Forward pass of ReLU activation.

        Args:
            input: Input sparse tensor

        Returns:
            SparseTensor: Output sparse tensor with ReLU applied

        Example:
            >>> import torch
            >>> from sparsetriton import randn
            >>> from sparsetriton.nn.modules import ReLU
            >>> sp_tensor = randn((16, 16, 16), batch_size=2, nnz=5, channels=4, device="cpu")
            >>> relu = ReLU()
            >>> output = relu(sp_tensor)
            >>> torch.all(output.feats >= 0)
            tensor(True)
        """
        if isinstance(input, SparseTensor):
            return input.replace(super().forward(input.F))
        return super().forward(input)


class LeakyReLU(nn.LeakyReLU):
    """Sparse-compatible LeakyReLU activation.

    Applies LeakyReLU activation: max(negative_slope * x, x).

    Attributes:
        negative_slope: Slope for negative values
        inplace: Whether to modify input in-place

    Example:
        >>> import torch
        >>> from sparsetriton import randn
        >>> from sparsetriton.nn.modules import LeakyReLU
        >>> sp_tensor = randn((16, 16, 16), batch_size=2, nnz=10, channels=4, device="cpu")
        >>> lrelu = LeakyReLU(negative_slope=0.01)
        >>> output = lrelu(sp_tensor)
        >>> output.feats.shape == sp_tensor.feats.shape
        True
    """

    def forward(self, input: SparseTensor) -> SparseTensor:
        """Forward pass of LeakyReLU activation.

        Args:
            input: Input sparse tensor

        Returns:
            SparseTensor: Output sparse tensor with LeakyReLU applied

        Example:
            >>> import torch
            >>> from sparsetriton import randn
            >>> from sparsetriton.nn.modules import LeakyReLU
            >>> sp_tensor = randn((16, 16, 16), batch_size=2, nnz=5, channels=4, device="cpu")
            >>> lrelu = LeakyReLU(negative_slope=0.01)
            >>> output = lrelu(sp_tensor)
            >>> output.feats.shape == sp_tensor.feats.shape
            True
        """
        if isinstance(input, SparseTensor):
            return input.replace(super().forward(input.F))
        return super().forward(input)


class SiLU(nn.SiLU):
    """Sparse-compatible SiLU (Swish) activation.

    Applies SiLU activation: x * sigmoid(x).

    Example:
        >>> import torch
        >>> from sparsetriton import randn
        >>> from sparsetriton.nn.modules import SiLU
        >>> sp_tensor = randn((16, 16, 16), batch_size=2, nnz=10, channels=4, device="cpu")
        >>> silu = SiLU()
        >>> output = silu(sp_tensor)
        >>> output.feats.shape == sp_tensor.feats.shape
        True
    """

    def forward(self, input: SparseTensor) -> SparseTensor:
        """Forward pass of SiLU activation.

        Args:
            input: Input sparse tensor

        Returns:
            SparseTensor: Output sparse tensor with SiLU applied
        """
        if isinstance(input, SparseTensor):
            return input.replace(super().forward(input.F))
        return super().forward(input)


class GELU(nn.GELU):
    """Sparse-compatible GELU activation.

    Applies GELU (Gaussian Error Linear Unit) activation.

    Example:
        >>> import torch
        >>> from sparsetriton import randn
        >>> from sparsetriton.nn.modules import GELU
        >>> sp_tensor = randn((16, 16, 16), batch_size=2, nnz=10, channels=4, device="cpu")
        >>> gelu = GELU()
        >>> output = gelu(sp_tensor)
        >>> output.feats.shape == sp_tensor.feats.shape
        True
    """

    def forward(self, input: SparseTensor) -> SparseTensor:
        """Forward pass of GELU activation.

        Args:
            input: Input sparse tensor

        Returns:
            SparseTensor: Output sparse tensor with GELU applied
        """
        if isinstance(input, SparseTensor):
            return input.replace(super().forward(input.F))
        return super().forward(input)


class Sigmoid(nn.Sigmoid):
    """Sparse-compatible Sigmoid activation.

    Applies Sigmoid activation: 1 / (1 + exp(-x)).

    Example:
        >>> import torch
        >>> from sparsetriton import randn
        >>> from sparsetriton.nn.modules import Sigmoid
        >>> sp_tensor = randn((16, 16, 16), batch_size=2, nnz=10, channels=4, device="cpu")
        >>> sigmoid = Sigmoid()
        >>> output = sigmoid(sp_tensor)
        >>> torch.all((output.feats >= 0) & (output.feats <= 1))
        tensor(True)
    """

    def forward(self, input: SparseTensor) -> SparseTensor:
        """Forward pass of Sigmoid activation.

        Args:
            input: Input sparse tensor

        Returns:
            SparseTensor: Output sparse tensor with Sigmoid applied

        Example:
            >>> import torch
            >>> from sparsetriton import randn
            >>> from sparsetriton.nn.modules import Sigmoid
            >>> sp_tensor = randn((16, 16, 16), batch_size=2, nnz=5, channels=4, device="cpu")
            >>> sigmoid = Sigmoid()
            >>> output = sigmoid(sp_tensor)
            >>> torch.all((output.feats >= 0) & (output.feats <= 1))
            tensor(True)
        """
        if isinstance(input, SparseTensor):
            return input.replace(super().forward(input.F))
        return super().forward(input)


class Tanh(nn.Tanh):
    """Sparse-compatible Tanh activation.

    Applies Hyperbolic Tangent activation.

    Example:
        >>> import torch
        >>> from sparsetriton import randn
        >>> from sparsetriton.nn.modules import Tanh
        >>> sp_tensor = randn((16, 16, 16), batch_size=2, nnz=10, channels=4, device="cpu")
        >>> tanh = Tanh()
        >>> output = tanh(sp_tensor)
        >>> torch.all((output.feats >= -1) & (output.feats <= 1))
        tensor(True)
    """

    def forward(self, input: SparseTensor) -> SparseTensor:
        """Forward pass of Tanh activation.

        Args:
            input: Input sparse tensor

        Returns:
            SparseTensor: Output sparse tensor with Tanh applied

        Example:
            >>> import torch
            >>> from sparsetriton import randn
            >>> from sparsetriton.nn.modules import Tanh
            >>> sp_tensor = randn((16, 16, 16), batch_size=2, nnz=5, channels=4, device="cpu")
            >>> tanh = Tanh()
            >>> output = tanh(sp_tensor)
            >>> torch.all((output.feats >= -1) & (output.feats <= 1))
            tensor(True)
        """
        if isinstance(input, SparseTensor):
            return input.replace(super().forward(input.F))
        return super().forward(input)
