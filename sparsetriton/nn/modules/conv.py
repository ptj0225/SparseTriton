"""Sparse convolution modules for 3D sparse tensors.

This module provides torch.nn.Module implementations of various 3D sparse
convolution layers that work with SparseTensor inputs.

Example:
    >>> import torch
    >>> from sparsetriton import SparseTensor, randn
    >>> from sparsetriton.nn.modules import SparseConv3D, SubMConv3D
    >>> sp_tensor = randn((16, 16, 16), batch_size=2, nnz=10, channels=4, device="cpu")
    >>> conv = SparseConv3D(in_channels=4, out_channels=8, kernel_size=3)
    >>> output = conv(sp_tensor)
    >>> print(output.feats.shape)
    torch.Size([<output_nnz>, 8])
"""

import math
from typing import Dict, List, Tuple, Union

import torch
from torch import nn

import sparsetriton
from sparsetriton import SparseTensor
from sparsetriton.nn import functional as F
from sparsetriton.utils import make_ntuple

__all__ = ["SparseConv3D", "SubMConv3D", "SparseConvTransposed3D"]


class SparseConv3DBase(nn.Module):
    """Base class for sparse 3D convolution layers.

    Implements common functionality for all sparse convolution variants
    including weight initialization and forward pass delegation.

    Attributes:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolution kernel (kd, kh, kw)
        stride: Stride of the convolution (sd, sh, sw)
        padding: Padding added to all sides (pd, ph, pw)
        dilation: Spacing between kernel elements (dd, dh, dw)
        subm: Whether this is a submanifold convolution
        transposed: Whether this is a transposed convolution
        weight: Learnable weights of shape (kernel_volume, in_channels, out_channels)
        bias: Optional learnable bias of shape (out_channels,)

    Example:
        >>> import torch
        >>> from sparsetriton import SparseTensor
        >>> from sparsetriton.nn.modules.conv import SparseConv3DBase
        >>> conv = SparseConv3DBase(in_channels=4, out_channels=8, kernel_size=3, subm=False)
        >>> conv.kernel_size
        (3, 3, 3)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        subm: bool = True,
        bias: bool = True,
        transposed: bool = False,
    ):
        """Initialize a sparse convolution layer.

        Args:
            in_channels: Number of input feature channels
            out_channels: Number of output feature channels
            kernel_size: Size of the convolution kernel (int or tuple of 3)
            stride: Stride of the convolution (int or tuple of 3)
            padding: Padding added to all sides (int or tuple of 3)
            dilation: Spacing between kernel elements (int or tuple of 3)
            subm: Whether to use submanifold convolution (preserve sparsity pattern)
            bias: Whether to include a bias term
            transposed: Whether this is a transposed convolution

        Example:
            >>> import torch
            >>> from sparsetriton.nn.modules.conv import SparseConv3DBase
            >>> conv = SparseConv3DBase(in_channels=4, out_channels=8, kernel_size=3, bias=True)
            >>> conv.weight.shape
            torch.Size([27, 4, 8])
            >>> conv.bias.shape
            torch.Size([8])
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = make_ntuple(kernel_size, 3)
        self.stride = make_ntuple(stride, 3)
        self.padding = make_ntuple(padding, 3)
        self.dilation = make_ntuple(dilation, 3)
        self.subm = subm
        self.transposed = transposed

        self.kernel_volume = self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
        self.weight = nn.Parameter(torch.empty(self.kernel_volume, in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset layer parameters using uniform initialization.

        Initializes weights from U(-stdv, stdv) where stdv = 1/sqrt(in_channels * kernel_volume).
        Bias is initialized with the same distribution.

        Example:
            >>> import torch
            >>> from sparsetriton.nn.modules.conv import SparseConv3DBase
            >>> conv = SparseConv3DBase(in_channels=4, out_channels=8, kernel_size=3)
            >>> conv.reset_parameters()
            >>> conv.weight.abs().max().item() < 1.0  # Should be small
            True
        """
        stdv = 1.0 / math.sqrt(self.in_channels * self.kernel_volume)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input: SparseTensor) -> SparseTensor:
        """Forward pass of sparse convolution.

        Args:
            input: Input sparse tensor

        Returns:
            SparseTensor: Output sparse tensor after convolution

        Example:
            >>> import torch
            >>> from sparsetriton import SparseTensor, randn
            >>> from sparsetriton.nn.modules.conv import SparseConv3DBase
            >>> sp_tensor = randn((16, 16, 16), batch_size=2, nnz=10, channels=4, device="cpu")
            >>> conv = SparseConv3DBase(in_channels=4, out_channels=8, kernel_size=3, subm=False)
            >>> output = conv(sp_tensor)
            >>> isinstance(output, SparseTensor)
            True
        """
        return F.conv.sparse_conv3d(
            tensor=input,
            weight=self.weight,
            kernel_size=self.kernel_size,
            bias=self.bias,
            stride=self.stride,
            dilation=self.dilation,
            padding=self.padding,
            submanifold=self.subm,
            transposed=self.transposed,
        )


class SparseConv3D(SparseConv3DBase):
    """Standard 3D sparse convolution layer.

    Performs sparse convolution with optional stride and dilation for
    downsampling. Does not preserve the input sparsity pattern.

    Attributes:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolution kernel
        stride: Stride for downsampling
        padding: Zero-padding added to all sides
        dilation: Spacing between kernel elements
        bias: Whether to include a bias term

    Example:
        >>> import torch
        >>> from sparsetriton import randn
        >>> from sparsetriton.nn.modules import SparseConv3D
        >>> sp_tensor = randn((16, 16, 16), batch_size=2, nnz=10, channels=4, device="cpu")
        >>> conv = SparseConv3D(in_channels=4, out_channels=8, kernel_size=3, stride=2)
        >>> output = conv(sp_tensor)
        >>> print(output.feats.shape[1])
        8
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        bias: bool = True,
    ):
        """Initialize a standard sparse 3D convolution layer.

        Args:
            in_channels: Number of input feature channels
            out_channels: Number of output feature channels
            kernel_size: Size of the convolution kernel (int or tuple of 3)
            stride: Stride for downsampling (int or tuple of 3)
            padding: Zero-padding added to all sides (int or tuple of 3)
            dilation: Spacing between kernel elements (int or tuple of 3)
            bias: Whether to include a bias term

        Example:
            >>> from sparsetriton.nn.modules import SparseConv3D
            >>> conv = SparseConv3D(in_channels=4, out_channels=8, kernel_size=3, stride=2)
            >>> conv.kernel_size
            (3, 3, 3)
            >>> conv.stride
            (2, 2, 2)
        """
        super(SparseConv3D, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            subm=False,
            transposed=False,
        )


class SubMConv3D(SparseConv3DBase):
    """Submanifold 3D sparse convolution layer.

    Performs sparse convolution that preserves the input sparsity pattern.
    Only outputs features for input voxel locations. Useful for deep
    architectures where maintaining sparsity is important.

    Attributes:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolution kernel
        stride: Must be 1 (required for submanifold conv)
        padding: Zero-padding added to all sides
        dilation: Spacing between kernel elements
        bias: Whether to include a bias term

    Example:
        >>> import torch
        >>> from sparsetriton import randn
        >>> from sparsetriton.nn.modules import SubMConv3D
        >>> sp_tensor = randn((16, 16, 16), batch_size=2, nnz=10, channels=4, device="cpu")
        >>> conv = SubMConv3D(in_channels=4, out_channels=8, kernel_size=3)
        >>> output = conv(sp_tensor)
        >>> # Output coordinates match input coordinates
        >>> torch.allclose(output.coords, sp_tensor.coords)
        tensor(True)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        stride: Union[int, Tuple[int, int, int]] = 1,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        bias: bool = True,
    ):
        """Initialize a submanifold sparse 3D convolution layer.

        Args:
            in_channels: Number of input feature channels
            out_channels: Number of output feature channels
            kernel_size: Size of the convolution kernel (int or tuple of 3)
            stride: Must be 1 for submanifold convolution (int or tuple of 3)
            dilation: Spacing between kernel elements (int or tuple of 3)
            bias: Whether to include a bias term

        Example:
            >>> from sparsetriton.nn.modules import SubMConv3D
            >>> conv = SubMConv3D(in_channels=4, out_channels=8, kernel_size=3)
            >>> conv.kernel_size
            (3, 3, 3)
            >>> conv.subm
            True
        """
        super(SubMConv3D, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,
            subm=True,
            transposed=False,
        )


class SparseConvTransposed3D(SparseConv3DBase):
    """Transposed 3D sparse convolution layer (sparse deconvolution).

    Performs the inverse operation of sparse convolution, useful for
    upsampling and decoder architectures.

    Attributes:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolution kernel
        stride: Stride for upsampling
        padding: Zero-padding added to all sides
        dilation: Spacing between kernel elements
        bias: Whether to include a bias term

    Example:
        >>> import torch
        >>> from sparsetriton import randn
        >>> from sparsetriton.nn.modules import SparseConvTransposed3D
        >>> sp_tensor = randn((16, 16, 16), batch_size=2, nnz=10, channels=4, device="cpu")
        >>> conv_t = SparseConvTransposed3D(in_channels=4, out_channels=8, kernel_size=3, stride=2)
        >>> output = conv_t(sp_tensor)
        >>> print(output.feats.shape[1])
        8
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        subm: bool = True,
        bias: bool = True,
        transposed: bool = False,
    ):
        """Initialize a transposed sparse 3D convolution layer.

        Args:
            in_channels: Number of input feature channels
            out_channels: Number of output feature channels
            kernel_size: Size of the convolution kernel (int or tuple of 3)
            stride: Stride for upsampling (int or tuple of 3)
            padding: Zero-padding added to all sides (int or tuple of 3)
            dilation: Spacing between kernel elements (int or tuple of 3)
            subm: Not used in this implementation (kept for API compatibility)
            bias: Whether to include a bias term
            transposed: Always True for this class

        Example:
            >>> from sparsetriton.nn.modules import SparseConvTransposed3D
            >>> conv_t = SparseConvTransposed3D(in_channels=4, out_channels=8, kernel_size=3, stride=2)
            >>> conv_t.kernel_size
            (3, 3, 3)
            >>> conv_t.transposed
            True
        """
        super(SparseConvTransposed3D, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            subm=False,
            transposed=True,
        )
