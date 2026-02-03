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
        transposed: bool =False,
    ):
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
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.in_channels * self.kernel_volume)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input: SparseTensor) -> SparseTensor:
        return F.conv.sparse_conv3d(
            tensor=input,
            weight=self.weight,
            kernel_size=self.kernel_size,
            bias=self.bias,
            stride=self.stride,
            dilation=self.dilation,
            padding=self.padding,
            submanifold=self.subm,
            transposed=self.transposed
        )

class SparseConv3D(SparseConv3DBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        bias: bool = True
    ):
        super(SparseConv3D, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            subm=False,
            transposed=False
        )

        
class SubMConv3D(SparseConv3DBase):
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
            super(SubMConv3D, self).__init__(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=bias,
                subm=True,
                transposed=False
            )

class SparseConvTransposed3D(SparseConv3DBase):
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
        transposed: bool =False,
    ):
        super(SparseConvTransposed3D, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            subm=False,
            transposed=True
        )

