import torch
from sparsetriton import SparseTensor
from sparsetriton.utils.hash import HashTable
from typing import *
from .funcs import ConvHashOnTheFlyImplicitGEMM
from .kmap import build_out_coords
from sparsetriton.config import get_h_table_f

def sparse_conv3d(
    tensor: SparseTensor,
    weight: torch.Tensor,
    kernel_size: Union[int, Tuple[int, int, int]],
    bias: torch.Tensor = None,
    stride: Union[int, Tuple[int, int, int]] = 1,
    dilation: Union[int, Tuple[int, int, int]] = 1,
    padding: Union[int, Tuple[int, int, int]] = 0,
    transposed: bool = False,
    submanifold: bool = True,
) -> SparseTensor:
    """
    3D sparse convolution function.

    Args:
        tensor (SparseTensor): Input sparse tensor.
        weight (torch.Tensor): Convolution weights. Shape should be (kx * ky * kz, C_in, C_out).
        kernel_size (int or tuple): Size of the convolution kernel.
        bias (torch.Tensor, optional): Bias tensor. Defaults to None.
        stride (int or tuple, optional): Stride of the convolution. Defaults to 1.
        dilation (int or tuple, optional): Dilation of the convolution. Defaults to 1.
        padding (int or tuple, optional): Padding of the convolution. Defaults to 0.
        submanifold (bool, optional): If True, performs a submanifold sparse convolution. Defaults to False.
    """
    if isinstance(kernel_size, tuple):
        # Assuming cubic kernel for now, as build_out_coords takes an int
        assert kernel_size[0] == kernel_size[1] == kernel_size[2], "Only cubic kernels supported for now"
        k_size = kernel_size[0]
    else:
        k_size = kernel_size

    if isinstance(stride, tuple):
        assert stride[0] == stride[1] == stride[2], "Only uniform stride supported for now"
        s = stride[0]
    else:
        s = stride
        
    if isinstance(dilation, tuple):
        assert dilation[0] == dilation[1] == dilation[2], "Only uniform dilation supported for now"
        d = dilation[0]
    else:
        d = dilation

    if submanifold:
        padding = ((k_size - 1) * d) // 2

    if isinstance(padding, tuple):
        assert padding[0] == padding[1] == padding[2], "Only uniform padding supported for now"
        p = padding[0]
    else:
        p = padding
    
    if submanifold:
        assert not transposed, "Submanifold convolution does not support transposed=True"

    device = tensor.F.device
    K, C_in, C_out = weight.shape
    
    assert C_in == tensor.F.shape[1], f"Input channels in weight({C_in}) and tensor({tensor.F.shape[1]}) must match"
    assert K == k_size**3, f"Kernel size in weight({K}) and argument({k_size**3}) must match"
    
    if transposed and s > 1:
        in_hash_table = HashTable(int(tensor.C.shape[0] * get_h_table_f()), device=device)
        key_coords = tensor.C.clone()
        key_coords[:, 1:] *= s
        in_hash_table.insert(key_coords)
    else:
        if tensor._cache.hashtable is not None:
            in_hash_table = tensor._cache.hashtable
        else:
            in_hash_table = HashTable(int(tensor.C.shape[0] * get_h_table_f()), device=device)
            in_hash_table.insert(tensor.C)
            tensor._cache.hashtable = in_hash_table
        
    out_coords, new_spatial_shape, kernel_offsets = build_out_coords(
        tensor.C,
        tensor.spatial_shape,
        k_size,
        s,
        d,
        p,
        transposed=transposed,
        submanifold=submanifold
    )
    if out_coords.shape[0] == 0:
        return SparseTensor(
            feats=torch.empty((0, C_out), device=device, dtype=tensor.F.dtype),
            coords=torch.empty((0, 4), device=device, dtype=tensor.C.dtype),
            spatial_shape=new_spatial_shape,
            batch_size=tensor.batch_size,
        )
    
    if transposed:
        if s > 1:
            spatial_shape = list(map(lambda x: x * s, tensor.spatial_shape))
        else:
            spatial_shape = tensor.spatial_shape

        out_feats = ConvHashOnTheFlyImplicitGEMM.apply(
            tensor.F,
            weight,
            out_coords,
            kernel_offsets,
            in_hash_table.table_keys,
            in_hash_table.table_values,
            spatial_shape,
            1,
        )
        
    else:
        out_feats = ConvHashOnTheFlyImplicitGEMM.apply(
            tensor.F,
            weight,
            out_coords,
            kernel_offsets,
            in_hash_table.table_keys,
            in_hash_table.table_values,
            tensor.spatial_shape,
            s,
        )

    if bias is not None:
        out_feats += bias

    if submanifold:
        if out_coords.shape == tensor.C.shape and torch.equal(out_coords, tensor.C):
            return tensor.replace(out_feats)

    return SparseTensor(
        feats=out_feats,
        coords=out_coords,
        spatial_shape=new_spatial_shape,
        batch_size=tensor.batch_size,
    )
