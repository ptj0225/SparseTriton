import torch
from sparsetriton.tensor import SparseTensor
from .funcs import ConvHashOnTheFlyImplicitGEMM
from .kmap import build_out_coords, build_kernel_offsets, HashTable
from typing import Tuple, Union

def sparse_conv3d(
    tensor: SparseTensor,
    weight: torch.Tensor,
    kernel_size: Union[int, Tuple[int, int, int]],
    bias: torch.Tensor = None,
    stride: Union[int, Tuple[int, int, int]] = 1,
    dilation: Union[int, Tuple[int, int, int]] = 1,
    padding: Union[int, Tuple[int, int, int]] = 0,
    submanifold: bool = True,
) -> SparseTensor:
    """
    3D sparse convolution function.

    Args:
        tensor (SparseTensor): Input sparse tensor.
        weight (torch.Tensor): Convolution weights. Shape should be (C_out, C_in, Kx, Ky, Kz) or (kx * ky * kz, C_in, C_out).
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

    if isinstance(padding, tuple):
        assert padding[0] == padding[1] == padding[2], "Only uniform padding supported for now"
        p = padding[0]
    else:
        p = padding

    device = tensor.F.device
    
    # Reshape weight tensor if needed
    # Assuming weight layout is (C_out, C_in, Kx, Ky, Kz)
    if weight.dim() == 5:
        C_out, C_in, kx, ky, kz = weight.shape
        # (C_out, C_in, Kx, Ky, Kz) -> (Kx, Ky, Kz, C_in, C_out) -> (K, C_in, C_out)
        weight = weight.permute(2, 3, 4, 1, 0).contiguous()
        weight = weight.view(-1, C_in, C_out)
    
    K, C_in, C_out = weight.shape
    assert C_in == tensor.F.shape[1], f"Input channels in weight({C_in}) and tensor({tensor.F.shape[1]}) must match"
    assert K == k_size**3, f"Kernel size in weight({K}) and argument({k_size**3}) must match"

    # Build input hash table
    in_hash_table = HashTable(tensor.C.shape[0] * 2, device=device)
    in_hash_table.insert(tensor.C)
    
    if submanifold:
        out_coords = tensor.C
        new_spatial_shape = tensor.spatial_shape
    else:
        out_coords, new_spatial_shape = build_out_coords(
            tensor.C,
            tensor.spatial_shape,
            k_size,
            s,
            d,
            p
        )

    if out_coords.shape[0] == 0:
        return SparseTensor(
            feats=torch.empty((0, C_out), device=device, dtype=tensor.F.dtype),
            coords=torch.empty((0, 4), device=device, dtype=tensor.C.dtype),
            spatial_shape=new_spatial_shape,
            batch_size=tensor.batch_size,
        )

    # Calculate kernel offsets
    k_offsets = build_kernel_offsets(
        torch.tensor([k_size]*3, device=device),
        torch.tensor([d]*3, device=device),
        device,
        tensor.C.dtype,
    )
    print(k_offsets)
    out_feats = ConvHashOnTheFlyImplicitGEMM.apply(
        tensor.F,
        weight,
        out_coords * s,
        k_offsets,
        in_hash_table.table_keys,
        in_hash_table.table_values,
    )

    if bias is not None:
        out_feats += bias
    if submanifold:
        return tensor.replace(out_feats)
    else:
        return SparseTensor(
            feats=out_feats,
            coords=out_coords,
            spatial_shape=new_spatial_shape,
            batch_size=tensor.batch_size,
        )