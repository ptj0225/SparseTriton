import torch
from sparsetriton import SparseTensor
from sparsetriton.utils.hash import HashTable
from typing import *
from .funcs import ConvHashOnTheFlyImplicitGEMM
from .funcs.implicit_precomputed_gemm import ConvPrecomputedNeighborGEMM
from .funcs.im2col_gemm import im2col_gemm_conv
from .kmap import build_out_coords
from sparsetriton.config import get_h_table_f, get_conv_algo, ConvAlgo


def compute_neighbor_indices(
    in_hash_table: HashTable,
    out_coords: torch.Tensor,
    kernel_offsets: torch.Tensor,
    spatial_shape: Tuple[int, int, int],
    stride: int,
) -> torch.Tensor:
    """Compute neighbor indices for each output coordinate.
    
    Vectorized PyTorch implementation for speed.
    
    Args:
        in_hash_table: Hash table mapping input coordinates to indices
        out_coords: (N_out, 4) output coordinates [batch, x, y, z]
        kernel_offsets: (K, 3) kernel offsets
        spatial_shape: (X, Y, Z) spatial dimensions
        stride: Convolution stride
    
    Returns:
        neighbor_indices: (N_out, K) tensor of input indices (-1 if not found)
    """
    N_out = out_coords.shape[0]
    K = kernel_offsets.shape[0]
    device = out_coords.device
    
    # Expand out_coords: (N_out, 1, 4) -> (N_out, K, 4)
    out_coords_expanded = out_coords.unsqueeze(1).expand(-1, K, -1)  # (N_out, K, 4)
    
    # Expand kernel_offsets: (1, K, 3) -> (N_out, K, 3)
    kernel_offsets_expanded = kernel_offsets.unsqueeze(0).expand(N_out, -1, -1)  # (N_out, K, 3)
    
    # Compute neighbor coords: (N_out, K, 4)
    neighbor_coords = out_coords_expanded.clone()
    neighbor_coords[:, :, 1] = out_coords_expanded[:, :, 1] * stride - kernel_offsets_expanded[:, :, 0]
    neighbor_coords[:, :, 2] = out_coords_expanded[:, :, 2] * stride - kernel_offsets_expanded[:, :, 1]
    neighbor_coords[:, :, 3] = out_coords_expanded[:, :, 3] * stride - kernel_offsets_expanded[:, :, 2]
    
    # Check bounds
    valid = (
        (neighbor_coords[:, :, 1] >= 0) & (neighbor_coords[:, :, 1] < spatial_shape[0]) &
        (neighbor_coords[:, :, 2] >= 0) & (neighbor_coords[:, :, 2] < spatial_shape[1]) &
        (neighbor_coords[:, :, 3] >= 0) & (neighbor_coords[:, :, 3] < spatial_shape[2])
    )
    
    # Flatten for batch query
    neighbor_coords_flat = neighbor_coords.reshape(-1, 4)  # (N_out * K, 4)
    valid_flat = valid.reshape(-1)  # (N_out * K,)
    
    # Query hash table
    indices_flat = torch.full((N_out * K,), -1, dtype=torch.int32, device=device)
    if valid_flat.any():
        valid_coords = neighbor_coords_flat[valid_flat]
        queried_indices = in_hash_table.query(valid_coords)
        indices_flat[valid_flat] = queried_indices
    
    # Reshape back
    neighbor_indices = indices_flat.reshape(N_out, K)
    
    return neighbor_indices

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
    
    # Select algorithm based on config
    algo = get_conv_algo()
    
    if algo == ConvAlgo.PrecomputedNeighborGEMM:
        # Pre-compute neighbor indices with caching
        cache_key = (k_size, s, d, submanifold, transposed)
        
        if cache_key in tensor._cache.neighbor_indices:
            # Use cached neighbor indices
            out_coords, neighbor_indices, new_spatial_shape = tensor._cache.neighbor_indices[cache_key]
        else:
            # Compute and cache
            if transposed and s > 1:
                spatial_shape_for_lookup = tuple(x * s for x in tensor.spatial_shape)
            else:
                spatial_shape_for_lookup = tensor.spatial_shape
            
            neighbor_indices = compute_neighbor_indices(
                in_hash_table, out_coords, kernel_offsets, spatial_shape_for_lookup, s if not transposed else 1
            )
            
            # Cache for future use
            tensor._cache.neighbor_indices[cache_key] = (out_coords, neighbor_indices, new_spatial_shape)
        
        out_feats = ConvPrecomputedNeighborGEMM.apply(
            tensor.F,
            weight,
            neighbor_indices,
        )
    else:
        # Default: ImplicitHashFlyGEMM
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
