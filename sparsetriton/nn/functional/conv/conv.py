import torch
from sparsetriton import SparseTensor
from sparsetriton.utils.hash import HashTable
from typing import *
from .funcs import ConvHashOnTheFlyImplicitGEMM
from .kmap import build_out_coords
from sparsetriton.config import get_h_table_f, get_force_cpu


def sparse_conv3d_cpu(
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
    CPU implementation of 3D sparse convolution using PyTorch operations.

    This is a fallback implementation for when Triton kernels are not available
    or when force_cpu is enabled. It uses hash-based coordinate lookups and
    gather/scatter operations to perform sparse convolution on CPU.

    Args:
        tensor: Input sparse tensor
        weight: Convolution weights (K, C_in, C_out)
        kernel_size: Size of the convolution kernel
        bias: Optional bias tensor
        stride: Convolution stride
        dilation: Convolution dilation
        padding: Convolution padding
        transposed: Whether to use transposed convolution
        submanifold: Whether to preserve input sparsity pattern
    """
    if isinstance(kernel_size, tuple):
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

    if submanifold:
        assert not transposed, "Submanifold convolution does not support transposed=True"

    device = tensor.F.device
    K, C_in, C_out = weight.shape

    assert C_in == tensor.F.shape[1], f"Input channels in weight({C_in}) and tensor({tensor.F.shape[1]}) must match"
    assert K == k_size**3, f"Kernel size in weight({K}) and argument({k_size**3}) must match"

    # Generate kernel offsets
    kernel_offsets = _generate_kernel_offsets_cpu(k_size, d, p, transposed)
    kernel_offsets = kernel_offsets.to(tensor.C.device)

    # Build hash table for input coordinates
    in_hash_table = _build_hash_table_cpu(tensor.C, device)

    # Build output coordinates
    out_coords, new_spatial_shape = _build_out_coords_cpu(
        tensor.C,
        tensor.spatial_shape,
        k_size,
        s,
        d,
        p,
        transposed=transposed,
        submanifold=submanifold,
        kernel_offsets=kernel_offsets
    )

    if out_coords.shape[0] == 0:
        return SparseTensor(
            feats=torch.empty((0, C_out), device=device, dtype=tensor.F.dtype),
            coords=torch.empty((0, 4), device=device, dtype=tensor.C.dtype),
            spatial_shape=new_spatial_shape,
            batch_size=tensor.batch_size,
        )

    # Perform convolution using hash-based lookups
    out_feats = _sparse_conv_forward_cpu(
        tensor.F,
        weight,
        out_coords,
        kernel_offsets,
        in_hash_table,
        tensor.spatial_shape if not transposed else tensor.C[:, 1:].max(dim=0).values + 1,
        transposed=transposed,
        stride=s,
        device=device
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


def _generate_kernel_offsets_cpu(kernel_size: int, dilation: int, padding: int, transposed: bool) -> torch.Tensor:
    """Generate kernel offsets for CPU implementation."""
    # Generate base offsets
    offset_range = torch.arange(kernel_size) - kernel_size // 2
    offsets = torch.stack(torch.meshgrid(offset_range, offset_range, offset_range, indexing='ij'), dim=-1)
    offsets = offsets.reshape(-1, 3) * dilation

    # Apply padding and sign based on transposed
    if transposed:
        # For transposed conv: out = in + offset
        offsets = offsets - (kernel_size - 1) * dilation // 2 - padding
    else:
        # For normal conv: out = in + offset, then we look for in = out - offset
        offsets = -offsets + padding - (kernel_size - 1) * dilation // 2

    return offsets


def _build_hash_table_cpu(coords: torch.Tensor, device: torch.device) -> Dict[Tuple[int, int, int, int], int]:
    """Build a hash table mapping coordinates to indices."""
    hash_table = {}
    for i, coord in enumerate(coords):
        # Convert to tuple for hashing
        key = (coord[0].item(), coord[1].item(), coord[2].item(), coord[3].item())
        hash_table[key] = i
    return hash_table


def _build_out_coords_cpu(
    in_coords: torch.Tensor,
    spatial_shape: Tuple[int, int, int],
    kernel_size: int,
    stride: int,
    dilation: int,
    padding: int,
    transposed: bool = False,
    submanifold: bool = True,
    kernel_offsets: torch.Tensor = None,
) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    """Build output coordinates for CPU implementation."""
    device = in_coords.device

    # Compute output spatial shape
    if transposed:
        new_spatial_shape = tuple(
            (s - 1) * stride + (kernel_size - 1) * dilation - 2 * padding + 1
            for s in spatial_shape
        )
    else:
        new_spatial_shape = tuple(
            (s - (kernel_size - 1) * dilation + 2 * padding - 1) // stride + 1
            for s in spatial_shape
        )

    # Submanifold: return input coords as output
    if submanifold and stride == 1 and ((kernel_size - 1) * dilation) // 2 == padding:
        return in_coords, new_spatial_shape

    # Generate kernel offsets if not provided
    if kernel_offsets is None:
        kernel_offsets = _generate_kernel_offsets_cpu(kernel_size, dilation, padding, transposed)

    # Expand input coordinates with kernel offsets
    N = in_coords.shape[0]
    K = kernel_offsets.shape[0]

    out_coords_set = set()

    for i in range(N):
        in_coord = in_coords[i]
        batch_idx = in_coord[0].item()

        for j in range(K):
            offset = kernel_offsets[j]

            if transposed:
                # Transposed conv: out = (in_coord[1:] * stride) + offset
                x, y, z = in_coord[1:] * stride + offset
            else:
                # Normal conv: out = in_coord[1:] + offset
                x, y, z = in_coord[1:] + offset

            # Apply stride filtering for normal conv
            if not transposed and stride > 1:
                if x % stride != 0 or y % stride != 0 or z % stride != 0:
                    continue
                x, y, z = x // stride, y // stride, z // stride

            # Spatial range filtering
            if (0 <= x < new_spatial_shape[0] and
                0 <= y < new_spatial_shape[1] and
                0 <= z < new_spatial_shape[2]):
                out_coords_set.add((batch_idx, int(x.item()), int(y.item()), int(z.item())))

    if not out_coords_set:
        return torch.empty((0, 4), device=device, dtype=in_coords.dtype), new_spatial_shape

    out_coords = torch.tensor(list(out_coords_set), device=device, dtype=in_coords.dtype)
    return out_coords, new_spatial_shape


def _sparse_conv_forward_cpu(
    in_feats: torch.Tensor,
    weight: torch.Tensor,
    out_coords: torch.Tensor,
    kernel_offsets: torch.Tensor,
    in_hash_table: Dict[Tuple[int, int, int, int], int],
    spatial_shape: Union[Tuple[int, int, int], torch.Tensor],
    transposed: bool = False,
    stride: int = 1,
    device: torch.device = torch.device('cpu'),
) -> torch.Tensor:
    """Perform forward sparse convolution on CPU."""
    N_out = out_coords.shape[0]
    _, C_in, C_out = weight.shape

    out_feats = torch.zeros((N_out, C_out), device=device, dtype=in_feats.dtype)

    if isinstance(spatial_shape, tuple):
        spatial_shape_tensor = torch.tensor(spatial_shape, device=device)
    else:
        spatial_shape_tensor = spatial_shape

    for i in range(N_out):
        out_coord = out_coords[i]
        batch_idx = out_coord[0].item()
        out_pos = [out_coord[1].item(), out_coord[2].item(), out_coord[3].item()]

        for j in range(kernel_offsets.shape[0]):
            offset = kernel_offsets[j]

            if transposed:
                # Transposed conv: in_pos = (out_pos - offset) / stride
                in_pos = torch.tensor([
                    out_pos[0] - offset[0],
                    out_pos[1] - offset[1],
                    out_pos[2] - offset[2]
                ], device=device)
            else:
                # Normal conv: in_pos = out_pos * stride - offset
                in_pos = torch.tensor([
                    out_pos[0] * stride - offset[0],
                    out_pos[1] * stride - offset[1],
                    out_pos[2] * stride - offset[2]
                ], device=device)

            # Spatial bounds check
            if (in_pos[0] < 0 or in_pos[0] >= spatial_shape_tensor[0] or
                in_pos[1] < 0 or in_pos[1] >= spatial_shape_tensor[1] or
                in_pos[2] < 0 or in_pos[2] >= spatial_shape_tensor[2]):
                continue

            # Look up in hash table
            in_key = (batch_idx, int(in_pos[0].item()), int(in_pos[1].item()), int(in_pos[2].item()))
            in_idx = in_hash_table.get(in_key)

            if in_idx is not None:
                # Apply convolution
                out_feats[i] += torch.mm(
                    in_feats[in_idx:in_idx+1],
                    weight[j].t()
                ).squeeze(0)

    return out_feats


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
    # Use CPU implementation if force_cpu is enabled or if device is CPU
    use_cpu = get_force_cpu() or tensor.F.device.type == 'cpu'

    if use_cpu:
        return sparse_conv3d_cpu(
            tensor, weight, kernel_size, bias, stride, dilation, padding,
            transposed, submanifold
        )

    # GPU implementation with Triton kernels
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
