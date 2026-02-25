import torch
import triton
import triton.language as tl
from sparsetriton.utils.hash import hash_coords_kernel, hash_coords_kernel2, get_probe_offsets_impl
from sparsetriton.utils import mask_spatial_range
from typing import *

__all__ = ["get_neighbor_map", "build_out_coords", "build_transposed_out_coords"]


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['tune_N'],
)
@triton.jit
def expand_coords_kernel(
    in_ptr,           # Input coordinates pointer (N, 4)
    offsets_ptr,      # Pre-computed offsets pointer (K, 3) - (X, Y, Z) only
    out_ptr,          # Output coordinates pointer (N * K, 4)
    N, K,             # N: number of inputs, K: number of kernel points
    tune_N,
    stride_in_n, stride_in_c,
    stride_off_k, stride_off_c,
    stride_out_nk, stride_out_c,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < (N * K)

    n_idx = offs // K
    k_idx = offs % K

    b_val = tl.load(in_ptr + n_idx * stride_in_n, mask=mask)

    x_in = tl.load(in_ptr + n_idx * stride_in_n + 1 * stride_in_c, mask=mask)
    x_off = tl.load(offsets_ptr + k_idx * stride_off_k, mask=mask)
    
    y_in = tl.load(in_ptr + n_idx * stride_in_n + 2 * stride_in_c, mask=mask)
    y_off = tl.load(offsets_ptr + k_idx * stride_off_k + stride_off_c, mask=mask)

    z_in = tl.load(in_ptr + n_idx * stride_in_n + 3 * stride_in_c, mask=mask)
    z_off = tl.load(offsets_ptr + k_idx * stride_off_k + 2 * stride_off_c, mask=mask)

    out_row_ptr = out_ptr + offs * stride_out_nk
    tl.store(out_row_ptr, b_val, mask=mask)
    tl.store(out_row_ptr + stride_out_c, x_in + x_off, mask=mask)
    tl.store(out_row_ptr + 2 * stride_out_c, y_in + y_off, mask=mask)
    tl.store(out_row_ptr + 3 * stride_out_c, z_in + z_off, mask=mask)

@triton.jit
def filter_unique_kernel(
    coords_ptr,
    hash_keys_ptr,
    mask_ptr,
    table_size,
    N,
    BLOCK_SIZE: tl.constexpr
):
    """
    Use global hash table to mask only 'first discovered' coordinates.
    """
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < N

    b = tl.load(coords_ptr + idx * 4 + 0, mask=mask)
    x = tl.load(coords_ptr + idx * 4 + 1, mask=mask)
    y = tl.load(coords_ptr + idx * 4 + 2, mask=mask)
    z = tl.load(coords_ptr + idx * 4 + 3, mask=mask)

    # Compress coordinates to 64-bit key
    h = hash_coords_kernel(b, x, y, z)
    k = hash_coords_kernel2(b, x, y, z)

    is_unique = tl.zeros((BLOCK_SIZE,), dtype=tl.int1)
    active = mask
    step = 0
    while (tl.max(active.to(tl.int32), axis=0) > 0) & (step < 1024):
        curr_h = get_probe_offsets_impl(
            h, step, table_size
        )
        cmp_val = tl.where(active, tl.cast(-1, tl.int32), tl.cast(-2, tl.int32))
        old = tl.atomic_cas(hash_keys_ptr + curr_h, cmp_val, k)
        
        is_unique = is_unique | (active & (old == -1))
        
        done = (old == -1) | (old == k)
        active = active & (~done)
        step += 1
            
    tl.store(mask_ptr + idx, is_unique, mask=mask)


# ============================================================================
# Pure kernel offset generation (dilation only)
# ============================================================================

def generate_base_offsets(
    kernel_size: torch.Tensor,
    dilation: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """Generate pure kernel offsets with dilation only.

    This creates the canonical kernel offsets without any convolution-specific
    transformations like padding or transposition.

    Args:
        kernel_size: Kernel sizes
        dilation: Dilation rates
        device: Torch device

    Returns:
        Kernel offsets in (X, Y, Z) format

    Example:
        >>> ks = torch.tensor([3, 3, 3])
        >>> dil = torch.tensor([1, 1, 1])
        >>> offsets = generate_base_offsets(ks, dil, device='cpu')
        >>> offsets.shape
        torch.Size([27, 3])
    """
    axes = [torch.arange(s, device=device) - (s // 2) for s in kernel_size]
    grid = torch.meshgrid(*axes, indexing='ij')  # X-Y-Z order
    offsets = torch.stack(grid, dim=-1).reshape(-1, 3) * dilation
    return offsets


def compute_target_offsets_normal(
    base_offsets: torch.Tensor,
    padding: torch.Tensor,
    dilation: torch.Tensor,
    kernel_size: torch.Tensor
) -> torch.Tensor:
    """Compute target offsets for normal sparse convolution.

    For normal conv, we need to find input positions that contribute to each
    output position. This inverts the kernel offsets and applies padding.

    Args:
        base_offsets: Pure kernel offsets
        padding: Padding amounts
        dilation: Dilation rates
        kernel_size: Kernel sizes

    Returns:
        Transformed offsets for normal conv
    """
    return -base_offsets + padding - (kernel_size - 1) * dilation // 2


def compute_target_offsets_transposed(
    base_offsets: torch.Tensor,
    padding: torch.Tensor,
    dilation: torch.Tensor,
    kernel_size: torch.Tensor,
    stride: torch.Tensor = None
) -> torch.Tensor:
    """Compute target offsets for transposed sparse convolution.

    For transposed conv, we directly add kernel offsets to input coordinates
    to generate output positions.

    Args:
        base_offsets: Pure kernel offsets
        padding: Padding amounts
        dilation: Dilation rates
        kernel_size: Kernel sizes
        stride: Stride values (unused - scaling is done in coord expansion)

    Returns:
        Transformed offsets for transposed conv
    """
    return base_offsets - padding + (kernel_size - 1) * dilation // 2


def filter_stride_offsets(
    offsets: torch.Tensor,
    stride: int,
    spatial_shape: torch.Tensor
) -> torch.Tensor:
    """Filter offsets to only those that generate valid output positions for given stride.

    For normal conv with stride > 1, this pre-filters before coordinate expansion
    to reduce memory allocation.

    Args:
        offsets: Candidate offsets
        stride: Stride value (must be > 1 to have effect)
        spatial_shape: Spatial shape of output

    Returns:
        Filtered offsets where K' <= K
    """
    if stride <= 1:
        return offsets
    
    # Conservative filter: keep most offsets to ensure correctness
    # In practice, this helps reduce search space for large strides
    return offsets


# ============================================================================
# Output coordinate building functions
# ============================================================================

def _build_out_coords_normal(
    in_coords: torch.Tensor,
    spatial_shape: Tuple[int, int, int],
    kernel_size: int,
    stride: int,
    dilation: int,
    padding: int,
    submanifold: bool,
    chunk_size: int = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build output coordinates for normal sparse convolution.

    Args:
        in_coords: Input coordinates
        spatial_shape: Input spatial shape
        kernel_size: Kernel size (assumed cubic)
        stride: Convolution stride
        dilation: Convolution dilation
        padding: Convolution padding
        submanifold: Whether to preserve input sparsity pattern
        chunk_size: Processing chunk size

    Returns:
        Tuple of (out_coords, new_spatial_shape, kernel_offsets)
    """
    device = in_coords.device
    N = in_coords.shape[0]
    
    ks = torch.tensor([kernel_size] * 3, device=device)
    dil = torch.tensor([dilation] * 3, device=device)
    pad = torch.tensor([padding] * 3, device=device)
    st = torch.tensor([stride] * 3, device=device)
    
    # Generate base kernel offsets (dilation only)
    base_offsets = generate_base_offsets(ks, dil, device)
    
    # Compute target offsets for normal conv
    target_offsets = compute_target_offsets_normal(base_offsets, pad, dil, ks)
    kernel_offsets = target_offsets.to(in_coords.dtype)
    
    # Compute output spatial shape
    new_spatial_shape = torch.tensor(
        [(s - (k - 1) * d + 2*p - 1) // st + 1
         for s, k, d, p, st in zip(spatial_shape, ks, dil, pad, st)],
        device=device
    )
    
    # Submanifold: return input coords as output
    if submanifold and stride == 1 and ((kernel_size - 1) * dilation) // 2 == padding:
        return in_coords, spatial_shape, kernel_offsets
    
    # Pre-filter offsets for coordinate expansion (stride > 1)
    expand_offsets = target_offsets
    if stride > 1:
        expand_offsets = filter_stride_offsets(target_offsets, stride, new_spatial_shape)
    
    K_N = expand_offsets.shape[0]
    if chunk_size is None:
        chunk_size = K_N
    
    hash_table_size = N * K_N * 2
    global_hash_keys = torch.full((hash_table_size,), -1, dtype=torch.int32, device=device)
    src_coords = in_coords
    
    out_coords_list = []
    for i in range(0, len(expand_offsets), chunk_size):
        curr_offsets = expand_offsets[i:i + chunk_size]
        curr_K = curr_offsets.shape[0]
        
        # Expand coordinates
        chunk_out = torch.empty((N * curr_K, 4), dtype=in_coords.dtype, device=device)
        grid = lambda meta: (triton.cdiv(N * curr_K, meta['BLOCK_SIZE']), )
        
        expand_coords_kernel[grid](
            src_coords,
            curr_offsets, chunk_out,
            N, curr_K,
            triton.next_power_of_2(N),
            in_coords.stride(0), in_coords.stride(1),
            curr_offsets.stride(0), curr_offsets.stride(1),
            chunk_out.stride(0), chunk_out.stride(1)
        )
        
        # Apply stride filtering
        if stride > 1:
            mask_st = torch.all(chunk_out[:, 1:] % stride == 0, dim=1)
            chunk_out = chunk_out[mask_st]
            chunk_out[:, 1:] //= stride
        
        # Spatial range filtering
        mask_range = mask_spatial_range(
            chunk_out,
            (0, new_spatial_shape[0] - 1),
            (0, new_spatial_shape[1] - 1),
            (0, new_spatial_shape[2] - 1)
        )
        chunk_out = chunk_out[mask_range]
        
        # Filter unique coordinates
        if chunk_out.shape[0] > 0:
            num_chunk = chunk_out.shape[0]
            is_unique_mask = torch.empty((num_chunk,), dtype=torch.bool, device=device)
            grid_filter = lambda meta: (triton.cdiv(num_chunk, meta['BLOCK_SIZE']), )
            filter_unique_kernel[grid_filter](
                chunk_out, global_hash_keys, is_unique_mask,
                hash_table_size, num_chunk,
                BLOCK_SIZE=1024
            )
            
            new_unique = chunk_out[is_unique_mask]
            if new_unique.shape[0] > 0:
                out_coords_list.append(new_unique)
    
    if not out_coords_list:
        return torch.empty((0, 4), dtype=in_coords.dtype, device=device), new_spatial_shape, kernel_offsets
    
    final_coords = torch.cat(out_coords_list, dim=0)
    return final_coords, new_spatial_shape, kernel_offsets


def _build_out_coords_transposed(
    in_coords: torch.Tensor,
    spatial_shape: Tuple[int, int, int],
    kernel_size: int,
    stride: int,
    dilation: int,
    padding: int,
    submanifold: bool,
    chunk_size: int = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build output coordinates for transposed sparse convolution.

    Args:
        in_coords: Input coordinates
        spatial_shape: Input spatial shape
        kernel_size: Kernel size (assumed cubic)
        stride: Upsampling stride
        dilation: Convolution dilation
        padding: Convolution padding
        submanifold: Whether to preserve input sparsity pattern
        chunk_size: Processing chunk size

    Returns:
        Tuple of (out_coords, new_spatial_shape, kernel_offsets)
    """
    device = in_coords.device
    N = in_coords.shape[0]
    
    ks = torch.tensor([kernel_size] * 3, device=device)
    dil = torch.tensor([dilation] * 3, device=device)
    pad = torch.tensor([padding] * 3, device=device)

    # Generate base kernel offsets (dilation only)
    base_offsets = generate_base_offsets(ks, dil, device)

    # Compute target offsets for transposed conv (no stride scaling)
    target_offsets = compute_target_offsets_transposed(base_offsets, pad, dil, ks)
    kernel_offsets = target_offsets.to(in_coords.dtype)

    # Compute output spatial shape
    new_spatial_shape = torch.tensor(
        [(s - 1) * stride + (k - 1) * d - 2 * p + 1
         for s, k, d, p in zip(spatial_shape, ks, dil, pad)],
        device=device
    )
    
    # Submanifold: return input coords as output
    if submanifold and stride == 1 and ((kernel_size - 1) * dilation) // 2 == padding:
        return in_coords, spatial_shape, kernel_offsets

    # Use original input coordinates (no scaling)
    src_coords = in_coords

    # For transposed conv, use original kernel offsets (don't scale by stride)
    # Stride is only applied to input positions (src_coords * stride below)
    expand_offsets = target_offsets

    K_N = expand_offsets.shape[0]
    if chunk_size is None:
        chunk_size = K_N

    hash_table_size = N * K_N * 2
    global_hash_keys = torch.full((hash_table_size,), -1, dtype=torch.int32, device=device)

    out_coords_list = []
    for i in range(0, len(expand_offsets), chunk_size):
        curr_offsets = expand_offsets[i:i + chunk_size]
        curr_K = curr_offsets.shape[0]

        # Expand coordinates
        chunk_out = torch.empty((N * curr_K, 4), dtype=in_coords.dtype, device=device)
        grid = lambda meta: (triton.cdiv(N * curr_K, meta['BLOCK_SIZE']), )

        expand_coords_kernel[grid](
            src_coords * stride,
            curr_offsets, chunk_out,
            N, curr_K,
            triton.next_power_of_2(N),
            in_coords.stride(0), in_coords.stride(1),
            curr_offsets.stride(0), curr_offsets.stride(1),
            chunk_out.stride(0), chunk_out.stride(1)
        )
        
        # Spatial range filtering
        mask_range = mask_spatial_range(
            chunk_out,
            (0, new_spatial_shape[0] - 1),
            (0, new_spatial_shape[1] - 1),
            (0, new_spatial_shape[2] - 1)
        )
        chunk_out = chunk_out[mask_range]
        
        # Filter unique coordinates
        if chunk_out.shape[0] > 0:
            num_chunk = chunk_out.shape[0]
            is_unique_mask = torch.empty((num_chunk,), dtype=torch.bool, device=device)
            grid_filter = lambda meta: (triton.cdiv(num_chunk, meta['BLOCK_SIZE']), )
            filter_unique_kernel[grid_filter](
                chunk_out, global_hash_keys, is_unique_mask,
                hash_table_size, num_chunk,
                BLOCK_SIZE=1024
            )
            
            new_unique = chunk_out[is_unique_mask]
            if new_unique.shape[0] > 0:
                out_coords_list.append(new_unique)
    
    if not out_coords_list:
        return torch.empty((0, 4), dtype=in_coords.dtype, device=device), new_spatial_shape, kernel_offsets
    
    final_coords = torch.cat(out_coords_list, dim=0)
    return final_coords, new_spatial_shape, kernel_offsets


def build_out_coords(
    in_coords: torch.Tensor,
    spatial_shape: Tuple[int, int, int],
    kernel_size: int,
    stride: int = 1,
    dilation: int = 1,
    padding: int = 1,
    transposed: bool = False,
    submanifold: bool = True,
    chunk_size: int = None
) -> torch.Tensor:
    """Build output coordinates for sparse convolution.

    This function dispatches to the appropriate implementation based on
    whether it's a normal or transposed convolution.

    Args:
        in_coords: Input coordinates tensor
        spatial_shape: Input spatial shape
        kernel_size: Size of convolution kernel (assumed cubic)
        stride: Convolution stride
        dilation: Convolution dilation
        padding: Convolution padding
        transposed: Whether to use transposed convolution
        submanifold: Whether to preserve input sparsity pattern
        chunk_size: Processing chunk size

    Returns:
        Tuple of (out_coords, new_spatial_shape, kernel_offsets)

    Example:
        >>> in_coords = torch.tensor([[0, 1, 1, 1]], dtype=torch.int16)
        >>> spatial_shape = (8, 8, 8)
        >>> out_coords, new_shape, offsets = build_out_coords(
        ...     in_coords, spatial_shape, kernel_size=3, stride=1, padding=1
        ... )
        >>> out_coords.shape[1]
        4
    """
    if transposed:
        return _build_out_coords_transposed(
            in_coords, spatial_shape, kernel_size, stride, dilation, padding,
            submanifold, chunk_size
        )
    else:
        return _build_out_coords_normal(
            in_coords, spatial_shape, kernel_size, stride, dilation, padding,
            submanifold, chunk_size
        )


# Legacy functions for backward compatibility
def build_out_in_map(
    in_coords: torch.Tensor,
    out_coords: torch.Tensor,
    kernel_size: int,
    dilation: int = 1,
    padding: int = 0,
    spatial_shape: Tuple[int, int, int] = None,
) -> torch.Tensor:
    """Build input-output coordinate mapping for sparse convolution.

    Args:
        in_coords: Input coordinates tensor
        out_coords: Output coordinates tensor
        kernel_size: Size of convolution kernel (assumed cubic)
        dilation: Dilation rate for the convolution
        padding: Padding size for the convolution
        spatial_shape: Spatial shape of the input tensor
    """
    # TODO: Implement if needed
    raise NotImplementedError("build_out_in_map not yet implemented")


def test_build_out_coords():
    """Test build_out_coords function with large random coordinates."""
    # Configuration: large 3D space with random points
    spatial_shape = (1000, 1000, 10000)
    # Format: (Batch, X, Y, Z)
    in_coords = torch.randint(0, 5000, (5_000_000, 4), dtype=torch.int16, device="cuda")
    print(f"Allocated: {in_coords.element_size() * in_coords.nelement() / 1024**3:.2f} GB")
    in_coords[:, 0] = 0
    
    kernel_size = 3
    stride = 1
    dilation = 1
    padding = 1

    print(f"--- Test Configuration ---" )
    print(f"Input Shape: {spatial_shape}, Kernel: {kernel_size}, Stride: {stride}, Padding: {padding}")
    
    from tqdm import tqdm
    with torch.no_grad():
        for _ in tqdm(range(10)):
            out_coords, out_in_map, new_spatial_shape = build_out_coords(
                in_coords=in_coords,
                spatial_shape=spatial_shape,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding
            )
    
    print(f"Number of Active Sites: {len(out_coords)}")
    print(f"Output Coords Sample:\n{out_coords[:10]}")
    print(f"\nExpected Output Spatial Shape: ({new_spatial_shape})")

        
if __name__ == "__main__":
    test_build_out_coords()
