import torch
import triton
import triton.language as tl
from sparsetriton.utils.hash import hash_coords_kernel, flatten_coords_kernel, HashTable
from typing import *

__all__ = ["get_neighbor_map", "build_kmap"]


@triton.jit
def get_neighbor_map_kernel(
    coords_ptr, hash_keys_ptr, hash_vals_ptr, neighbor_map_ptr,
    N, table_size, 
    kernel_size: tl.constexpr, 
    BLOCK_SIZE: tl.constexpr
):
    """
    For each voxel, find indices of its 27 neighbors using the hash table.
    """
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < N

    b = tl.load(coords_ptr + idx * 4 + 0, mask=mask)
    x = tl.load(coords_ptr + idx * 4 + 1, mask=mask)
    y = tl.load(coords_ptr + idx * 4 + 2, mask=mask)
    z = tl.load(coords_ptr + idx * 4 + 3, mask=mask)

    for dk in range(kernel_size ** 3):
        dx = (dk // kernel_size**2) - kernel_size // 2
        dy = ((dk // kernel_size) % kernel_size) - kernel_size // 2
        dz = (dk % kernel_size) - kernel_size // 2

        nx, ny, nz = x + dx, y + dy, z + dz
        n_hash = hash_coords_kernel(b, nx, ny, nz) % table_size
        n_key = flatten_coords_kernel(b, nx, ny, nz)
        # Probe hash table
        found_idx = -1
        active_mask = mask
        probe_step = 0
        while (tl.max(active_mask.to(tl.int32), axis=0) > 0 & (probe_step < 32)):
            curr_hash = (n_hash + probe_step) % table_size
            k = tl.load(hash_keys_ptr + curr_hash, mask=active_mask)
            v = tl.load(hash_vals_ptr + curr_hash, mask=active_mask)
            active_mask = active_mask & (k == n_key)
            tl.store(neighbor_map_ptr + idx * kernel_size**3 + dk, v, mask=active_mask)
            probe_step += 1


def get_neighbor_map(coords: torch.Tensor, hash_table: HashTable, kernel_size: int) -> torch.Tensor:
    """
    Generates a neighbor map for sparse convolution.
    
    Args:
        coords: (N, 4) coordinates tensor (Batch, X, Y, Z)
        hash_table: HashTable object containing the sparse tensor coordinates
        kernel_size: Size of the convolution kernel (assumed cubic)
        
    Returns:
        neighbor_map: (N, kernel_volume) tensor containing indices of neighbors. 
                      -1 indicates no neighbor.
    """
    N = coords.shape[0]
    kernel_vol = kernel_size ** 3
    neighbor_map = torch.full((N, kernel_vol), -1, dtype=torch.int64, device=coords.device)
    
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    
    get_neighbor_map_kernel[grid](
        coords, 
        hash_table.table_keys, 
        hash_table.table_values, 
        neighbor_map,
        N, 
        hash_table.capacity,
        kernel_size=kernel_size,
        BLOCK_SIZE=128
    )
    return neighbor_map


def build_out_in_map(
    in_coords: torch.Tensor,
    kernel_size: int,
    dilation: int = 1,
    padding: int = 0,
    spatial_shape: Tuple[int, int, int] = None,
    submanifold: bool = True
) -> torch.Tensor:
    """
    Builds input-output coordinate mapping for sparse convolution.
    
    Args:
        in_coords: (N, 4) input coordinates tensor (Batch, X, Y, Z)
        kernel_size: Size of the convolution kernel (assumed cubic)
        dilation: Dilation rate for the convolution
        padding: Padding size for the convolution
        spatial_shape: Spatial shape of the input tensor (D, H, W)
    """
    if submanifold:
        return in_coords
