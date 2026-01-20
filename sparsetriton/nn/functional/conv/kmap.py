import torch
import triton
import triton.language as tl
from sparsetriton.utils.hash import hash_coords_kernel, flatten_coords_kernel, HashTable, coalesce_coords, hash_coords
from sparsetriton.utils import mask_spatial_range
from typing import *

__all__ = ["get_neighbor_map", "build_kmap"]


@triton.jit
def expand_coords_kernel(
    in_ptr,           # 입력 좌표 포인터 (N, 4)
    offsets_ptr,      # 미리 계산된 오프셋 포인터 (K, 3) - (X, Y, Z)만
    out_ptr,          # 출력 좌표 포인터 (N * K, 4)
    N, K,             # N: 입력 개수, K: 커널 포인트 개수
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

    x_in = tl.load(in_ptr + n_idx * stride_in_n + 1, mask=mask)
    x_off = tl.load(offsets_ptr + k_idx * stride_off_k + 0, mask=mask)
    
    y_in = tl.load(in_ptr + n_idx * stride_in_n + 2, mask=mask)
    y_off = tl.load(offsets_ptr + k_idx * stride_off_k + 1, mask=mask)

    z_in = tl.load(in_ptr + n_idx * stride_in_n + 3, mask=mask)
    z_off = tl.load(offsets_ptr + k_idx * stride_off_k + 2, mask=mask)

    out_row_ptr = out_ptr + offs * stride_out_nk
    tl.store(out_row_ptr + 0, b_val, mask=mask)
    tl.store(out_row_ptr + 1, x_in + x_off, mask=mask)
    tl.store(out_row_ptr + 2, y_in + y_off, mask=mask)
    tl.store(out_row_ptr + 3, z_in + z_off, mask=mask)

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
    out_coords: torch.Tensor,
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
    

# def build_out_coords(
#     in_coords: torch.Tensor,
#     spatial_shape: Tuple[int, int, int],
#     kernel_size: int,
#     stride: Union[int, Tuple[int, int, int]] = 1,
#     dilation: Union[int, Tuple[int, int, int]] = 1,
#     padding: Union[int, Tuple[int, int, int]] = 1,
# ) -> torch.Tensor:
#     """
#     Builds output coordinates for sparse convolution.
    
#     Args:
#         in_coords: (N, 4) input coordinates tensor (Batch, X, Y, Z)
#         kernel_size: Size of the convolution kernel (assumed cubic)
#         stride: Stride for the convolution
#         dilation: Dilation rate for the convolution
#         padding: Padding size for the convolution
#         spatial_shape: Spatial shape of the input tensor (D, H, W)
#     """
#     if isinstance(stride, int): stride = (stride, stride, stride)
#     if isinstance(dilation, int): dilation = (dilation, dilation, dilation)
#     if isinstance(padding, int): padding = (padding, padding, padding)
#     if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size, kernel_size)
#     new_spatial_shape = []
#     for i in range(3):
#         new_spatial_shape.append(
#             (spatial_shape[i] - kernel_size[i] + 2 * padding[i]) // stride[i] + 1
#         )
    
#     out_coords = None
#     paded_coords = in_coords + torch.tensor([0, *padding], device=in_coords.device)
    
#     kernel_n = kernel_size[0] * kernel_size[1] * kernel_size[2]
#     k_offsets = - torch.tensor([0, kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2], device=in_coords.device)
#     for dk in range(kernel_n):
#         dx = ((dk // (kernel_size[-1] * kernel_size[-2])) - kernel_size[0] // 2) * dilation[0]
#         dy = (((dk // kernel_size[-1]) % kernel_size[-2]) - kernel_size[1] // 2) * dilation[1]
#         dz = ((dk % kernel_size[-1]) - kernel_size[-1] // 2) * dilation[2]
#         diff_xyz = torch.tensor([0, dx, dy, dz], dtype=in_coords.dtype, device=in_coords.device)
#         target_coords = paded_coords + (diff_xyz + k_offsets)
#         mask = torch.all(target_coords[:, 1:] % torch.tensor(stride, device=in_coords.device) == 0, dim=1)
#         target_coords = target_coords[mask]
#         if out_coords is None:
#             out_coords = target_coords
#         else:
#             out_coords = torch.concat([out_coords, target_coords], dim=0)
#         out_coords = out_coords[coalesce_coords(out_coords)]

#     for i in range(1, 4): # X, Y, Z 축 순회
#         limit = new_spatial_shape[i-1] - 1
#         mask = (out_coords[:, i] >= 0) & (out_coords[:, i] <= limit)
#         out_coords = out_coords[mask]

#     return out_coords, new_spatial_shape

def build_out_coords(
    in_coords: torch.Tensor,
    spatial_shape: Tuple[int, int, int],
    kernel_size: int,
    stride: int = 1,
    dilation: int = 1,
    padding: int = 1,
) -> torch.Tensor:
    """
    Builds output coordinates for sparse convolution.
    
    Args:
        in_coords: (N, 4) input coordinates tensor (Batch, X, Y, Z)
        kernel_size: Size of the convolution kernel (assumed cubic)
        stride: Stride for the convolution
        dilation: Dilation rate for the convolution
        padding: Padding size for the convolution
        spatial_shape: Spatial shape of the input tensor (D, H, W)
    """
    device = in_coords.device
    N = in_coords.shape[0]
    
    # --- 1. 오프셋 준비 (가벼운 연산은 PyTorch로 미리 처리) ---
    ks = torch.tensor([kernel_size] * 3, device=device)
    dil = torch.tensor([dilation] * 3, device=device)
    pad = torch.tensor([padding] * 3, device=device)

    K_N = int(torch.prod(ks).item())
    
    # 커널 오프셋 생성 (기존 로직과 동일)
    kernel_mult = torch.tensor([ks[1]*ks[2], ks[2], 1], device=device)
    k_arange = torch.arange(0, K_N, device=device).view(-1, 1).repeat(1, 3)
    base_offsets = (k_arange // kernel_mult % ks - ks // 2) * dil
    
    # Padding과 Center Shift를 미리 오프셋에 반영하여 "최종 더할 값"을 만듦
    # 로직: (in + padding) + offset - (kernel * dilation // 2)
    #    => in + (padding + offset - shift)
    shift = (ks * dil // 2)
    final_offsets = base_offsets + pad - shift # (K, 3) 형태
    
    # --- 2. Triton 커널 실행 (메모리 할당 및 좌표 확장) ---
    out_coords = torch.empty((N * K_N, 4), dtype=in_coords.dtype, device=device)
    out_range = torch.arange(N * K_N, device=device)
    grid = lambda meta: (triton.cdiv(N * K_N, meta['BLOCK_SIZE']), )
    
    expand_coords_kernel[grid](
        in_coords, final_offsets, out_coords,
        N, K_N,
        in_coords.stride(0), in_coords.stride(1),
        final_offsets.stride(0), final_offsets.stride(1),
        out_coords.stride(0), out_coords.stride(1),
        BLOCK_SIZE=1024
    )
    
    unique_indices = coalesce_coords(out_coords) 
    unique_out_in_map = out_coords[unique_indices]
    in_out_map = torch.full((len(unique_out_in_map), K_N), -1, device=device)

    hash_table = HashTable(capacity = 2 * len(unique_out_in_map), device=device)
    hash_table.insert(unique_out_in_map)
    q_out = hash_table.query(out_coords)
    in_out_map[q_out, out_range % K_N] = out_range // K_N
    del out_range
    out_coords = unique_out_in_map


    # Spatial Masking (범위 체크) - Triton 안에서 할 수도 있지만 여기서도 충분히 빠름
    # 필요하다면 이 부분도 Triton 커널 안에 'if'문으로 넣어 필터링 가능
    new_spatial_shape = torch.tensor([(s - k + 2*p) // st + 1 
                                      for s, k, p, st in zip(spatial_shape, ks, pad, [stride]*3)], 
                                     device=device)
    mask = mask_spatial_range(
        out_coords,
        x_lim=(0, new_spatial_shape[0]),
        y_lim=(0, new_spatial_shape[1]),
        z_lim=(0, new_spatial_shape[2]),
        )
    
    out_coords = out_coords[mask]
    in_out_map = in_out_map[mask]

    return out_coords, in_out_map, new_spatial_shape

def test_build_out_coords():
    # 1. 설정 (3x3x3 공간의 중심 근처에 점 2개 배치)
    spatial_shape = (1000, 1000, 1000)
    # (Batch, X, Y, Z) 형식
    in_coords = torch.randint(0, 100, (1_000_000, 4), dtype=torch.int16, device="cuda")
    in_coords[:, 0] = 0
    # in_coords = torch.tensor([
    #     [0, 0, 1, 1],
    #     [0, 0, 1, 2]
    # ], dtype=torch.long, device="cuda")
    
    kernel_size = 3
    stride = 1
    dilation = 1
    padding = 1

    print(f"--- Test Configuration ---")
    print(f"Input Shape: {spatial_shape}, Kernel: {kernel_size}, Stride: {stride}, Padding: {padding}")
    print(f"Input Coords:\n{in_coords}\n")
    from tqdm import tqdm

    for _ in tqdm(range(1000)):
        out_coords, out_in_map, new_spatial_shape = build_out_coords(
            in_coords=in_coords,
            spatial_shape=spatial_shape,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding
        )
    print(new_spatial_shape)
    # 3. 결과 출력
    print(f"--- Output Results ---")
    print(f"Number of Active Sites: {len(out_coords)}")
    print(f"Output Coords Sample:\n{out_coords[:10]}") # 상위 10개만 출력
    
    # 4. 검증 (Padding=1, Stride=1이면 출력 shape은 입력과 같아야 함)
    print(f"\nExpected Output Spatial Shape: ({new_spatial_shape})")

        
if __name__ == "__main__":
    test_build_out_coords()