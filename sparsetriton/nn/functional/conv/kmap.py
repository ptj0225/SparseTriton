import torch
import triton
import triton.language as tl
from sparsetriton.utils.hash import hash_coords_kernel, flatten_coords_kernel, HashTable, coalesce_coords, hash_coords
from sparsetriton.utils import mask_spatial_range
from typing import *

__all__ = ["get_neighbor_map", "build_out_coords"]


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['K'],
)
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
def filter_unique_kernel(
    coords_ptr,
    hash_keys_ptr,
    mask_ptr,
    table_size,
    N,
    BLOCK_SIZE: tl.constexpr
):
    """
    전역 해시 테이블을 사용하여 현재 좌표 뭉치 중 '처음 발견된' 것들만 마스킹합니다.
    """
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < N

    b = tl.load(coords_ptr + idx * 4 + 0, mask=mask)
    x = tl.load(coords_ptr + idx * 4 + 1, mask=mask)
    y = tl.load(coords_ptr + idx * 4 + 2, mask=mask)
    z = tl.load(coords_ptr + idx * 4 + 3, mask=mask)

    # 좌표를 64비트 키로 압축
    h = hash_coords_kernel(b, x, y, z)

    is_unique = tl.zeros((BLOCK_SIZE,), dtype=tl.int1)
    active = mask
    step = 0
    while (tl.max(active.to(tl.int32), axis=0) > 0) & (step < 1024):
        curr_h = (h + step) % table_size
        cmp_val = tl.where(active, tl.cast(-1, tl.int32), tl.cast(-2, tl.int32))
        old = tl.atomic_cas(hash_keys_ptr + curr_h, cmp_val, h)
        
        is_unique = is_unique | (active & (old == -1))
        
        done = (old == -1) | (old == h)
        active = active & (~done)
        step += 1
            
    tl.store(mask_ptr + idx, is_unique, mask=mask)

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

def build_kernel_offsets(kernel_size, dilation, device, dtype):
    axes = [torch.arange(s, device=device) - (s // 2) for s in kernel_size]
    
    # 2. meshgrid로 모든 조합 생성 (X, Y, Z 순서 명시)
    grid = torch.meshgrid(*axes, indexing='ij') # 'ij'는 X-Y-Z 순서 유지
    
    # 3. 팽창(dilation) 적용 후 합치기
    # kernel_offsets = torch.stack(grid, dim=-1).reshape(-1, 3) * dilation
    kernel_offsets = torch.stack(grid, dim=-1).flip(-1).reshape(-1, 3) * dilation
    return kernel_offsets.to(dtype)
    
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
    st = torch.tensor([stride] * 3, device=device)

    K_N = int(torch.prod(ks).item())
    kernel_offsets = build_kernel_offsets(ks, dil, device, in_coords.dtype)
    
    new_spatial_shape = torch.tensor([(s - k + 2*p) // st + 1 
                                      for s, k, p, st in zip(spatial_shape, ks, pad, [stride]*3)], 
                                     device=device)

    out_coords_list = []
    # 전역 해시 테이블을 사용하여 모든 청크에 대해 중복을 체크함
    # N * 10 정도면 충돌이 매우 적어 효율적임
    hash_table_size = N * K_N * 2
    global_hash_keys = torch.full((hash_table_size,), -1, dtype=torch.int32, device=device)
    
    CHUNK_SIZE = 1
    
    for i in range(0, K_N, CHUNK_SIZE):
        curr_offsets = kernel_offsets[i : i + CHUNK_SIZE]
        curr_K = curr_offsets.shape[0]
        
        # 1. 현재 청크만큼만 좌표 확장 (N * CHUNK_SIZE)
        chunk_out = torch.empty((N * curr_K, 4), dtype=in_coords.dtype, device=device)
        grid = lambda meta: (triton.cdiv(N * curr_K, meta['BLOCK_SIZE']), )
        expand_coords_kernel[grid](
            in_coords, curr_offsets, chunk_out,
            N, curr_K,
            in_coords.stride(0), in_coords.stride(1),
            curr_offsets.stride(0), curr_offsets.stride(1),
            chunk_out.stride(0), chunk_out.stride(1)
        )
        # 2. 스트라이드 필터링 및 다운샘플링
        if stride > 1:
            mask_st = torch.all(chunk_out[:, 1:] % stride == 0, dim=1)
            chunk_out = chunk_out[mask_st]
            
        chunk_out[:, 1:] //= stride
        
        # 3. 공간 범위 필터링 (유효한 인덱스만 남김)
        mask_range = mask_spatial_range(
            chunk_out, 
            (0, new_spatial_shape[0]-1), 
            (0, new_spatial_shape[1]-1), 
            (0, new_spatial_shape[2]-1)
        )
        chunk_out = chunk_out[mask_range]
        
        if chunk_out.shape[0] > 0:
            # 전역 해시 테이블을 사용하여 이전에 발견되지 않은 새로운 좌표만 필터링
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
        return torch.empty((0, 4), dtype=in_coords.dtype, device=device), new_spatial_shape

    # 이미 전역적으로 중복이 제거되었으므로 단 한 번의 cat만 수행하여 메모리 피크 최소화
    final_coords = torch.cat(out_coords_list, dim=0)
    return final_coords, new_spatial_shape

def test_build_out_coords():
    # 1. 설정 (3x3x3 공간의 중심 근처에 점 2개 배치)
    spatial_shape = (1000, 1000, 10000)
    # (Batch, X, Y, Z) 형식
    in_coords = torch.randint(0, 5000, (5_000_000, 4), dtype=torch.int16, device="cuda")
    print(f"Allocated: {in_coords.element_size() * in_coords.nelement() / 1024**3:.2f} GB")
    in_coords[:, 0] = 0
    # in_coords = torch.tensor([
    #     [0, 0, 1, 1],
    #     [0, 0, 1, 2]
    # ], dtype=torch.long, device="cuda")
    
    kernel_size = 3
    stride = 1
    dilation = 1
    padding = 1

    print(f"--- Test Configuration ---" )
    print(f"Input Shape: {spatial_shape}, Kernel: {kernel_size}, Stride: {stride}, Padding: {padding}")
    print(f"Input Coords:\n{in_coords}\n")
    from tqdm import tqdm
    with torch.no_grad():
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