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
    in_ptr,           # 입력 좌표 포인터 (N, 4)
    offsets_ptr,      # 미리 계산된 오프셋 포인터 (K, 3) - (X, Y, Z)만
    out_ptr,          # 출력 좌표 포인터 (N * K, 4)
    N, K,             # N: 입력 개수, K: 커널 포인트 개수
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


def build_out_in_map(
    in_coords: torch.Tensor,
    out_coords: torch.Tensor,
    kernel_size: int,
    dilation: int = 1,
    padding: int = 0,
    spatial_shape: Tuple[int, int, int] = None,
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

def build_kernel_offsets(kernel_size, dilation, padding, device, dtype, transposed):
    axes = [torch.arange(s, device=device) - (s // 2) for s in kernel_size]
    
    # 2. meshgrid로 모든 조합 생성 (X, Y, Z 순서 명시)
    grid = torch.meshgrid(*axes, indexing='ij') # 'ij'는 X-Y-Z 순서 유지
    
    # 3. 팽창(dilation) 적용 후 합치기
    # kernel_offsets = torch.stack(grid, dim=-1).reshape(-1, 3) * dilation
    kernel_offsets = torch.stack(grid, dim=-1).reshape(-1, 3) * dilation
    if transposed:
        kernel_offsets = kernel_offsets - padding + (kernel_size - 1) * dilation // 2
        return kernel_offsets.to(dtype)
    else:
        kernel_offsets = - kernel_offsets + padding - (kernel_size - 1) * dilation // 2
        return kernel_offsets.to(dtype)
    
def build_out_coords(
    in_coords: torch.Tensor,
    spatial_shape: Tuple[int, int, int],
    kernel_size: int,
    stride: int = 1,
    dilation: int = 1,
    padding: int = 1,
    transposed: bool = False,
    submanifold: bool = True,
    chunk_size:int = None
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
    
    ks = torch.tensor([kernel_size] * 3, device=device)
    dil = torch.tensor([dilation] * 3, device=device)
    pad = torch.tensor([padding] * 3, device=device)
    st = torch.tensor([stride] * 3, device=device)

    K_N = int(torch.prod(ks).item())
    if chunk_size is None:
        chunk_size = K_N
    kernel_offsets = build_kernel_offsets(ks, dil, pad, device, in_coords.dtype, transposed=transposed)
    if submanifold and stride == 1 and ((kernel_size -1) * dilation) // 2 == padding:
        return in_coords, spatial_shape, kernel_offsets
    if not transposed:
        new_spatial_shape = torch.tensor([(s - (k - 1) * d + 2*p - 1) // st + 1
                                        for s, k, d, p, st in zip(spatial_shape, ks, dil, pad, [stride]*3)], 
                                        device=device)
    else:
        new_spatial_shape = torch.tensor([(s - 1) * st + (k - 1) * d - 2 * p + 1
                                        for s, k, d, p, st in zip(spatial_shape, ks, dil, pad, [stride]*3)], 
                                        device=device)

    out_coords_list = []
    hash_table_size = N * K_N * 2
    global_hash_keys = torch.full((hash_table_size,), -1, dtype=torch.int32, device=device)
    
    if transposed and stride > 1:
        src_coords = in_coords.clone()
        src_coords[:, 1:] *= stride
    else:
        src_coords = in_coords


    if submanifold:
        target_offsets = pad - (ks - 1) * dil // 2
        target_offsets = target_offsets.unsqueeze(0)
        hash_table_size = N * 2
    else:
        target_offsets = kernel_offsets
        hash_table_size = N * K_N * 2

    global_hash_keys = torch.full((hash_table_size,), -1, dtype=torch.int32, device=device)
    for i in range(0, len(target_offsets), chunk_size):
        curr_offsets = target_offsets[i : i + chunk_size]
        curr_K = curr_offsets.shape[0]
        
        # 1. 현재 청크만큼만 좌표 확장 (N * CHUNK_SIZE)
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

        if not transposed:
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