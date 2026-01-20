import torch
import triton
import triton.language as tl


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
def mask_spatial_range_kernel(
    in_coordd_ptr,
    x_min, x_max,
    y_min, y_max,
    z_min, z_max,
    mask_ptr, # all True
    N,
    tune_N,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < N

    zs = tl.load(in_coordd_ptr + (idx * 4) + 3, mask=mask)
    range_mask = (zs < z_min) | (zs > z_max)
    ys = tl.load(in_coordd_ptr + idx * 4 + 2, mask)
    range_mask = range_mask | ((ys < y_min) | (ys > y_max))
    xs = tl.load(in_coordd_ptr + idx * 4 + 1, mask)
    range_mask = range_mask | ((xs < x_min) | (xs > x_max))

    tl.store(mask_ptr + idx, False, mask=range_mask & mask)


def mask_spatial_range(
    coords: torch.Tensor,
    x_lim: tuple = (None, None),
    y_lim: tuple = (None, None),
    z_lim: tuple = (None, None)
):
    N = coords.shape[0]
    mask = torch.full((N,), True, dtype=torch.bool, device=coords.device)
    if N ==0: return mask
    
    # [핵심] Dtype에 따른 Min/Max 값 추출
    # coords의 dtype을 확인하여 정수/실수 여부에 따라 극한값 설정
    if coords.dtype.is_floating_point:
        info = torch.finfo(coords.dtype)
        MIN_VAL, MAX_VAL = info.min, info.max
    else:
        info = torch.iinfo(coords.dtype)
        MIN_VAL, MAX_VAL = info.min, info.max

    # Helper: None이면 위에서 구한 MIN/MAX 사용
    def get_val(val, default_val):
        return val if val is not None else default_val

    x_min = get_val(x_lim[0], MIN_VAL)
    x_max = get_val(x_lim[1], MAX_VAL)
    y_min = get_val(y_lim[0], MIN_VAL)
    y_max = get_val(y_lim[1], MAX_VAL)
    z_min = get_val(z_lim[0], MIN_VAL)
    z_max = get_val(z_lim[1], MAX_VAL)

    if not coords.is_contiguous():
        coords = coords.contiguous()

    # Grid
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']), )
    
    mask_spatial_range_kernel[grid](
        coords,
        int(x_min), int(x_max),
        int(y_min), int(y_max),
        int(z_min), int(z_max),
        mask,
        N,
        triton.next_power_of_2(N)
    )
    return mask