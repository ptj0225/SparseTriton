import triton
import triton.language as tl
import torch
from sparsetriton.utils.hash import query_hash_table_impl, hash_coords_kernel, hash_coords_kernel2

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_C_OUT': 64, 'BLOCK_SIZE_C_IN': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_C_OUT': 64, 'BLOCK_SIZE_C_IN': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_C_OUT': 64, 'BLOCK_SIZE_C_IN': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_C_OUT': 64, 'BLOCK_SIZE_C_IN': 32}, num_warps=8)
    ],
    key=['C_in', 'C_out'],
)
@triton.jit
def implicit_gemm_hash_on_fly_fwd_kernel(
    features_ptr, weights_ptr, out_ptr,
    coords_ptr,
    k_offsets_ptr, # (K_N, 3)
    hash_table_keys_ptr, hash_table_vals_ptr,
    table_size,
    N, C_in, C_out,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_C_OUT: tl.constexpr,
    BLOCK_SIZE_C_IN: tl.constexpr, K_VOL: tl.constexpr,
    max_probe_step: tl.constexpr = 128
):
    """
    Sparse Convolution Forward Kernel (Implicit GEMM with On-the-fly Hash Lookup)

    Computes: Output = Input * Weight

    Block Shapes:
      - Input Feature Tile: (BLOCK_SIZE_N, BLOCK_SIZE_C_IN)
      - Weight Tile:        (BLOCK_SIZE_C_IN, BLOCK_SIZE_C_OUT)
      - Output Tile:        (BLOCK_SIZE_N, BLOCK_SIZE_C_OUT)

    Grid: (N / BLOCK_SIZE_N, C_out / BLOCK_SIZE_C_OUT)
    """
    pid_n = tl.program_id(0)
    pid_cout = tl.program_id(1)

    off_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)  # (BLOCK_SIZE_N,)
    off_cout = pid_cout * BLOCK_SIZE_C_OUT + tl.arange(0, BLOCK_SIZE_C_OUT)  # (BLOCK_SIZE_C_OUT,)
    
    mask_n = off_n < N  # (BLOCK_SIZE_N,)
    mask_cout = off_cout < C_out  # (BLOCK_SIZE_C_OUT,)
    b = tl.load(coords_ptr + off_n * 4, mask=mask_n)  # (BLOCK_SIZE_N,)
    x = tl.load(coords_ptr + off_n * 4 + 1, mask=mask_n)  # (BLOCK_SIZE_N,)
    y = tl.load(coords_ptr + off_n * 4 + 2, mask=mask_n)  # (BLOCK_SIZE_N,)
    z = tl.load(coords_ptr + off_n * 4 + 3, mask=mask_n)  # (BLOCK_SIZE_N,)

    acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_C_OUT), dtype=tl.float32)  # (BLOCK_SIZE_N, BLOCK_SIZE_C_OUT)
    for k in range(K_VOL):
        dx = tl.load(k_offsets_ptr + k * 3 )
        dy = tl.load(k_offsets_ptr + k * 3 + 1)
        dz = tl.load(k_offsets_ptr + k * 3 + 2)
        curr_x = x - dx  # (BLOCK_SIZE_N,)
        curr_y = y - dy  # (BLOCK_SIZE_N,)
        curr_z = z - dz  # (BLOCK_SIZE_N,)
        hashes = hash_coords_kernel(b, curr_x, curr_y, curr_z)  # (BLOCK_SIZE_N,)
        keys = hash_coords_kernel2(b, curr_x, curr_y, curr_z)  # (BLOCK_SIZE_N,)
        in_indices = query_hash_table_impl(
            hashes,
            keys,
            hash_table_keys_ptr,
            hash_table_vals_ptr,
            table_size,
            off_n,
            N,
            BLOCK_SIZE_N
        )  # (BLOCK_SIZE_N,)
        
        valid_mask = (in_indices >= 0) & mask_n & (curr_x >= 0) & (curr_y >= 0) & (curr_z >= 0)

        for c_in_off in range(0, C_in, BLOCK_SIZE_C_IN):
            off_cin = c_in_off + tl.arange(0, BLOCK_SIZE_C_IN)  # (BLOCK_SIZE_C_IN,)
            mask_cin = off_cin < C_in  # (BLOCK_SIZE_C_IN,)

            f_tile = tl.load(
                features_ptr + in_indices[:, None] * C_in + off_cin[None, :],
                mask=valid_mask[:, None] & mask_cin[None, :],
                other=0.0
            )  # (BLOCK_SIZE_N, BLOCK_SIZE_C_IN)

            w_tile = tl.load(
                weights_ptr + (k * C_in * C_out) + (off_cin[:, None] * C_out + off_cout[None, :]),
                mask=mask_cin[:, None] & mask_cout[None, :],
                other=0.0
            )  # (BLOCK_SIZE_C_IN, BLOCK_SIZE_C_OUT)

            acc = tl.dot(f_tile, w_tile, acc=acc)

    out_off = off_n[:, None] * C_out + off_cout[None, :]  # (BLOCK_SIZE_N, BLOCK_SIZE_C_OUT)
    tl.store(out_ptr + out_off, acc.to(out_ptr.dtype.element_ty), mask=mask_n[:, None] & mask_cout[None, :])

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_C_IN': 64, 'BLOCK_SIZE_C_OUT': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_C_IN': 64, 'BLOCK_SIZE_C_OUT': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_C_IN': 32, 'BLOCK_SIZE_C_OUT': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_C_IN': 64, 'BLOCK_SIZE_C_OUT': 32}, num_warps=8, num_stages=3),
    ],
    key=['C_in', 'C_out'],
    cache_results=True,
)
@triton.jit
def implicit_gemm_bwd_feat_kernel(
    d_out_ptr, weights_ptr, d_features_ptr,
    coords_ptr,
    k_offsets_ptr,
    hash_table_keys_ptr, hash_table_vals_ptr,
    table_size,
    N, C_in, C_out,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_C_IN: tl.constexpr,
    BLOCK_SIZE_C_OUT: tl.constexpr, K_VOL: tl.constexpr,
):
    """
    Sparse Convolution Backward Data Kernel (d_Input)

    Computes: d_Input += d_Output * Weight^T

    Block Shapes:
      - d_Output Tile:       (BLOCK_SIZE_N, BLOCK_SIZE_C_OUT)
      - Weight Tile:         (BLOCK_SIZE_C_IN, BLOCK_SIZE_C_OUT)
      - d_Input Accumulator: (BLOCK_SIZE_N, BLOCK_SIZE_C_IN)

    Grid: (N / BLOCK_SIZE_N, C_in / BLOCK_SIZE_C_IN)

    Note: Uses atomic_add to accumulate gradients into d_features_ptr because multiple output points might map to the same input point (scatter).
    """
    pid_n = tl.program_id(0)
    pid_cin = tl.program_id(1)

    off_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)  # (BLOCK_SIZE_N,)
    off_cin = pid_cin * BLOCK_SIZE_C_IN + tl.arange(0, BLOCK_SIZE_C_IN)  # (BLOCK_SIZE_C_IN,)

    mask_n = off_n < N  # (BLOCK_SIZE_N,)
    mask_cin = off_cin < C_in  # (BLOCK_SIZE_C_IN,)

    b = tl.load(coords_ptr + off_n * 4, mask=mask_n)  # (BLOCK_SIZE_N,)
    x = tl.load(coords_ptr + off_n * 4 + 1, mask=mask_n)  # (BLOCK_SIZE_N,)
    y = tl.load(coords_ptr + off_n * 4 + 2, mask=mask_n)  # (BLOCK_SIZE_N,)
    z = tl.load(coords_ptr + off_n * 4 + 3, mask=mask_n)  # (BLOCK_SIZE_N,)

    for k in range(K_VOL):
        dx = tl.load(k_offsets_ptr + k * 3)
        dy = tl.load(k_offsets_ptr + k * 3 + 1)
        dz = tl.load(k_offsets_ptr + k * 3 + 2)
        
        curr_x = x - dx  # (BLOCK_SIZE_N,)
        curr_y = y - dy  # (BLOCK_SIZE_N,)
        curr_z = z - dz  # (BLOCK_SIZE_N,)
        
        hashes = hash_coords_kernel(b, curr_x, curr_y, curr_z)  # (BLOCK_SIZE_N,)
        keys = hash_coords_kernel2(b, curr_x, curr_y, curr_z)  # (BLOCK_SIZE_N,)
        in_indices = query_hash_table_impl(
            hashes, keys, hash_table_keys_ptr, hash_table_vals_ptr,
            table_size, off_n, N, BLOCK_SIZE_N
        )  # (BLOCK_SIZE_N,)
        
        valid_mask = (in_indices >= 0) & mask_n & (curr_x >= 0) & (curr_y >= 0) & (curr_z >= 0)
        # valid_mask = (in_indices >= 0) & mask_n  # (BLOCK_SIZE_N,)
        
        acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_C_IN), dtype=tl.float32)  # (BLOCK_SIZE_N, BLOCK_SIZE_C_IN)
        
        for c_out_off in range(0, C_out, BLOCK_SIZE_C_OUT):
            off_cout = c_out_off + tl.arange(0, BLOCK_SIZE_C_OUT)  # (BLOCK_SIZE_C_OUT,)
            mask_cout = off_cout < C_out  # (BLOCK_SIZE_C_OUT,)
            
            do_tile = tl.load(
                d_out_ptr + off_n[:, None] * C_out + off_cout[None, :],
                mask=mask_n[:, None] & mask_cout[None, :],
                other=0.0
            )  # (BLOCK_SIZE_N, BLOCK_SIZE_C_OUT)
            
            w_tile = tl.load(
                weights_ptr + (k * C_in * C_out) + (off_cin[:, None] * C_out + off_cout[None, :]),
                mask=mask_cin[:, None] & mask_cout[None, :],
                other=0.0
            )  
            acc = tl.dot(do_tile, tl.trans(w_tile), acc=acc)
            
        target_ptrs = d_features_ptr + in_indices[:, None] * C_in + off_cin[None, :]  # (BLOCK_SIZE_N, BLOCK_SIZE_C_IN)
        tl.atomic_add(target_ptrs, acc.to(d_out_ptr.dtype.element_ty), mask=valid_mask[:, None] & mask_cin[None, :])
    
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_C_IN': 64, 'BLOCK_SIZE_C_OUT': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_C_IN': 64, 'BLOCK_SIZE_C_OUT': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_C_IN': 64, 'BLOCK_SIZE_C_OUT': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_C_IN': 64, 'BLOCK_SIZE_C_OUT': 32}, num_warps=8, num_stages=3),
    ],
    key=['C_in', 'C_out'],
    cache_results=True,
)
@triton.jit
def implicit_gemm_bwd_weight_kernel(
    features_ptr, d_out_ptr, d_weights_ptr,
    coords_ptr,
    k_offsets_ptr,
    hash_table_keys_ptr, hash_table_vals_ptr,
    table_size,
    N, C_in, C_out,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_C_IN: tl.constexpr,
    BLOCK_SIZE_C_OUT: tl.constexpr, K_VOL: tl.constexpr, 
):
    """
    Sparse Convolution Backward Weight Kernel (d_Weight)

    Computes: d_Weight = Input^T * d_Output

    Block Shapes:
      - Input Feature Tile:   (BLOCK_SIZE_N, BLOCK_SIZE_C_IN)
      - d_Output Tile:        (BLOCK_SIZE_N, BLOCK_SIZE_C_OUT)
      - d_Weight Accumulator: (BLOCK_SIZE_C_IN, BLOCK_SIZE_C_OUT)

    Grid: (N / BLOCK_SIZE_N, K, (C_in / BLOCK_SIZE_C_IN) * (C_out / BLOCK_SIZE_C_OUT))

    Note: Parallelizes over N, accumulating gradients into d_weights via atomic_add.
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_c = tl.program_id(2)

    num_pid_cout = (C_out + BLOCK_SIZE_C_OUT - 1) // BLOCK_SIZE_C_OUT
    pid_cin = pid_c // num_pid_cout
    pid_cout = pid_c % num_pid_cout
    
    off_cin = pid_cin * BLOCK_SIZE_C_IN + tl.arange(0, BLOCK_SIZE_C_IN)  # (BLOCK_SIZE_C_IN,)
    off_cout = pid_cout * BLOCK_SIZE_C_OUT + tl.arange(0, BLOCK_SIZE_C_OUT)  # (BLOCK_SIZE_C_OUT,)
    
    mask_cin = off_cin < C_in  # (BLOCK_SIZE_C_IN,)
    mask_cout = off_cout < C_out  # (BLOCK_SIZE_C_OUT,)
    
    dx = tl.load(k_offsets_ptr + pid_k * 3)
    dy = tl.load(k_offsets_ptr + pid_k * 3 + 1)
    dz = tl.load(k_offsets_ptr + pid_k * 3 + 2)
    
    off_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)  # (BLOCK_SIZE_N,)
    mask_n = off_n < N  # (BLOCK_SIZE_N,)
    
    b = tl.load(coords_ptr + off_n * 4, mask=mask_n)  # (BLOCK_SIZE_N,)
    x = tl.load(coords_ptr + off_n * 4 + 1, mask=mask_n)  # (BLOCK_SIZE_N,)
    y = tl.load(coords_ptr + off_n * 4 + 2, mask=mask_n)  # (BLOCK_SIZE_N,)
    z = tl.load(coords_ptr + off_n * 4 + 3, mask=mask_n)  # (BLOCK_SIZE_N,)
    
    curr_x = x - dx  # (BLOCK_SIZE_N,)
    curr_y = y - dy  # (BLOCK_SIZE_N,)
    curr_z = z - dz  # (BLOCK_SIZE_N,)
    
    hashes = hash_coords_kernel(b, curr_x, curr_y, curr_z)  # (BLOCK_SIZE_N,)
    keys = hash_coords_kernel2(b, curr_x, curr_y, curr_z)  # (BLOCK_SIZE_N,)
    in_indices = query_hash_table_impl(
        hashes, keys, hash_table_keys_ptr, hash_table_vals_ptr,
        table_size, off_n, N, BLOCK_SIZE_N
    )  # (BLOCK_SIZE_N,)
    
    # valid_mask = (in_indices >= 0) & mask_n  # (BLOCK_SIZE_N,)
    valid_mask = (in_indices != -1) & mask_n & (curr_x >= 0) & (curr_y >= 0) & (curr_z >= 0)
    
    f_tile = tl.load(
        features_ptr + in_indices[:, None] * C_in + off_cin[None, :],
        mask=valid_mask[:, None] & mask_cin[None, :],
        other=0.0
    )  # (BLOCK_SIZE_N, BLOCK_SIZE_C_IN)
    
    do_tile = tl.load(
        d_out_ptr + off_n[:, None] * C_out + off_cout[None, :],
        mask=mask_n[:, None] & mask_cout[None, :],
        other=0.0
    )  # (BLOCK_SIZE_N, BLOCK_SIZE_C_OUT)
    
    acc = tl.dot(tl.trans(f_tile), do_tile)
    
    w_offset = (pid_k * C_in * C_out) + (off_cin[:, None] * C_out + off_cout[None, :])  # (BLOCK_SIZE_C_IN, BLOCK_SIZE_C_OUT)
    tl.atomic_add(d_weights_ptr + w_offset, acc.to(d_weights_ptr.dtype.element_ty), mask=mask_cin[:, None] & mask_cout[None, :])

class ConvHashOnTheFlyImplicitGEMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, weights, coords, kernel_offsets, hash_table_keys, hash_table_vals):
        """
        Args:
            features: (N_in, C_in) 입력 특징 텐서
            weights: (K, C_in, C_out) 가중치 텐서
            in_out_map: (N_out, K) 출력-입력 매핑 테이블 (Rulebook)
            output_size: 출력 텐서의 N 크기 (N_out)
        """
        # 입력 검증 및 차원 확인
        N_in, C_in = features.shape
        K, _, C_out = weights.shape
        N_out = coords.shape[0]
    
        # 커널 실행을 위한 텐서 준비 (contiguous 필수)
        features = features.contiguous()
        weights = weights.contiguous()
        coords = coords.contiguous()
        kernel_offsets = kernel_offsets.contiguous()
        # 출력 텐서 할당
        out_features = torch.zeros((N_out, C_out), device=features.device, dtype=features.dtype)

        
        # Grid 설정
        grid = lambda META: (
            triton.cdiv(N_out, META['BLOCK_SIZE_N']),
            triton.cdiv(C_out, META['BLOCK_SIZE_C_OUT'])
        )

        implicit_gemm_hash_on_fly_fwd_kernel[grid](
            features_ptr=features,
            weights_ptr=weights,
            out_ptr=out_features,
            coords_ptr=coords,
            k_offsets_ptr=kernel_offsets,
            hash_table_keys_ptr=hash_table_keys,
            hash_table_vals_ptr=hash_table_vals,
            table_size=len(hash_table_keys),
            N=N_out,
            C_in=C_in,
            C_out=C_out,
            K_VOL=K,
        )

        ctx.save_for_backward(features, weights, coords, kernel_offsets, hash_table_keys, hash_table_vals)
        ctx.K = K
        ctx.C_in = C_in
        ctx.C_out = C_out
        ctx.N_out = N_out
        
        return out_features

    @staticmethod
    def backward(ctx, grad_output):
        features, weights, coords, kernel_offsets, hash_table_keys, hash_table_vals = ctx.saved_tensors
        K = ctx.K
        C_in = ctx.C_in
        C_out = ctx.C_out
        N_out = ctx.N_out
        kernel_offsets = kernel_offsets.contiguous()
        weights = weights.contiguous()
        grad_output = grad_output.contiguous()
        hash_table_keys = hash_table_keys.contiguous()
        hash_table_vals = hash_table_vals.contiguous()
        # 1. d_features
        grad_input = torch.zeros_like(features).contiguous()
        
        grid_feat = lambda META: (
            triton.cdiv(N_out, META['BLOCK_SIZE_N']),
            triton.cdiv(C_in, META['BLOCK_SIZE_C_IN'])
        )
        
        implicit_gemm_bwd_feat_kernel[grid_feat](
            d_out_ptr=grad_output,
            weights_ptr=weights,
            d_features_ptr=grad_input,
            coords_ptr=coords,
            k_offsets_ptr=kernel_offsets,
            hash_table_keys_ptr=hash_table_keys,
            hash_table_vals_ptr=hash_table_vals,
            table_size=len(hash_table_keys),
            N=N_out, C_in=C_in, C_out=C_out,
            K_VOL=K,
        )

        # 2. d_weights
        grad_weights = torch.zeros_like(weights).contiguous()
        grid_weight = lambda META: (
            triton.cdiv(N_out, META['BLOCK_SIZE_N']),
            K,
            triton.cdiv(C_in, META['BLOCK_SIZE_C_IN']) * triton.cdiv(C_out, META['BLOCK_SIZE_C_OUT'])
        )
        
        implicit_gemm_bwd_weight_kernel[grid_weight](
            features_ptr=features,
            d_out_ptr=grad_output,
            d_weights_ptr=grad_weights,
            coords_ptr=coords,
            k_offsets_ptr=kernel_offsets,
            hash_table_keys_ptr=hash_table_keys,
            hash_table_vals_ptr=hash_table_vals,
            table_size=len(hash_table_keys),
            N=N_out, C_in=C_in, C_out=C_out,
            K_VOL=K,
        )
        
        return grad_input, grad_weights, None, None, None, None
        
    

if __name__ == "__main__":
    from sparsetriton.tensor import randn
    from sparsetriton.nn.functional.conv.kmap import build_out_coords, build_kernel_offsets
    from sparsetriton.utils.hash import HashTable
    from torch.optim import Adam
    
    device = "cuda"
    test_tensor = randn(batch_size=10, spatial_shape=(512, 512, 512), nnz=13421772 // 100, channels=16, device=device)
    stride = torch.tensor([1] * 3, device=device)
    pad = torch.tensor([1] * 3, device=device)
    ks = torch.tensor([3] * 3, device=device)
    dilation = torch.tensor([1] * 3, device=device)
    weights = torch.rand((27, 16, 16), device=device, dtype=torch.float)
    weights = torch.nn.Parameter(weights)
    bias = torch.rand((16), device=device, dtype=torch.float)
    ht = HashTable(test_tensor.C.__len__() * 4, device=device)
    ht.insert(test_tensor.C)
    optim = Adam([weights, bias], lr=0.1)
    out_coords, _ = build_out_coords(test_tensor.C, test_tensor.spatial_shape, 3, 1, 1, 1)
    # torch.cuda.empty_cache()
    kernel_offsets = build_kernel_offsets(device=device, dtype=test_tensor.C.dtype, dilation=dilation, kernel_size=ks, padding=pad)
    # print(out_coords.shape)
    from tqdm import tqdm
    for _ in tqdm(range(1000)):
        test_tensor = randn(batch_size=10, spatial_shape=(512, 512, 512), nnz=13421772 // 100, channels=16, device=device)
        out_coords, _ = build_out_coords(test_tensor.C, test_tensor.spatial_shape, 3, 1, 1, 1)
        ht = HashTable(test_tensor.C.__len__() * 4, device=device)
        ht.insert(test_tensor.C)
        result = ConvHashOnTheFlyImplicitGEMM.apply(
            test_tensor.F, weights, out_coords, kernel_offsets,
            ht.table_keys, ht.table_values
        )
        result = result + bias
        optim.zero_grad()
        (result-3).pow(2).mean().backward()
        print(result.abs().mean().item())
        print((result-3).pow(2).mean().item())
        optim.step()
    print(result)
