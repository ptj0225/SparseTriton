import triton
import triton.language as tl

@triton.jit
def implicit_gemm_fwd_kernel(
    features_ptr, weights_ptr, in_out_map, out_ptr,
    N, C_in, C_out,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_C_OUT: tl.constexpr,
    BLOCK_SIZE_C_IN: tl.constexpr, K_VOL: tl.constexpr
):
    """
    Sparse Convolution Implicit GEMM Kernel
    """
    pid_n = tl.program_id(0)
    pid_cout = tl.program_id(1)

    off_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    off_cout = pid_cout * BLOCK_SIZE_C_OUT + tl.arange(0, BLOCK_SIZE_C_OUT)
    
    mask_n = off_n < N
    mask_cout = off_cout < C_out

    acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_C_OUT), dtype=tl.float32)

    for k in range(K_VOL):
        n_idx = tl.load(in_out_map + off_n * K_VOL + k, mask=mask_n, other=-1)
        
        valid_mask = (n_idx >= 0) & mask_n

        for c_in_off in range(0, C_in, BLOCK_SIZE_C_IN):
            off_cin = c_in_off + tl.arange(0, BLOCK_SIZE_C_IN)
            mask_cin = off_cin < C_in

            f_tile = tl.load(
                features_ptr + n_idx[:, None] * C_in + off_cin[None, :],
                mask=valid_mask[:, None] & mask_cin[None, :],
                other=0.0
            )

            w_tile = tl.load(
                weights_ptr + (k * C_in * C_out) + (off_cin[:, None] * C_out + off_cout[None, :]),
                mask=mask_cin[:, None] & mask_cout[None, :],
                other=0.0
            )

            acc += tl.dot(f_tile, w_tile)

    out_off = off_n[:, None] * C_out + off_cout[None, :]
    tl.store(out_ptr + out_off, acc.to(out_ptr.dtype.element_ty), mask=mask_n[:, None] & mask_cout[None, :])


@triton.jit
def implicit_gemm_bwd_feat_kernel(
    d_out_ptr, weights_ptr, in_out_map, d_features_ptr,
    N, C_in, C_out,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_C_IN: tl.constexpr,
    BLOCK_SIZE_C_OUT: tl.constexpr, K_VOL: tl.constexpr
):
    """ d_L / d_features 계산 (Backward Input) """
    pid_n = tl.program_id(0)
    pid_cin = tl.program_id(1)

    off_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    off_cin = pid_cin * BLOCK_SIZE_C_IN + tl.arange(0, BLOCK_SIZE_C_IN)
    
    mask_n = off_n < N
    mask_cin = off_cin < C_in

    acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_C_IN), dtype=tl.float32)

    for k in range(K_VOL):
        # Forward에서 썼던 k의 대응되는 인덱스 로드
        n_idx = tl.load(in_out_map + off_n * K_VOL + k, mask=mask_n, other=-1)
        valid_mask = (n_idx >= 0) & mask_n

        for c_out_off in range(0, C_out, BLOCK_SIZE_C_OUT):
            off_cout = c_out_off + tl.arange(0, BLOCK_SIZE_C_OUT)
            mask_cout = off_cout < C_out

            # d_out 로드: (N, C_out)
            do_tile = tl.load(
                d_out_ptr + off_n[:, None] * C_out + off_cout[None, :],
                mask=mask_n[:, None] & mask_cout[None, :],
                other=0.0
            )

            # Weight 로드 및 Transpose 효과: (C_in, C_out) -> (C_out, C_in)를 dot에서 처리
            # Weight shape: (K_VOL, C_in, C_out)
            w_tile = tl.load(
                weights_ptr + (k * C_in * C_out) + (off_cin[None, :] + off_cout[:, None] * C_in),
                mask=mask_cin[None, :] & mask_cout[:, None],
                other=0.0
            )

            # d_out (N, Cout) @ W.T (Cout, Cin) = d_feat (N, Cin)
            acc += tl.dot(do_tile, w_tile)

    # d_features는 중복 업데이트가 발생할 수 있으므로 상황에 따라 atomic_add 필요 
    # (단, Subm에선 n_idx가 고유하므로 일반 store 가능)
    tl.store(d_features_ptr + off_n[:, None] * C_in + off_cin[None, :], 
             acc.to(d_features_ptr.dtype.element_ty), 
             mask=mask_n[:, None] & mask_cin[None, :])