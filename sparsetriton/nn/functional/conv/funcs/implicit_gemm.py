import triton
import triton.language as tl
import torch

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
        n_idx = tl.load(in_out_map + off_n * K_VOL + k, mask=mask_n, other=-1)
        valid_mask = (n_idx >= 0) & mask_n

        for c_out_off in range(0, C_out, BLOCK_SIZE_C_OUT):
            off_cout = c_out_off + tl.arange(0, BLOCK_SIZE_C_OUT)
            mask_cout = off_cout < C_out

            do_tile = tl.load(
                d_out_ptr + off_n[:, None] * C_out + off_cout[None, :],
                mask=mask_n[:, None] & mask_cout[None, :],
                other=0.0
            )

            w_tile = tl.load(
                weights_ptr + (k * C_in * C_out) + (off_cin[None, :] + off_cout[:, None] * C_in),
                mask=mask_cin[None, :] & mask_cout[:, None],
                other=0.0
            )
            acc += tl.dot(do_tile, w_tile)
    tl.store(d_features_ptr + off_n[:, None] * C_in + off_cin[None, :], 
             acc.to(d_features_ptr.dtype.element_ty), 
             mask=mask_n[:, None] & mask_cin[None, :])
    
@triton.jit
def implicit_gemm_bwd_weight_kernel(
    features_ptr, grad_output_ptr, in_out_map, d_weights_ptr,
    N_out, C_in, C_out,
    BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_C_IN: tl.constexpr, 
    BLOCK_SIZE_C_OUT: tl.constexpr, 
    K_VOL: tl.constexpr
):
    """ d_L / d_weights 계산 (Backward Weight) """
    # Grid: (K, C_in_chunks, C_out_chunks)
    pid_k = tl.program_id(0)
    pid_cin = tl.program_id(1)
    pid_cout = tl.program_id(2)

    # 1. C_in, C_out 오프셋 설정
    off_cin = pid_cin * BLOCK_SIZE_C_IN + tl.arange(0, BLOCK_SIZE_C_IN)
    off_cout = pid_cout * BLOCK_SIZE_C_OUT + tl.arange(0, BLOCK_SIZE_C_OUT)
    
    mask_cin = off_cin < C_in
    mask_cout = off_cout < C_out

    # 2. 누적할 레지스터 초기화 (Block_Cin, Block_Cout)
    acc = tl.zeros((BLOCK_SIZE_C_IN, BLOCK_SIZE_C_OUT), dtype=tl.float32)

    # 3. N(배치/공간) 차원을 따라 루프를 돌며 축소(Reduce) 수행
    # Forward 때 사용한 Rulebook(in_out_map)을 그대로 사용합니다.
    # in_out_map의 shape: (N_out, K) -> 값은 Input Index
    
    for start_n in range(0, N_out, BLOCK_SIZE_N):
        off_n = start_n + tl.arange(0, BLOCK_SIZE_N)
        mask_n = off_n < N_out

        # 현재 커널 오프셋(pid_k)에 해당하는 입력 인덱스 로딩
        n_in_indices = tl.load(in_out_map + off_n * K_VOL + pid_k, mask=mask_n, other=-1)
        
        # 유효한 연결만 계산
        valid_mask = (n_in_indices >= 0) & mask_n

        # 입력 특징(Features) 로딩: (BLOCK_N, BLOCK_C_IN)
        # Transpose를 위해 나중에 tl.trans() 사용
        f_ptrs = features_ptr + (n_in_indices[:, None] * C_in + off_cin[None, :])
        f_tile = tl.load(f_ptrs, mask=valid_mask[:, None] & mask_cin[None, :], other=0.0)

        # 출력 그래디언트(Grad Output) 로딩: (BLOCK_N, BLOCK_C_OUT)
        g_ptrs = grad_output_ptr + (off_n[:, None] * C_out + off_cout[None, :])
        g_tile = tl.load(g_ptrs, mask=mask_n[:, None] & mask_cout[None, :], other=0.0)

        # 행렬 곱: Features.T (Cin, N) x GradOut (N, Cout) -> (Cin, Cout)
        # N 차원이 사라지며 합쳐짐
        acc += tl.dot(tl.trans(f_tile), g_tile)

    # 4. 결과 저장
    # d_weights shape: (K, C_in, C_out)
    w_offset = (pid_k * C_in * C_out) + (off_cin[:, None] * C_out + off_cout[None, :])
    tl.store(d_weights_ptr + w_offset, acc.to(d_weights_ptr.dtype.element_ty), mask=mask_cin[:, None] & mask_cout[None, :])
class SparseConvImplicitGEMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, weights, in_out_map, output_size):
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
        N_out = output_size

        # 커널 실행을 위한 텐서 준비 (contiguous 필수)
        features = features.contiguous()
        weights = weights.contiguous()
        in_out_map = in_out_map.contiguous()

        # 출력 텐서 할당
        out_features = torch.zeros((N_out, C_out), device=features.device, dtype=features.dtype)

        # 튜닝 파라미터 (Block Size)
        # 실제 환경에서는 triton.autotune을 사용하는 것이 좋습니다.
        BLOCK_SIZE_N = 128
        BLOCK_SIZE_C_IN = 32
        BLOCK_SIZE_C_OUT = 32
        
        # Grid 설정
        grid = lambda META: (
            triton.cdiv(N_out, META['BLOCK_SIZE_N']),
            triton.cdiv(C_out, META['BLOCK_SIZE_C_OUT'])
        )

        implicit_gemm_fwd_kernel[grid](
            features_ptr=features,
            weights_ptr=weights,
            in_out_map=in_out_map,
            out_ptr=out_features,
            N=N_out,
            C_in=C_in,
            C_out=C_out,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_C_OUT=BLOCK_SIZE_C_OUT,
            BLOCK_SIZE_C_IN=BLOCK_SIZE_C_IN,
            K_VOL=K
        )

        # Backward를 위해 저장
        ctx.save_for_backward(features, weights, in_out_map)
        ctx.K = K
        ctx.C_in = C_in
        ctx.C_out = C_out
        ctx.N_in = N_in
        
        return out_features

    @staticmethod
    def backward(ctx, grad_output):
        features, weights, in_out_map = ctx.saved_tensors
        K = ctx.K
        C_in = ctx.C_in
        C_out = ctx.C_out
        N_in = ctx.N_in
        N_out = grad_output.shape[0]
        
        grad_output = grad_output.contiguous()
        
        # 1. d_features 계산 (기존 커널)
        # (주의: 실제로는 in_out_map의 Transpose 버전이 필요함)
        grad_input = torch.zeros((N_in, C_in), device=grad_output.device, dtype=grad_output.dtype)
        
        grid_feat = lambda META: (
            triton.cdiv(N_in, META['BLOCK_SIZE_N']),
            triton.cdiv(C_in, META['BLOCK_SIZE_C_IN'])
        )
        
        # d_features 커널 호출 (생략된 기존 코드 사용)
        implicit_gemm_bwd_feat_kernel[grid_feat](
            d_out_ptr=grad_output,
            weights_ptr=weights,
            in_out_map=in_out_map,
            d_features_ptr=grad_input,
            N=N_in,
            C_in=C_in,
            C_out=C_out,
            BLOCK_SIZE_N=128, BLOCK_SIZE_C_IN=32, BLOCK_SIZE_C_OUT=32, K_VOL=K
        )

        # 2. d_weights 계산 (새로 추가된 커널)
        grad_weights = torch.zeros_like(weights) # (K, C_in, C_out)
        
        grid_weight = lambda META: (
            K, # Grid Z: Kernel Elements
            triton.cdiv(C_in, META['BLOCK_SIZE_C_IN']),  # Grid Y: Input Channels
            triton.cdiv(C_out, META['BLOCK_SIZE_C_OUT']) # Grid X: Output Channels
        )
        
        implicit_gemm_bwd_weight_kernel[grid_weight](
            features_ptr=features,
            grad_output_ptr=grad_output,
            in_out_map=in_out_map,
            d_weights_ptr=grad_weights,
            N_out=N_out,
            C_in=C_in,
            C_out=C_out,
            BLOCK_SIZE_N=128, BLOCK_SIZE_C_IN=32, BLOCK_SIZE_C_OUT=32, K_VOL=K
        )
        
        # 3. 모든 그래디언트 반환
        return grad_input, grad_weights, None, None

# 사용하기 쉬운 Wrapper 함수
def sparse_conv(features, weights, in_out_map, output_size):
    return SparseConvImplicitGEMM.apply(features, weights, in_out_map, output_size)