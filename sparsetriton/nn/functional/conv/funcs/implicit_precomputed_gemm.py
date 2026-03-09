"""Pre-computed neighbor indices sparse convolution using Triton.

This module provides sparse convolution kernels that use pre-computed
neighbor indices instead of on-the-fly hash lookup for better performance.
"""

import triton
import triton.language as tl
import torch

_ROCM_MODE = torch.version.hip is not None

_FWD_CONFIGS = [
    # Stage 2 configs (less shared memory)
    triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_C_OUT': 16, 'BLOCK_SIZE_C_IN': 16}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_C_OUT': 16, 'BLOCK_SIZE_C_IN': 16}, num_warps=4, num_stages=2),
    # Stage 3 configs (balanced)
    triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_C_OUT': 32, 'BLOCK_SIZE_C_IN': 16}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_C_OUT': 32, 'BLOCK_SIZE_C_IN': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_C_OUT': 32, 'BLOCK_SIZE_C_IN': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_C_OUT': 64, 'BLOCK_SIZE_C_IN': 32}, num_warps=4, num_stages=3),
    # Stage 4 configs (more shared memory, better pipelining)
    triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_C_OUT': 32, 'BLOCK_SIZE_C_IN': 32}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_C_OUT': 32, 'BLOCK_SIZE_C_IN': 32}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_C_OUT': 64, 'BLOCK_SIZE_C_IN': 32}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_C_OUT': 64, 'BLOCK_SIZE_C_IN': 64}, num_warps=4, num_stages=4),
    # Stage 5 configs (aggressive pipelining)
    triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_C_OUT': 32, 'BLOCK_SIZE_C_IN': 32}, num_warps=4, num_stages=5),
    triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_C_OUT': 32, 'BLOCK_SIZE_C_IN': 32}, num_warps=4, num_stages=5),
    triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_C_OUT': 64, 'BLOCK_SIZE_C_IN': 32}, num_warps=8, num_stages=5),
    # Larger block configs
    triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_C_OUT': 32, 'BLOCK_SIZE_C_IN': 32}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_C_OUT': 64, 'BLOCK_SIZE_C_IN': 32}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_C_OUT': 64, 'BLOCK_SIZE_C_IN': 32}, num_warps=8, num_stages=5),
]

_BWD_FEAT_CONFIGS = [
    triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_C_IN': 16, 'BLOCK_SIZE_C_OUT': 16}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_C_IN': 16, 'BLOCK_SIZE_C_OUT': 16}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_C_IN': 32, 'BLOCK_SIZE_C_OUT': 16}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_C_IN': 32, 'BLOCK_SIZE_C_OUT': 32}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_C_IN': 32, 'BLOCK_SIZE_C_OUT': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_C_IN': 64, 'BLOCK_SIZE_C_OUT': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_C_IN': 64, 'BLOCK_SIZE_C_OUT': 64}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_C_IN': 32, 'BLOCK_SIZE_C_OUT': 32}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_C_IN': 64, 'BLOCK_SIZE_C_OUT': 32}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_C_IN': 64, 'BLOCK_SIZE_C_OUT': 64}, num_warps=8, num_stages=4),
]

_BWD_WEIGHT_CONFIGS = [
    triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_C_IN': 16, 'BLOCK_SIZE_C_OUT': 16}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_C_IN': 16, 'BLOCK_SIZE_C_OUT': 16}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_C_IN': 16, 'BLOCK_SIZE_C_OUT': 32}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_C_IN': 32, 'BLOCK_SIZE_C_OUT': 32}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_C_IN': 32, 'BLOCK_SIZE_C_OUT': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_C_IN': 64, 'BLOCK_SIZE_C_OUT': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_C_IN': 64, 'BLOCK_SIZE_C_OUT': 64}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_C_IN': 32, 'BLOCK_SIZE_C_OUT': 32}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_C_IN': 64, 'BLOCK_SIZE_C_OUT': 32}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_C_IN': 64, 'BLOCK_SIZE_C_OUT': 64}, num_warps=8, num_stages=4),
]


@triton.autotune(
    configs=_FWD_CONFIGS,
    key=['C_in', 'C_out'],
)
@triton.jit
def precomputed_fwd_kernel(
    features_ptr, weights_ptr, out_ptr,
    neighbor_indices_ptr,  # (N_out, K_vol) - precomputed neighbor indices
    N_out, C_in, C_out, K_VOL: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_C_OUT: tl.constexpr, BLOCK_SIZE_C_IN: tl.constexpr,
):
    """Forward kernel using pre-computed neighbor indices.
    
    Optimized with:
    - Weight pipelining via num_stages
    - Efficient feature gathering
    - Coalesced memory access
    """
    pid_n = tl.program_id(0)
    pid_cout = tl.program_id(1)

    off_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    off_cout = pid_cout * BLOCK_SIZE_C_OUT + tl.arange(0, BLOCK_SIZE_C_OUT)
    
    mask_n = off_n < N_out
    mask_cout = off_cout < C_out

    acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_C_OUT), dtype=tl.float32)

    for k in range(K_VOL):
        # Load neighbor indices for this kernel position
        neighbor_idx = tl.load(neighbor_indices_ptr + off_n * K_VOL + k, mask=mask_n, other=-1)
        valid_mask = (neighbor_idx >= 0) & mask_n
        
        # Early skip if no valid neighbors in this block
        # (Triton will optimize this out if all threads have valid neighbors)
        
        for c_in_off in range(0, C_in, BLOCK_SIZE_C_IN):
            off_cin = c_in_off + tl.arange(0, BLOCK_SIZE_C_IN)
            mask_cin = off_cin < C_in

            # Load features from neighbor indices (gather)
            # Use clamped indices to avoid OOB access
            neighbor_idx_clamped = tl.where(valid_mask, neighbor_idx, 0)
            feat_ptrs = features_ptr + neighbor_idx_clamped[:, None] * C_in + off_cin[None, :]
            f_tile = tl.load(
                feat_ptrs,
                mask=valid_mask[:, None] & mask_cin[None, :],
                other=0.0
            )

            # Load weights - these will be cached in shared memory via num_stages
            w_ptrs = weights_ptr + (k * C_in * C_out) + (off_cin[:, None] * C_out + off_cout[None, :])
            w_tile = tl.load(
                w_ptrs,
                mask=mask_cin[:, None] & mask_cout[None, :],
                other=0.0
            )

            acc = tl.dot(f_tile, w_tile, acc=acc)

    out_off = off_n[:, None] * C_out + off_cout[None, :]
    tl.store(out_ptr + out_off, acc.to(out_ptr.dtype.element_ty), mask=mask_n[:, None] & mask_cout[None, :])


@triton.autotune(
    configs=_BWD_FEAT_CONFIGS,
    key=['C_in', 'C_out'],
)
@triton.jit
def precomputed_bwd_feat_kernel(
    d_out_ptr, weights_ptr, d_features_ptr,
    neighbor_indices_ptr,
    N_out, C_in, C_out, K_VOL: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_C_IN: tl.constexpr, BLOCK_SIZE_C_OUT: tl.constexpr,
):
    """Backward kernel for features using pre-computed neighbor indices."""
    pid_n = tl.program_id(0)
    pid_cin = tl.program_id(1)

    off_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    off_cin = pid_cin * BLOCK_SIZE_C_IN + tl.arange(0, BLOCK_SIZE_C_IN)

    mask_n = off_n < N_out
    mask_cin = off_cin < C_in

    for k in range(K_VOL):
        neighbor_idx = tl.load(neighbor_indices_ptr + off_n * K_VOL + k, mask=mask_n, other=-1)
        valid_mask = (neighbor_idx >= 0) & mask_n
        
        acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_C_IN), dtype=tl.float32)
        
        for c_out_off in range(0, C_out, BLOCK_SIZE_C_OUT):
            off_cout = c_out_off + tl.arange(0, BLOCK_SIZE_C_OUT)
            mask_cout = off_cout < C_out
            
            do_tile = tl.load(
                d_out_ptr + off_n[:, None] * C_out + off_cout[None, :],
                mask=mask_n[:, None] & mask_cout[None, :],
                other=0.0
            )
            
            w_tile = tl.load(
                weights_ptr + (k * C_in * C_out) + (off_cin[:, None] * C_out + off_cout[None, :]),
                mask=mask_cin[:, None] & mask_cout[None, :],
                other=0.0
            )
            acc = tl.dot(do_tile, tl.trans(w_tile), acc=acc)
        
        # Scatter gradients using atomic add
        neighbor_idx_safe = tl.where(valid_mask, neighbor_idx, 0)
        target_ptrs = d_features_ptr + neighbor_idx_safe[:, None] * C_in + off_cin[None, :]
        tl.atomic_add(target_ptrs, acc.to(d_out_ptr.dtype.element_ty), mask=valid_mask[:, None] & mask_cin[None, :])


@triton.autotune(
    configs=_BWD_WEIGHT_CONFIGS,
    key=['C_in', 'C_out'],
)
@triton.jit
def precomputed_bwd_weight_kernel(
    features_ptr, d_out_ptr, d_weights_ptr,
    neighbor_indices_ptr,
    N_out, C_in, C_out, K_VOL: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_C_IN: tl.constexpr, BLOCK_SIZE_C_OUT: tl.constexpr,
):
    """Backward kernel for weights using pre-computed neighbor indices."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_c = tl.program_id(2)

    num_pid_cout = (C_out + BLOCK_SIZE_C_OUT - 1) // BLOCK_SIZE_C_OUT
    pid_cin = pid_c // num_pid_cout
    pid_cout = pid_c % num_pid_cout
    
    off_cin = pid_cin * BLOCK_SIZE_C_IN + tl.arange(0, BLOCK_SIZE_C_IN)
    off_cout = pid_cout * BLOCK_SIZE_C_OUT + tl.arange(0, BLOCK_SIZE_C_OUT)
    
    mask_cin = off_cin < C_in
    mask_cout = off_cout < C_out
    
    off_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_n = off_n < N_out
    
    neighbor_idx = tl.load(neighbor_indices_ptr + off_n * K_VOL + pid_k, mask=mask_n, other=-1)
    valid_mask = (neighbor_idx >= 0) & mask_n
    
    neighbor_idx_safe = tl.where(valid_mask, neighbor_idx, 0)
    
    f_tile = tl.load(
        features_ptr + neighbor_idx_safe[:, None] * C_in + off_cin[None, :],
        mask=valid_mask[:, None] & mask_cin[None, :],
        other=0.0
    )
    
    do_tile = tl.load(
        d_out_ptr + off_n[:, None] * C_out + off_cout[None, :],
        mask=valid_mask[:, None] & mask_cout[None, :],
        other=0.0
    )
    
    acc = tl.dot(tl.trans(f_tile), do_tile)
    
    w_offset = (pid_k * C_in * C_out) + (off_cin[:, None] * C_out + off_cout[None, :])
    tl.atomic_add(d_weights_ptr + w_offset, acc.to(d_weights_ptr.dtype.element_ty), mask=mask_cin[:, None] & mask_cout[None, :])


class ConvPrecomputedNeighborGEMM(torch.autograd.Function):
    """Sparse convolution using pre-computed neighbor indices."""
    
    @staticmethod
    def forward(ctx, features, weights, neighbor_indices):
        """
        Args:
            features: (N_in, C_in) input features
            weights: (K, C_in, C_out) convolution weights
            neighbor_indices: (N_out, K) precomputed neighbor indices (-1 for invalid)
        """
        N_in, C_in = features.shape
        K, _, C_out = weights.shape
        N_out = neighbor_indices.shape[0]
        
        features = features.contiguous()
        weights = weights.contiguous()
        neighbor_indices = neighbor_indices.contiguous()
        
        out_features = torch.zeros((N_out, C_out), device=features.device, dtype=features.dtype)
        
        grid = lambda META: (
            triton.cdiv(N_out, META['BLOCK_SIZE_N']),
            triton.cdiv(C_out, META['BLOCK_SIZE_C_OUT'])
        )
        
        precomputed_fwd_kernel[grid](
            features_ptr=features,
            weights_ptr=weights,
            out_ptr=out_features,
            neighbor_indices_ptr=neighbor_indices,
            N_out=N_out,
            C_in=C_in,
            C_out=C_out,
            K_VOL=K,
        )
        
        ctx.save_for_backward(features, weights, neighbor_indices)
        ctx.K = K
        ctx.C_in = C_in
        ctx.C_out = C_out
        ctx.N_out = N_out
        
        return out_features
    
    @staticmethod
    def backward(ctx, grad_output):
        features, weights, neighbor_indices = ctx.saved_tensors
        K = ctx.K
        C_in = ctx.C_in
        C_out = ctx.C_out
        N_out = ctx.N_out
        
        weights = weights.contiguous()
        grad_output = grad_output.contiguous()
        
        # d_features
        grad_input = torch.zeros_like(features).contiguous()
        grid_feat = lambda META: (
            triton.cdiv(N_out, META['BLOCK_SIZE_N']),
            triton.cdiv(C_in, META['BLOCK_SIZE_C_IN'])
        )
        precomputed_bwd_feat_kernel[grid_feat](
            d_out_ptr=grad_output,
            weights_ptr=weights,
            d_features_ptr=grad_input,
            neighbor_indices_ptr=neighbor_indices,
            N_out=N_out,
            C_in=C_in,
            C_out=C_out,
            K_VOL=K,
        )
        
        # d_weights
        grad_weights = torch.zeros_like(weights).contiguous()
        grid_weight = lambda META: (
            triton.cdiv(N_out, META['BLOCK_SIZE_N']),
            K,
            triton.cdiv(C_in, META['BLOCK_SIZE_C_IN']) * triton.cdiv(C_out, META['BLOCK_SIZE_C_OUT'])
        )
        precomputed_bwd_weight_kernel[grid_weight](
            features_ptr=features,
            d_out_ptr=grad_output,
            d_weights_ptr=grad_weights,
            neighbor_indices_ptr=neighbor_indices,
            N_out=N_out,
            C_in=C_in,
            C_out=C_out,
            K_VOL=K,
        )
        
        return grad_input, grad_weights, None
