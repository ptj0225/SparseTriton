import torch
from torch.autograd import Function

# from torch.cuda.amp import custom_bwd, custom_fwd
from typing import Tuple

import triton
import triton.language as tl

from sparsetriton.config import get_coords_dtype

__all__ = ["to_dense"]


@triton.jit
def _to_dense_fwd_kernel(
    feats_ptr,
    coords_ptr, # (N, D)
    out_ptr,
    strides_ptr, # (3)
    batch_size,
    N,
    C,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
    D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    mask_n = offs_n < N
    mask_c = offs_c < C
    out_off_n = tl.zeros((BLOCK_N,), dtype=tl.int64)
    # Load features
    for d in range(D):
        stride = tl.load(strides_ptr + d)
        coords_d = tl.load(coords_ptr + offs_n * D + d, mask=mask_n, other=0).to(tl.int64)
        out_off_n += coords_d * stride

    feat_vals = tl.load(
        feats_ptr + offs_n[:, None] * C + offs_c[None, :],
        mask=mask_n[:, None] & mask_c[None, :],
        other=0.0,
    ) # (BLOCK_N, BLOCK_C)
    # Store to output
    tl.store(
        out_ptr + out_off_n[:, None] * C + offs_c[None, :],
        feat_vals,
        mask=mask_n[:, None] & mask_c[None, :],
    )


@triton.jit
def _to_dense_bwd_kernel(
    grad_out_ptr,
    coords_ptr,
    grad_feats_ptr,
    strides_ptr,
    N,
    C,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
    D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    mask_n = offs_n < N
    mask_c = offs_c < C

    # Calculate output offset
    out_off_n = tl.zeros((BLOCK_N,), dtype=tl.int64)
    for d in range(D):
        coord_val = tl.load(coords_ptr + offs_n * D + d, mask=mask_n, other=0).to(tl.int64)
        stride_val = tl.load(strides_ptr + d)
        out_off_n += coord_val * stride_val

    # Load grad_out
    grad_vals = tl.load(
        grad_out_ptr + out_off_n[:, None] * C + offs_c[None, :],
        mask=mask_n[:, None] & mask_c[None, :],
        other=0.0,
    )

    # Store grad_feats
    tl.store(
        grad_feats_ptr + offs_n[:, None] * C + offs_c[None, :],
        grad_vals,
        mask=mask_n[:, None] & mask_c[None, :],
    )


class ToDenseFunction(Function):
    @staticmethod
    # @custom_fwd(cast_inputs=torch.half)
    def forward(
        ctx,
        feats: torch.Tensor,
        coords: torch.Tensor,
        spatial_range: Tuple[int],
        batch_size: int,
    ) -> torch.Tensor:
        feats = feats.contiguous()
        coords = coords.contiguous().to(get_coords_dtype())
        
        output_shape = (batch_size,) + spatial_range + (feats.size(1),)
        outputs = torch.zeros(
            output_shape, dtype=feats.dtype, device=feats.device
        )

        N, C = feats.shape
        D = coords.shape[1]
        
        spatial_strides = torch.empty((D,), dtype=torch.int64, device=feats.device)
        spatial_strides[-1] = 1
        for d in range(0, D - 1):
            spatial_strides[-d - 2] = spatial_strides[-d - 1] * spatial_range[-d - 1]

        ctx.save_for_backward(coords, spatial_strides)
        ctx.N = N
        ctx.C = C
        ctx.D = D

        grid = lambda meta: (
            triton.cdiv(N, meta["BLOCK_N"]),
            triton.cdiv(C, meta["BLOCK_C"]),
        )

        _to_dense_fwd_kernel[grid](
            feats,
            coords,
            outputs,
            spatial_strides,
            batch_size,
            N,
            C,
            BLOCK_N=128,
            BLOCK_C=64,
            D=D,
        )
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        coords, spatial_strides = ctx.saved_tensors
        N, C, D = ctx.N, ctx.C, ctx.D

        grad_feats = torch.zeros((N, C), dtype=grad_output.dtype, device=grad_output.device)

        grid = lambda meta: (
            triton.cdiv(N, meta["BLOCK_N"]),
            triton.cdiv(C, meta["BLOCK_C"]),
        )

        _to_dense_bwd_kernel[grid](
            grad_output,
            coords,
            grad_feats,
            spatial_strides,
            N,
            C,
            BLOCK_N=128,
            BLOCK_C=64,
            D=D,
        )

        return grad_feats, None, None, None


def to_dense(
    feats: torch.Tensor, coords: torch.Tensor, spatial_range: Tuple[int]
) -> torch.Tensor:
    if coords.shape[0] == 0:
        batch_size = 1
    else:
        batch_size = int(coords[:, 0].max().item()) + 1
    return ToDenseFunction.apply(feats, coords, spatial_range, batch_size)
