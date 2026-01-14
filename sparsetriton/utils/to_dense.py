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
    coords_ptr,
    out_ptr,
    strides_ptr,
    stride_feats_n,
    stride_feats_c,
    stride_coords_n,
    stride_coords_d,
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

    curr_offset = tl.zeros([BLOCK_N], dtype=tl.int64)

    for d in range(D):
        coords_d_ptr = coords_ptr + offs_n * stride_coords_n + d * stride_coords_d
        coord_val = tl.load(coords_d_ptr, mask=mask_n, other=0)
        stride_val = tl.load(strides_ptr + d)
        curr_offset += coord_val.to(tl.int64) * stride_val

    feats_ptrs = (
        feats_ptr + offs_n[:, None] * stride_feats_n + offs_c[None, :] * stride_feats_c
    )
    feat_vals = tl.load(feats_ptrs, mask=mask_n[:, None] & mask_c[None, :], other=0.0)

    out_ptrs = out_ptr + curr_offset[:, None] + offs_c[None, :]
    tl.store(out_ptrs, feat_vals, mask=mask_n[:, None] & mask_c[None, :])


@triton.jit
def _to_dense_bwd_kernel(
    grad_out_ptr,
    coords_ptr,
    grad_feats_ptr,
    strides_ptr,
    stride_coords_n,
    stride_coords_d,
    stride_grad_feats_n,
    stride_grad_feats_c,
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

    curr_offset = tl.zeros([BLOCK_N], dtype=tl.int64)

    for d in range(D):
        coords_d_ptr = coords_ptr + offs_n * stride_coords_n + d * stride_coords_d
        coord_val = tl.load(coords_d_ptr, mask=mask_n, other=0)
        stride_val = tl.load(strides_ptr + d)
        curr_offset += coord_val.to(tl.int64) * stride_val

    grad_out_ptrs = grad_out_ptr + curr_offset[:, None] + offs_c[None, :]
    grad_vals = tl.load(
        grad_out_ptrs, mask=mask_n[:, None] & mask_c[None, :], other=0.0
    )

    grad_feats_ptrs = (
        grad_feats_ptr
        + offs_n[:, None] * stride_grad_feats_n
        + offs_c[None, :] * stride_grad_feats_c
    )
    tl.store(grad_feats_ptrs, grad_vals, mask=mask_n[:, None] & mask_c[None, :])


class ToDenseFunction(Function):
    @staticmethod
    # @custom_fwd(cast_inputs=torch.half)
    def forward(
        ctx,
        feats: torch.Tensor,
        coords: torch.Tensor,
        spatial_range: Tuple[int],
    ) -> torch.Tensor:
        feats = feats.contiguous()
        coords = coords.contiguous().to(get_coords_dtype())
        outputs = torch.zeros(
            spatial_range + (feats.size(1),), dtype=feats.dtype, device=feats.device
        )

        N, C = feats.shape
        D = len(spatial_range)
        spatial_strides = torch.tensor(
            outputs.stride()[:-1], dtype=torch.int64, device=feats.device
        )

        grid = lambda meta: (
            triton.cdiv(N, meta["BLOCK_N"]),
            triton.cdiv(C, meta["BLOCK_C"]),
        )
        _to_dense_fwd_kernel[grid](
            feats,
            coords,
            outputs,
            spatial_strides,
            feats.stride(0),
            feats.stride(1),
            coords.stride(0),
            coords.stride(1),
            N,
            C,
            BLOCK_N=128,
            BLOCK_C=64,
            D=D,
        )

        ctx.for_backwards = (coords, spatial_range)
        return outputs.to(feats.dtype)

    @staticmethod
    # @custom_bwd
    def backward(ctx, grad_output: torch.Tensor):
        coords, spatial_range = ctx.for_backwards
        grad_output = grad_output.contiguous()
        grad_feats = torch.zeros(
            coords.size(0),
            grad_output.size(-1),
            dtype=grad_output.dtype,
            device=grad_output.device,
        )

        N, C = grad_feats.shape
        D = len(spatial_range)
        spatial_strides = torch.tensor(
            grad_output.stride()[:-1], dtype=torch.int64, device=grad_output.device
        )

        grid = lambda meta: (
            triton.cdiv(N, meta["BLOCK_N"]),
            triton.cdiv(C, meta["BLOCK_C"]),
        )
        _to_dense_bwd_kernel[grid](
            grad_output,
            coords,
            grad_feats,
            spatial_strides,
            coords.stride(0),
            coords.stride(1),
            grad_feats.stride(0),
            grad_feats.stride(1),
            N,
            C,
            BLOCK_N=128,
            BLOCK_C=64,
            D=D,
        )

        return grad_feats, None, None


def to_dense(
    feats: torch.Tensor, coords: torch.Tensor, spatial_range: Tuple[int]
) -> torch.Tensor:
    return ToDenseFunction.apply(feats, coords, spatial_range)
