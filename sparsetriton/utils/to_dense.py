"""Conversion utilities between sparse and dense tensors.

This module provides functions for converting SparseTensor to dense
tensor format using Triton kernels for GPU acceleration.

Example:
    >>> import torch
    >>> from sparsetriton import SparseTensor
    >>> from sparsetriton.utils.to_dense import to_dense
    >>> feats = torch.tensor([[1.0], [2.0], [3.0]])
    >>> coords = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]])
    >>> sp = SparseTensor(feats, coords)
    >>> dense = to_dense(sp.feats, sp.coords, sp.spatial_shape)
    >>> dense.shape
    torch.Size([2, 1, 1, 1, 2])
"""

import torch
from torch.autograd import Function
from typing import Tuple

import triton
import triton.language as tl

from sparsetriton.config import get_coords_dtype

__all__ = ["to_dense"]


@triton.jit
def _to_dense_fwd_kernel(
    feats_ptr,
    coords_ptr,  # (N, D)
    out_ptr,
    strides_ptr,  # (3)
    batch_size,
    N,
    C,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
    D: tl.constexpr,
):
    """Forward kernel for sparse to dense conversion.

    Scatters sparse features into a dense tensor using coordinate indexing.

    Args:
        feats_ptr: Pointer to feature data (N, C)
        coords_ptr: Pointer to coordinate data (N, D)
        out_ptr: Pointer to output dense tensor
        strides_ptr: Pointer to spatial stride array (D,)
        batch_size: Number of batches
        N: Number of sparse points
        C: Number of channels
        BLOCK_N: Block size for N dimension
        BLOCK_C: Block size for C dimension
        D: Number of coordinate dimensions (4 for batch, x, y, z)
    """
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
    )  # (BLOCK_N, BLOCK_C)
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
    """Backward kernel for sparse to dense conversion.

    Gathers gradients from dense tensor back to sparse features.

    Args:
        grad_out_ptr: Pointer to gradient of output dense tensor
        coords_ptr: Pointer to coordinate data (N, D)
        grad_feats_ptr: Pointer to output gradient for features
        strides_ptr: Pointer to spatial stride array (D,)
        N: Number of sparse points
        C: Number of channels
        BLOCK_N: Block size for N dimension
        BLOCK_C: Block size for C dimension
        D: Number of coordinate dimensions
    """
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    mask_n = offs_n < N
    mask_c = offs_c < C
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
    """Autograd function for sparse to dense conversion."""

    @staticmethod
    def forward(
        ctx,
        feats: torch.Tensor,
        coords: torch.Tensor,
        spatial_range: Tuple[int],
        batch_size: int,
    ) -> torch.Tensor:
        """Forward pass: convert sparse to dense.

        Args:
            ctx: Autograd context
            feats: Sparse feature tensor of shape (N, C)
            coords: Sparse coordinate tensor of shape (N, D)
            spatial_range: Spatial dimensions (D, H, W)
            batch_size: Number of batches

        Returns:
            torch.Tensor: Dense tensor of shape (B, C, D, H, W)
        """
        feats = feats.contiguous()
        coords = coords.contiguous().to(get_coords_dtype())

        output_shape = (batch_size,) + spatial_range + (feats.size(1),)
        outputs = torch.zeros(output_shape, dtype=feats.dtype, device=feats.device)

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
        return outputs.contiguous()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None, None]:
        """Backward pass: compute gradients.

        Args:
            ctx: Autograd context
            grad_output: Gradient of output dense tensor

        Returns:
            Tuple: (grad_feats, None, None, None) - gradient for features, others None
        """
        grad_output = grad_output.contiguous()

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
    """Convert sparse tensor to dense tensor.

    Scatters sparse features into a dense 5D tensor (B, C, D, H, W).

    Args:
        feats: Sparse feature tensor of shape (N, C) where N is number of active points
        coords: Sparse coordinate tensor of shape (N, 4) in (batch, x, y, z) format
        spatial_range: Spatial dimensions as a tuple (D, H, W)

    Returns:
        torch.Tensor: Dense tensor of shape (B, C, D, H, W)

    Raises:
        AssertionError: If shapes are incompatible

    Example:
        >>> import torch
        >>> from sparsetriton.utils.to_dense import to_dense
        >>> feats = torch.tensor([[1.0], [2.0], [3.0]])
        >>> coords = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]])
        >>> dense = to_dense(feats, coords, (1, 1, 2))
        >>> dense.shape
        torch.Size([2, 1, 1, 1, 2])
        >>> dense[0, 0, 0, 0, 0].item()
        1.0
        >>> dense[0, 0, 0, 0, 1].item()
        2.0
        >>> dense[1, 0, 0, 0, 0].item()
        3.0
    """
    if coords.shape[0] == 0:
        batch_size = 1
    else:
        batch_size = int(coords[:, 0].max().item()) + 1
    # Convert torch.Size to tuple if needed
    if isinstance(spatial_range, torch.Size):
        spatial_range = tuple(spatial_range)
    return ToDenseFunction.apply(feats, coords, spatial_range, batch_size)
