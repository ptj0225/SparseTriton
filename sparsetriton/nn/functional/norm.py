"""Functional API for sparse normalization operations.

This module provides functional implementations of normalization
operations for sparse tensors.

Example:
    >>> import torch
    >>> from sparsetriton import SparseTensor, randn
    >>> from sparsetriton.nn.functional.norm import sparse_batch_norm
    >>> sp_tensor = randn((16, 16, 16), batch_size=2, nnz=10, channels=4, device="cpu")
    >>> running_mean = torch.zeros(4)
    >>> running_var = torch.ones(4)
    >>> output = sparse_batch_norm(sp_tensor, running_mean=running_mean, running_var=running_var, training=False)
"""

import torch
from sparsetriton import SparseTensor


def sparse_batch_norm(
    tensor: SparseTensor,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    running_mean: torch.Tensor | None = None,
    running_var: torch.Tensor | None = None,
    training: bool = True,
    eps: float = 1e-5,
    momentum: float = 0.1,
) -> SparseTensor:
    """Apply batch normalization to sparse tensor features.

    Normalizes features across the batch dimension separately for each
    channel. During training, computes batch statistics. During inference,
    uses running statistics.

    Args:
        tensor: Input sparse tensor
        weight: Optional affine scale parameter of shape (num_channels,)
        bias: Optional affine shift parameter of shape (num_channels,)
        running_mean: Optional running mean statistics for inference
        running_var: Optional running variance statistics for inference
        training: Whether to compute batch statistics (True) or use running stats (False)
        eps: Small value for numerical stability
        momentum: Momentum for updating running statistics

    Returns:
        SparseTensor: Normalized sparse tensor with same coordinates

    Example:
        >>> import torch
        >>> from sparsetriton import randn
        >>> from sparsetriton.nn.functional.norm import sparse_batch_norm
        >>> sp_tensor = randn((16, 16, 16), batch_size=2, nnz=10, channels=4, device="cpu")
        >>> running_mean = torch.zeros(4)
        >>> running_var = torch.ones(4)
        >>> output = sparse_batch_norm(
        ...     sp_tensor,
        ...     running_mean=running_mean,
        ...     running_var=running_var,
        ...     training=False
        ... )
        >>> output.feats.shape == sp_tensor.feats.shape
        True
        >>> torch.allclose(output.coords, sp_tensor.coords)
        tensor(True)
    """
    feats = tensor.F
    coords = tensor.C
    batch_size = tensor.batch_size
    num_channels = feats.shape[-1]
    device = feats.device

    batch_idx = coords[:, 0].long()

    if training:
        # Compute mean per batch and channel
        sum_feats = torch.zeros(batch_size, num_channels, device=device)
        sum_feats.index_add_(0, batch_idx, feats)

        # Count points in each batch
        counts = torch.bincount(batch_idx, minlength=batch_size).to(feats.dtype).view(-1, 1).clamp(min=1)
        mean = sum_feats / counts  # (batch_size, num_channels)

        # Compute variance per batch and channel
        feats_centered = feats - mean[batch_idx]
        sum_sq_err = torch.zeros(batch_size, num_channels, device=device)
        sum_sq_err.index_add_(0, batch_idx, feats_centered.pow(2))
        var = sum_sq_err / counts  # (batch_size, num_channels)

        # Update running statistics for inference
        if running_mean is not None:
            running_mean.copy_((1 - momentum) * running_mean + momentum * mean.mean(0))
        if running_var is not None:
            running_var.copy_((1 - momentum) * running_var + momentum * var.mean(0))
    else:
        # Use running statistics during inference
        mean = running_mean
        var = running_var

    curr_mean = mean[batch_idx] if training else mean
    curr_var = var[batch_idx] if training else var

    # Normalize
    normalized_feats = (feats - curr_mean) / torch.sqrt(curr_var + eps)

    # Apply affine transformation
    if weight is not None:
        normalized_feats *= weight
    if bias is not None:
        normalized_feats += bias

    return tensor.replace(normalized_feats)
