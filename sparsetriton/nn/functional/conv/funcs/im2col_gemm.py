"""Im2Col + Dense GEMM sparse convolution.

This module provides sparse convolution using im2col transformation
followed by dense GEMM (cuBLAS) for maximum performance.
"""

import torch
from torch.autograd import Function


class Im2ColGEMM(Function):
    """Sparse convolution using im2col + dense GEMM.
    
    How it works:
    1. Gather features from neighbor indices → im2col matrix (N_out, K*C_in)
    2. Reshape weights to (K*C_in, C_out)
    3. Dense GEMM via torch.matmul (uses cuBLAS/Tensor Core)
    
    Pros:
    - Uses highly optimized cuBLAS GEMM
    - Tensor Core acceleration on modern GPUs
    - Similar to spconv's approach
    
    Cons:
    - Higher memory usage (im2col matrix)
    - Gather overhead
    """
    
    @staticmethod
    def forward(ctx, features, weight, neighbor_indices):
        """
        Args:
            features: (N_in, C_in) input features
            weight: (K, C_in, C_out) convolution weights
            neighbor_indices: (N_out, K) precomputed neighbor indices (-1 for invalid)
        
        Returns:
            output: (N_out, C_out) output features
        """
        N_in, C_in = features.shape
        K, _, C_out = weight.shape
        N_out = neighbor_indices.shape[0]
        device = features.device
        dtype = features.dtype
        
        # Step 1: Im2Col - gather features from neighbors
        # neighbor_indices: (N_out, K)
        # Output: im2col (N_out, K, C_in)
        
        # Expand indices for gathering: (N_out, K, C_in)
        neighbor_indices_expanded = neighbor_indices.unsqueeze(-1).expand(-1, -1, C_in)
        
        # Clamp invalid indices to 0 (will be zeroed later)
        valid_mask = (neighbor_indices >= 0)  # (N_out, K)
        neighbor_indices_clamped = neighbor_indices.clamp(min=0)
        
        # Gather: (N_out, K, C_in)
        # Use advanced indexing
        im2col = features[neighbor_indices_clamped.flatten()].reshape(N_out, K, C_in)
        
        # Zero out invalid entries
        im2col = im2col * valid_mask.unsqueeze(-1).to(dtype)
        
        # Step 2: Reshape for GEMM
        # im2col: (N_out, K*C_in)
        # weight: (K*C_in, C_out)
        im2col_flat = im2col.reshape(N_out, K * C_in)
        weight_flat = weight.reshape(K * C_in, C_out)
        
        # Step 3: Dense GEMM via torch.matmul (uses cuBLAS)
        output = torch.matmul(im2col_flat, weight_flat)
        
        # Save for backward
        ctx.save_for_backward(features, weight, neighbor_indices, valid_mask)
        ctx.K = K
        ctx.C_in = C_in
        ctx.C_out = C_out
        ctx.N_out = N_out
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Args:
            grad_output: (N_out, C_out) gradient of output
        
        Returns:
            grad_features: (N_in, C_in)
            grad_weight: (K, C_in, C_out)
            None (neighbor_indices doesn't need grad)
        """
        features, weight, neighbor_indices, valid_mask = ctx.saved_tensors
        K = ctx.K
        C_in = ctx.C_in
        C_out = ctx.C_out
        N_out = ctx.N_out
        
        # Reconstruct im2col for backward
        neighbor_indices_clamped = neighbor_indices.clamp(min=0)
        im2col = features[neighbor_indices_clamped.flatten()].reshape(N_out, K, C_in)
        im2col = im2col * valid_mask.unsqueeze(-1).to(features.dtype)
        im2col_flat = im2col.reshape(N_out, K * C_in)
        weight_flat = weight.reshape(K * C_in, C_out)
        
        # grad_weight = im2col^T @ grad_output
        grad_weight_flat = torch.matmul(im2col_flat.t(), grad_output)
        grad_weight = grad_weight_flat.reshape(K, C_in, C_out)
        
        # grad_im2col = grad_output @ weight^T
        grad_im2col_flat = torch.matmul(grad_output, weight_flat.t())
        grad_im2col = grad_im2col_flat.reshape(N_out, K, C_in)
        
        # Scatter grad back to features
        grad_features = torch.zeros_like(features)
        
        # Scatter-add gradients
        valid_grad = grad_im2col * valid_mask.unsqueeze(-1).to(features.dtype)
        
        # Use index_add for efficient scatter
        neighbor_indices_clamped_flat = neighbor_indices_clamped.flatten()
        valid_mask_flat = valid_mask.flatten()
        
        # Scatter each channel
        for c in range(C_in):
            grad_features[:, c].index_add_(
                0,
                neighbor_indices_clamped_flat,
                valid_grad[:, :, c].flatten() * valid_mask_flat.to(features.dtype)
            )
        
        return grad_features, grad_weight, None


def im2col_gemm_conv(features, weight, neighbor_indices):
    """Functional interface for im2col GEMM convolution."""
    return Im2ColGEMM.apply(features, weight, neighbor_indices)
