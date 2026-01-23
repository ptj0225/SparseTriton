import torch
from torch import nn
from sparsetriton import SparseTensor
from sparsetriton.nn.functional.norm import sparse_batch_norm

__all__ = ["SparseBatchNorm", "SparseLayerNorm"]

class SparseBatchNorm(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, tensor: SparseTensor) -> SparseTensor:
        feats = tensor.F
        coords = tensor.C
        batch_size = tensor.batch_size
        device = feats.device

        batch_idx = coords[:, 0].long()

        if self.training:
            sum_feats = torch.zeros(batch_size, self.num_features, device=device)
            sum_feats.index_add_(0, batch_idx, feats)
            
            counts = torch.bincount(batch_idx, minlength=batch_size).to(feats.dtype).view(-1, 1).clamp(min=1)
            
            batch_mean = sum_feats / counts
            
            feats_centered = feats - batch_mean[batch_idx]
            sum_sq_err = torch.zeros(batch_size, self.num_features, device=device)
            sum_sq_err.index_add_(0, batch_idx, feats_centered.pow(2))
            batch_var = sum_sq_err / counts

            active_batches = (counts > 1).any()
            if active_batches:
                with torch.no_grad():
                    self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean.mean(0)
                    self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var.mean(0)
            
            mu = batch_mean[batch_idx]
            var = batch_var[batch_idx]
        else:
            mu = self.running_mean
            var = self.running_var

        normalized_feats = (feats - mu) / torch.sqrt(var + self.eps)
        output_feats = normalized_feats * self.weight + self.bias

        return tensor.replace(output_feats)

class SparseLayerNorm(nn.LayerNorm):
    def forward(self, input: SparseTensor) -> SparseTensor:
        if isinstance(input, SparseTensor):
            return input.replace(super().forward(input.F))
        return super().forward(input)
