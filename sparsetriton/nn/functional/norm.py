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
    momentum: float = 0.1
) -> SparseTensor:
    feats = tensor.F
    coords = tensor.C
    batch_size = tensor.batch_size
    num_channels = feats.shape[-1]
    device = feats.device

    batch_idx = coords[:, 0].long()

    if training:
        sum_feats = torch.zeros(batch_size, num_channels, device=device)
        sum_feats.index_add_(0, batch_idx, feats)
        
        # 각 배치에 속한 점의 개수
        counts = torch.bincount(batch_idx, minlength=batch_size).to(feats.dtype).view(-1, 1).clamp(min=1)
        mean = sum_feats / counts  # (batch_size, num_channels)

        # 3. 배치별 분산 계산
        feats_centered = feats - mean[batch_idx]
        sum_sq_err = torch.zeros(batch_size, num_channels, device=device)
        sum_sq_err.index_add_(0, batch_idx, feats_centered.pow(2))
        var = sum_sq_err / counts  # (batch_size, num_channels)

        # 4. Running Statistics 업데이트 (Inference용)
        if running_mean is not None:
            running_mean.copy_((1 - momentum) * running_mean + momentum * mean.mean(0))
        if running_var is not None:
            running_var.copy_((1 - momentum) * running_var + momentum * var.mean(0))
    else:
        mean = running_mean
        var = running_var

    curr_mean = mean[batch_idx] if training else mean
    curr_var = var[batch_idx] if training else var
    
    normalized_feats = (feats - curr_mean) / torch.sqrt(curr_var + eps)

    if weight is not None:
        normalized_feats *= weight
    if bias is not None:
        normalized_feats += bias

    return tensor.replace(normalized_feats)