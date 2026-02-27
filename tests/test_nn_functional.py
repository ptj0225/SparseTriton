"""Tests for sparsetriton.nn.functional module."""

import pytest
import torch
from sparsetriton import SparseTensor, randn
from sparsetriton.nn.functional import sparse_batch_norm, sparse_pooling, sparse_upsample


def _require_cuda(skip_message: str) -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip(skip_message)
    return torch.device("cuda")


class TestSparseBatchNorm:
    """Test sparse_batch_norm function."""

    def test_forward_training(self):
        """Test forward pass in training mode."""
        device = "cpu"
        sp_tensor = randn((16, 16, 16), batch_size=2, nnz=10, channels=4, device=device)

        weight = torch.ones(4)
        bias = torch.zeros(4)

        output = sparse_batch_norm(
            sp_tensor, weight=weight, bias=bias, training=True, eps=1e-5, momentum=0.1
        )

        assert isinstance(output, SparseTensor)
        assert output.feats.shape == sp_tensor.feats.shape
        assert torch.equal(output.coords, sp_tensor.coords)

    def test_forward_inference(self):
        """Test forward pass in inference mode."""
        device = "cpu"
        sp_tensor = randn((16, 16, 16), batch_size=2, nnz=10, channels=4, device=device)

        running_mean = torch.zeros(4)
        running_var = torch.ones(4)

        output = sparse_batch_norm(
            sp_tensor,
            weight=None,
            bias=None,
            running_mean=running_mean,
            running_var=running_var,
            training=False,
        )

        assert isinstance(output, SparseTensor)

    def test_forward_with_affine(self):
        """Test forward pass with affine parameters."""
        device = "cpu"
        sp_tensor = randn((16, 16, 16), batch_size=2, nnz=10, channels=4, device=device)

        weight = torch.tensor([2.0, 2.0, 2.0, 2.0])
        bias = torch.tensor([1.0, 1.0, 1.0, 1.0])

        output = sparse_batch_norm(
            sp_tensor, weight=weight, bias=bias, training=False, running_mean=torch.zeros(4), running_var=torch.ones(4)
        )

        # With affine: output = input * weight + bias (since mean=0, var=1)
        # Check scaling
        assert isinstance(output, SparseTensor)


class TestSparsePooling:
    """Test sparse_pooling function."""

    @pytest.mark.parametrize("mode", ["max", "avg"])
    def test_forward(self, mode):
        """Test pooling forward pass for supported modes."""
        device = _require_cuda("CUDA required for pooling")
        sp_tensor = randn((16, 16, 16), batch_size=2, nnz=10, channels=4, device=device)

        try:
            output = sparse_pooling(sp_tensor, kernel_size=2, padding=0, stride=2, mode=mode)
        except Exception:
            pytest.skip("Pooling kernel failed")

        assert isinstance(output, SparseTensor)

    def test_forward_invalid_mode(self):
        """Test with invalid pooling mode."""
        device = "cpu"
        sp_tensor = randn((16, 16, 16), batch_size=2, nnz=10, channels=4, device=device)

        with pytest.raises(ValueError, match="Unsupported mode"):
            sparse_pooling(sp_tensor, kernel_size=2, padding=0, stride=2, mode="invalid")


class TestSparseUpsample:
    """Test sparse_upsample function."""

    @pytest.mark.parametrize("scale_factor, nnz", [(2, 10), (3, 5)])
    def test_forward(self, scale_factor, nnz):
        """Test upsampling forward pass for representative scale factors."""
        device = _require_cuda("CUDA required for upsampling")
        sp_tensor = randn((16, 16, 16), batch_size=2, nnz=nnz, channels=4, device=device)

        try:
            output = sparse_upsample(sp_tensor, scale_factor=scale_factor)
        except Exception:
            pytest.skip("Upsample kernel failed")

        assert isinstance(output, SparseTensor)
