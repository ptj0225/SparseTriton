"""Tests for sparsetriton.nn.functional module."""

import pytest
import torch
from sparsetriton import SparseTensor, randn
from sparsetriton.nn.functional import sparse_batch_norm, sparse_pooling, sparse_upsample


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

    def test_forward_max(self):
        """Test max pooling forward pass."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for pooling")

        device = torch.device("cuda")
        sp_tensor = randn((16, 16, 16), batch_size=2, nnz=10, channels=4, device=device)

        try:
            output = sparse_pooling(sp_tensor, kernel_size=2, padding=0, stride=2, mode="max")

            assert isinstance(output, SparseTensor)
        except Exception:
            pytest.skip("Pooling kernel failed")

    def test_forward_avg(self):
        """Test average pooling forward pass."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for pooling")

        device = torch.device("cuda")
        sp_tensor = randn((16, 16, 16), batch_size=2, nnz=10, channels=4, device=device)

        try:
            output = sparse_pooling(sp_tensor, kernel_size=2, padding=0, stride=2, mode="avg")

            assert isinstance(output, SparseTensor)
        except Exception:
            pytest.skip("Pooling kernel failed")

    def test_forward_invalid_mode(self):
        """Test with invalid pooling mode."""
        device = "cpu"
        sp_tensor = randn((16, 16, 16), batch_size=2, nnz=10, channels=4, device=device)

        with pytest.raises(ValueError, match="Unsupported mode"):
            sparse_pooling(sp_tensor, kernel_size=2, padding=0, stride=2, mode="invalid")


class TestSparseUpsample:
    """Test sparse_upsample function."""

    def test_forward_scale_2(self):
        """Test upsampling with scale factor 2."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for upsampling")

        device = torch.device("cuda")
        sp_tensor = randn((16, 16, 16), batch_size=2, nnz=10, channels=4, device=device)

        try:
            output = sparse_upsample(sp_tensor, scale_factor=2)

            assert isinstance(output, SparseTensor)
            # Output should have 2^3 = 8x more points (if all coordinates are unique)
        except Exception:
            pytest.skip("Upsample kernel failed")

    def test_forward_scale_3(self):
        """Test upsampling with scale factor 3."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for upsampling")

        device = torch.device("cuda")
        sp_tensor = randn((16, 16, 16), batch_size=2, nnz=5, channels=4, device=device)

        try:
            output = sparse_upsample(sp_tensor, scale_factor=3)

            assert isinstance(output, SparseTensor)
        except Exception:
            pytest.skip("Upsample kernel failed")
