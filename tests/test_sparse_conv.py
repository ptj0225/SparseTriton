"""Tests for sparse 3D convolution."""

import pytest
import torch
import torch.nn.functional as F
from sparsetriton import SparseTensor, randn
from sparsetriton.nn.functional import sparse_conv3d

# Disable TF32 for reproducibility
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def _require_cuda(skip_message: str) -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip(skip_message)
    return torch.device("cuda")


class TestSparseConv3DForward:
    """Test sparse 3D convolution forward pass."""

    @pytest.mark.parametrize(
        "C_in, C_out, kernel_size, stride, padding, dilation",
        [(8, 16, 3, 1, 1, 1), (4, 8, 3, 2, 1, 1), (16, 16, 5, 2, 2, 1), (16, 16, 5, 1, 2, 1)],
    )
    def test_sparse_conv3d_forward(self, C_in, C_out, kernel_size, stride, padding, dilation):
        """Test forward pass and compare with PyTorch dense conv."""
        device = _require_cuda("CUDA required for sparse convolution")

        # 1. Create input sparse tensor
        spatial_shape = (10, 10, 10)
        st_tensor = randn(spatial_shape, batch_size=1, channels=C_in, nnz=27, device=device).half()

        # 2. Create convolution weight (K, C_in, C_out)
        weight = torch.rand(kernel_size**3, C_in, C_out, device=device, dtype=torch.float16)

        # 3. Run sparsetriton convolution (submanifold=False)
        st_out_tensor = sparse_conv3d(
            st_tensor,
            weight,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            submanifold=False,
            transposed=False
        ).float()

        # 4. Run torch dense convolution
        # Weight: (K, C_in, C_out) -> (C_out, C_in, k, k, k)
        k = kernel_size
        weight_torch = weight.view(k, k, k, C_in, C_out).permute(4, 3, 0, 1, 2).contiguous().float()

        # Input: Sparse -> Dense (N, D, H, W, C) -> (N, C, D, H, W)
        dense_input = st_tensor.dense().permute(0, 4, 1, 2, 3).contiguous().float()

        dense_output = F.conv3d(
            dense_input,
            weight_torch,
            stride=stride,
            padding=padding,
            dilation=dilation
        )

        # 5. Compare dense results
        st_dense_output = st_out_tensor.dense()
        torch_dense_output = dense_output.permute(0, 2, 3, 4, 1).contiguous()

        torch.testing.assert_close(st_dense_output, torch_dense_output, atol=1e-3, rtol=1e-3)


class TestSparseConvTranspose3DForward:
    """Test sparse 3D transposed convolution forward pass."""

    @pytest.mark.parametrize(
        "C_in, C_out, kernel_size, stride, padding, dilation",
        [
            (16, 16, 3, 1, 1, 1),
            (32, 32, 3, 1, 1, 1),
            (32, 32, 5, 1, 2, 1),
            (4, 8, 3, 2, 1, 1),
            (8, 16, 3, 2, 1, 1),
            (8, 16, 3, 4, 1, 1),
        ],
    )
    def test_sparse_conv_transpose3d_forward(self, C_in, C_out, kernel_size, stride, padding, dilation):
        """Test transposed convolution forward pass."""
        device = _require_cuda("CUDA required for sparse transposed convolution")
        spatial_shape = (10, 10, 10)
        st_tensor = randn(spatial_shape, batch_size=1, channels=C_in, nnz=27, device=device).half()

        # Create convolution weight (K, C_in, C_out)
        weight = torch.rand(kernel_size**3, C_in, C_out, device=device, dtype=torch.float16) / (kernel_size**3)

        # Run sparsetriton transposed convolution
        st_out_tensor = sparse_conv3d(
            st_tensor,
            weight,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            submanifold=False,
            transposed=True
        ).float()

        # Run torch dense transposed convolution
        # Weight: (K, C_in, C_out) -> (C_in, C_out, k, k, k)
        k = kernel_size
        weight_torch = weight.view(k, k, k, C_in, C_out).permute(3, 4, 0, 1, 2).contiguous().float()

        # Input: Sparse -> Dense
        dense_input = st_tensor.dense().permute(0, 4, 1, 2, 3).contiguous().float()

        dense_output = F.conv_transpose3d(
            dense_input,
            weight_torch,
            stride=stride,
            padding=padding,
            dilation=dilation
        )

        # Compare
        st_dense_output = st_out_tensor.dense()
        torch_dense_output = dense_output.permute(0, 2, 3, 4, 1).contiguous()

        torch.testing.assert_close(st_dense_output, torch_dense_output, atol=1e-3, rtol=1e-3)


class TestSparseConv3DBackward:
    """Test sparse 3D convolution backward pass."""

    @pytest.mark.parametrize(
        "C_in, C_out, kernel_size, stride, padding, dilation",
        [
            (16, 16, 3, 1, 2, 1),
            (32, 32, 3, 1, 0, 1),
            (32, 32, 5, 1, 2, 1),
            (16, 16, 3, 2, 1, 1),
            (16, 16, 3, 2, 1, 2),
        ],
    )

    def test_sparse_conv3d_backward(self, C_in, C_out, kernel_size, stride, padding, dilation):
        """Test backward pass gradient computation."""
        device = _require_cuda("CUDA required for sparse convolution backward")
        spatial_shape = (64, 64, 64)

        # 1. Input preparation (float32 recommended for gradient checking)
        st_input = randn(spatial_shape, batch_size=1, channels=C_in, nnz=32**3, device=device)
        st_input.F = st_input.F.half()
        st_input.F.requires_grad_(True)

        # Weight
        weight = torch.randn(kernel_size**3, C_in, C_out, device=device, dtype=torch.float16) / (kernel_size**3)
        weight.requires_grad_(True)

        # 2. Run sparse convolution
        st_out = sparse_conv3d(
            st_input, weight, kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, submanifold=False
        )
        st_out_dense = st_out.dense()
        st_out_dense.float().abs().mean().backward()
        spconv_w_grad = weight.grad.clone()
        spconv_f_grad = st_input.F.grad.clone()

        # 3. Run dense convolution for comparison
        weight.grad = None
        st_input.F.grad = None

        k = kernel_size
        weight_torch = weight.view(k, k, k, C_in, C_out).permute(4, 3, 0, 1, 2).contiguous()
        dense_input = st_input.dense().permute(0, 4, 1, 2, 3).contiguous()
        dense_input.requires_grad_(True)

        dense_out = F.conv3d(
            dense_input, weight_torch, stride=stride, padding=padding, dilation=dilation
        )

        dense_out = dense_out.permute(0, 2, 3, 4, 1).contiguous()
        dense_out.float().abs().mean().backward()

        tconv_w_grad = weight.grad.clone()
        tconv_f_grad = st_input.F.grad.clone()

        # 4. Compare forward pass
        torch.testing.assert_close(st_out_dense, dense_out, atol=1e-3, rtol=1e-3)
        # 5. Compare gradients
        torch.testing.assert_close(tconv_w_grad, spconv_w_grad, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(spconv_f_grad, tconv_f_grad, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("spatial_shape, nnz", [((32, 32, 32), 4096), ((48, 48, 48), 8192)])
    def test_sparse_conv3d_backward_various_spatial_shapes(self, spatial_shape, nnz):
        """Run backward on multiple spatial shapes and verify gradients are finite."""
        device = _require_cuda("CUDA required for sparse convolution backward")
        C_in, C_out, kernel_size = 16, 16, 3
        st_input = randn(spatial_shape, batch_size=1, channels=C_in, nnz=nnz, device=device)
        st_input.F = st_input.F.half()
        st_input.F.requires_grad_(True)
        weight = torch.randn(kernel_size**3, C_in, C_out, device=device, dtype=torch.float16) / (kernel_size**3)
        weight.requires_grad_(True)

        st_out = sparse_conv3d(
            st_input, weight, kernel_size=kernel_size, stride=1, padding=1, dilation=1, submanifold=False
        )
        st_out.dense().float().abs().mean().backward()

        assert st_out.F.shape[1] == C_out
        assert torch.isfinite(weight.grad).all()
        assert torch.isfinite(st_input.F.grad).all()


class TestSubmanifoldConv3D:
    """Test submanifold convolution."""

    @staticmethod
    def _dense_ref_feats(st_input, st_out, weight, kernel_size, padding, dilation, C_in, C_out):
        weight_torch = weight.view(kernel_size, kernel_size, kernel_size, C_in, C_out)
        weight_torch = weight_torch.permute(4, 3, 0, 1, 2).contiguous().float()
        dense_input = st_input.dense().permute(0, 4, 1, 2, 3).contiguous().float()
        dense_output = F.conv3d(dense_input, weight_torch, stride=1, padding=padding, dilation=dilation)
        dense_output = dense_output.permute(0, 2, 3, 4, 1).contiguous()
        coords = st_out.coords.long()
        return dense_output[coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]]

    @pytest.mark.parametrize(
        "C_in, C_out, kernel_size, padding, dilation, nnz",
        [(8, 16, 3, 1, 1, 100), (8, 16, 5, 2, 1, 120)],
    )
    def test_submanifold_center_padding_matches_dense(
        self, C_in, C_out, kernel_size, padding, dilation, nnz
    ):
        """Center padding: coords preserved and feats match dense conv at active coords."""
        device = _require_cuda("CUDA required for sparse convolution")
        st_input = randn((10, 10, 10), batch_size=1, channels=C_in, nnz=nnz, device=device).half()
        weight = torch.rand(kernel_size**3, C_in, C_out, device=device, dtype=torch.float16)

        st_out = sparse_conv3d(
            st_input,
            weight,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            submanifold=True,
            transposed=False,
        )

        assert torch.equal(st_out.coords, st_input.coords)
        assert st_out.F.shape == (st_input.F.shape[0], C_out)

        ref_feats = self._dense_ref_feats(
            st_input, st_out, weight, kernel_size, padding, dilation, C_in, C_out
        )
        torch.testing.assert_close(st_out.F.float(), ref_feats, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize(
        "C_in, C_out, kernel_size, padding, dilation, nnz",
        [(8, 16, 3, 0, 1, 120)],
    )
    def test_submanifold_non_center_padding_runs(
        self, C_in, C_out, kernel_size, padding, dilation, nnz
    ):
        """Non-center padding: operation runs and returns finite features."""
        device = _require_cuda("CUDA required for sparse convolution")
        st_input = randn((10, 10, 10), batch_size=1, channels=C_in, nnz=nnz, device=device).half()
        weight = torch.rand(kernel_size**3, C_in, C_out, device=device, dtype=torch.float16)

        st_out = sparse_conv3d(
            st_input,
            weight,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            submanifold=True,
            transposed=False,
        )

        assert isinstance(st_out, SparseTensor)
        assert st_out.F.shape[1] == C_out
        assert st_out.coords.shape[1] == 4
        assert torch.isfinite(st_out.F).all()
