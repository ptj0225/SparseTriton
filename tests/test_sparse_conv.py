import pytest
import torch
import torch.nn.functional as F
from sparsetriton import SparseTensor, randn
from sparsetriton.nn.functional import sparse_conv3d

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def _cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return torch.device("cuda")


class TestSparseConv3DForward:
    @pytest.mark.parametrize(
        "C_in, C_out, kernel_size, stride, padding, dilation",
        [
            (8, 16, 3, 1, 1, 1),
            (4, 8, 3, 2, 1, 1),
            (16, 16, 5, 2, 2, 1),
            (16, 16, 5, 1, 2, 1),
        ],
    )
    def test_matches_dense(self, C_in, C_out, kernel_size, stride, padding, dilation):
        device = _cuda()
        sp = randn((10, 10, 10), batch_size=1, channels=C_in, nnz=27, device=device).half()
        weight = torch.rand(kernel_size**3, C_in, C_out, device=device, dtype=torch.float16)

        st_out = sparse_conv3d(
            sp, weight, kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, submanifold=False,
        ).float()

        k = kernel_size
        w_torch = weight.view(k, k, k, C_in, C_out).permute(4, 3, 0, 1, 2).contiguous().float()
        dense_in = sp.dense().permute(0, 4, 1, 2, 3).contiguous().float()
        dense_out = F.conv3d(dense_in, w_torch, stride=stride, padding=padding, dilation=dilation)

        st_dense = st_out.dense()
        torch_dense = dense_out.permute(0, 2, 3, 4, 1).contiguous()
        torch.testing.assert_close(st_dense, torch_dense, atol=1e-3, rtol=1e-3)


class TestSparseConvTranspose3DForward:
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
    def test_matches_dense(self, C_in, C_out, kernel_size, stride, padding, dilation):
        device = _cuda()
        sp = randn((10, 10, 10), batch_size=1, channels=C_in, nnz=27, device=device).half()
        weight = torch.rand(kernel_size**3, C_in, C_out, device=device, dtype=torch.float16) / kernel_size**3

        st_out = sparse_conv3d(
            sp, weight, kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, submanifold=False, transposed=True,
        ).float()

        k = kernel_size
        w_torch = weight.view(k, k, k, C_in, C_out).permute(3, 4, 0, 1, 2).contiguous().float()
        dense_in = sp.dense().permute(0, 4, 1, 2, 3).contiguous().float()
        dense_out = F.conv_transpose3d(dense_in, w_torch, stride=stride, padding=padding, dilation=dilation)

        st_dense = st_out.dense()
        torch_dense = dense_out.permute(0, 2, 3, 4, 1).contiguous()
        torch.testing.assert_close(st_dense, torch_dense, atol=1e-3, rtol=1e-3)


class TestSparseConv3DBackward:
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
    def test_gradients_vs_dense(self, C_in, C_out, kernel_size, stride, padding, dilation):
        device = _cuda()

        sp_in = randn((64, 64, 64), batch_size=1, channels=C_in, nnz=32**3, device=device)
        feats = sp_in.F.half().requires_grad_(True)
        feats.retain_grad()
        sp_in.F = feats

        weight = torch.randn(kernel_size**3, C_in, C_out, device=device, dtype=torch.float16) / kernel_size**3
        weight.requires_grad_(True)

        st_out = sparse_conv3d(
            sp_in, weight, kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, submanifold=False,
        )
        st_out.dense().float().abs().mean().backward()
        sp_w_grad = weight.grad.clone()
        sp_f_grad = feats.grad.clone()

        k = kernel_size
        w_torch = weight.view(k, k, k, C_in, C_out).permute(4, 3, 0, 1, 2).contiguous()
        dense_in = sp_in.dense().permute(0, 4, 1, 2, 3).contiguous().detach().requires_grad_(True)

        dense_out = F.conv3d(dense_in, w_torch, stride=stride, padding=padding, dilation=dilation)
        dense_out = dense_out.permute(0, 2, 3, 4, 1).contiguous()
        dense_out.float().abs().mean().backward()

        coords = sp_in.coords.long()
        dense_f_grad = dense_in.grad.permute(0, 2, 3, 4, 1).contiguous()
        dense_f_grad_sparse = dense_f_grad[coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]]

        torch.testing.assert_close(weight.grad, sp_w_grad, atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(dense_f_grad_sparse, sp_f_grad, atol=5e-2, rtol=5e-2)

    @pytest.mark.parametrize(
        "spatial_shape, nnz",
        [((32, 32, 32), 4096), ((48, 48, 48), 8192)],
    )
    def test_gradients_finite(self, spatial_shape, nnz):
        device = _cuda()
        C_in, C_out, ks = 16, 16, 3

        sp_in = randn(spatial_shape, batch_size=1, channels=C_in, nnz=nnz, device=device)
        feats = sp_in.F.half().requires_grad_(True)
        feats.retain_grad()
        sp_in.F = feats
        weight = torch.randn(ks**3, C_in, C_out, device=device, dtype=torch.float16) / ks**3
        weight.requires_grad_(True)

        st_out = sparse_conv3d(
            sp_in, weight, kernel_size=ks, stride=1, padding=1, dilation=1, submanifold=False,
        )
        st_out.dense().float().abs().mean().backward()

        assert torch.isfinite(weight.grad).all()
        assert torch.isfinite(sp_in.F.grad).all()


class TestSubmanifoldConv3D:
    @staticmethod
    def _dense_ref(st_in, st_out, weight, ks, padding, dilation, C_in, C_out):
        w_torch = weight.view(ks, ks, ks, C_in, C_out).permute(4, 3, 0, 1, 2).contiguous().float()
        dense_in = st_in.dense().permute(0, 4, 1, 2, 3).contiguous().float()
        dense_out = F.conv3d(dense_in, w_torch, stride=1, padding=padding, dilation=dilation)
        dense_out = dense_out.permute(0, 2, 3, 4, 1).contiguous()
        c = st_out.coords.long()
        return dense_out[c[:, 0], c[:, 1], c[:, 2], c[:, 3]]

    @pytest.mark.parametrize(
        "C_in, C_out, ks, padding, dilation, nnz",
        [(8, 16, 3, 1, 1, 100), (8, 16, 5, 2, 1, 120)],
    )
    def test_center_padding_matches_dense(self, C_in, C_out, ks, padding, dilation, nnz):
        device = _cuda()
        sp_in = randn((10, 10, 10), batch_size=1, channels=C_in, nnz=nnz, device=device).half()
        weight = torch.rand(ks**3, C_in, C_out, device=device, dtype=torch.float16)

        st_out = sparse_conv3d(
            sp_in, weight, kernel_size=ks, stride=1, padding=padding,
            dilation=dilation, submanifold=True,
        )

        assert torch.equal(st_out.coords, sp_in.coords)
        assert st_out.F.shape == (sp_in.F.shape[0], C_out)

        ref = self._dense_ref(sp_in, st_out, weight, ks, padding, dilation, C_in, C_out)
        torch.testing.assert_close(st_out.F.float(), ref, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize(
        "C_in, C_out, ks, padding, dilation, nnz",
        [(8, 16, 3, 0, 1, 120)],
    )
    def test_non_center_padding_finite(self, C_in, C_out, ks, padding, dilation, nnz):
        device = _cuda()
        sp_in = randn((10, 10, 10), batch_size=1, channels=C_in, nnz=nnz, device=device).half()
        weight = torch.rand(ks**3, C_in, C_out, device=device, dtype=torch.float16)

        st_out = sparse_conv3d(
            sp_in, weight, kernel_size=ks, stride=1, padding=padding,
            dilation=dilation, submanifold=True,
        )

        assert isinstance(st_out, SparseTensor)
        assert st_out.F.shape[1] == C_out
        assert torch.isfinite(st_out.F).all()
