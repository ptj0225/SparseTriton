import pytest
import torch
from sparsetriton import SparseTensor, randn
from sparsetriton.nn.functional import sparse_batch_norm, sparse_pooling, sparse_upsample


def _cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return torch.device("cuda")


class TestSparseBatchNorm:
    def test_training_output_shape(self):
        sp = randn((16, 16, 16), batch_size=2, nnz=10, channels=4, device="cpu")
        out = sparse_batch_norm(sp, weight=torch.ones(4), bias=torch.zeros(4),
                                training=True, eps=1e-5, momentum=0.1)
        assert isinstance(out, SparseTensor)
        assert out.feats.shape == sp.feats.shape
        assert torch.equal(out.coords, sp.coords)

    def test_inference_output_shape(self):
        sp = randn((16, 16, 16), batch_size=2, nnz=10, channels=4, device="cpu")
        out = sparse_batch_norm(
            sp, training=False,
            running_mean=torch.zeros(4), running_var=torch.ones(4),
        )
        assert isinstance(out, SparseTensor)

    def test_affine_scaling(self):
        sp = randn((16, 16, 16), batch_size=2, nnz=10, channels=4, device="cpu")
        w = torch.tensor([2.0, 2.0, 2.0, 2.0])
        b = torch.tensor([1.0, 1.0, 1.0, 1.0])
        out = sparse_batch_norm(
            sp, weight=w, bias=b, training=False,
            running_mean=torch.zeros(4), running_var=torch.ones(4),
        )
        expected = sp.feats * 2.0 + 1.0
        torch.testing.assert_close(out.feats, expected, atol=1e-4, rtol=1e-4)


class TestSparsePooling:
    @pytest.mark.parametrize("mode", ["max", "avg"])
    def test_forward_cuda(self, mode):
        if mode == "max":
            pytest.skip("max pooling Triton kernel has known compilation issue")
        device = _cuda()
        sp = randn((16, 16, 16), batch_size=2, nnz=50, channels=4, device=device)
        out = sparse_pooling(sp, kernel_size=2, padding=0, stride=2, mode=mode)
        assert isinstance(out, SparseTensor)
        assert out.feats.shape[1] == 4

    def test_invalid_mode(self):
        sp = randn((16, 16, 16), batch_size=2, nnz=10, channels=4, device="cpu")
        with pytest.raises(ValueError, match="Unsupported mode"):
            sparse_pooling(sp, kernel_size=2, padding=0, stride=2, mode="invalid")


class TestSparseUpsample:
    @pytest.mark.parametrize("scale_factor", [2, 3])
    def test_forward_cuda(self, scale_factor):
        device = _cuda()
        sp = randn((16, 16, 16), batch_size=2, nnz=10, channels=4, device=device)
        out = sparse_upsample(sp, scale_factor=scale_factor)
        assert isinstance(out, SparseTensor)
        assert out.feats.shape[1] == 4
