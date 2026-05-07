import pytest
import torch
from sparsetriton import SparseTensor, randn
from sparsetriton.nn.modules import (
    SparseConv3D,
    SubMConv3D,
    SparseConvTransposed3D,
    SparseLinear,
    SparseBatchNorm,
    SparseLayerNorm,
    ReLU,
    LeakyReLU,
    SiLU,
    GELU,
    Sigmoid,
    Tanh,
    SparsePooling,
    SparseUpsample,
    SparseDownsample,
)


def _cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return torch.device("cuda")


class TestSparseConv3D:
    def test_init(self):
        conv = SparseConv3D(in_channels=4, out_channels=8, kernel_size=3)
        assert conv.in_channels == 4
        assert conv.out_channels == 8
        assert conv.kernel_size == (3, 3, 3)
        assert conv.stride == (1, 1, 1)
        assert conv.weight.shape == (27, 4, 8)
        assert conv.bias is not None

    def test_init_no_bias(self):
        conv = SparseConv3D(in_channels=4, out_channels=8, kernel_size=3, bias=False)
        assert conv.bias is None

    def test_init_custom_params(self):
        conv = SparseConv3D(in_channels=4, out_channels=8, kernel_size=3, stride=2, padding=1)
        assert conv.stride == (2, 2, 2)
        assert conv.padding == (1, 1, 1)

    def test_reset_parameters_changes_weight(self):
        conv = SparseConv3D(in_channels=4, out_channels=8, kernel_size=3)
        w_before = conv.weight.data.clone()
        conv.reset_parameters()
        assert not torch.allclose(conv.weight.data, w_before)

    def test_forward(self):
        device = _cuda()
        sp = randn((16, 16, 16), batch_size=2, nnz=10, channels=4, device=device)
        conv = SparseConv3D(in_channels=4, out_channels=8, kernel_size=3).to(device)
        out = conv(sp)
        assert isinstance(out, SparseTensor)
        assert out.feats.shape[1] == 8


class TestSubMConv3D:
    def test_init(self):
        conv = SubMConv3D(in_channels=4, out_channels=8, kernel_size=3)
        assert conv.subm is True

    def test_forward(self):
        device = _cuda()
        sp = randn((16, 16, 16), batch_size=2, nnz=10, channels=4, device=device)
        conv = SubMConv3D(in_channels=4, out_channels=8, kernel_size=3).to(device)
        out = conv(sp)
        assert isinstance(out, SparseTensor)
        assert torch.equal(out.coords, sp.coords)

    def test_default_padding(self):
        conv = SubMConv3D(in_channels=4, out_channels=8, kernel_size=5)
        assert conv.padding == (0, 0, 0)


class TestSparseConvTransposed3D:
    def test_init(self):
        conv = SparseConvTransposed3D(in_channels=4, out_channels=8, kernel_size=3)
        assert conv.transposed is True


class TestSparseLinear:
    def test_init(self):
        linear = SparseLinear(in_features=4, out_features=8)
        assert linear.weight.shape == (8, 4)
        assert linear.bias is not None

    def test_forward_sparse(self):
        sp = randn((16, 16, 16), batch_size=2, nnz=5, channels=4, device="cpu")
        linear = SparseLinear(in_features=4, out_features=8)
        out = linear(sp)
        assert isinstance(out, SparseTensor)
        assert out.feats.shape == (5, 8)
        assert torch.equal(out.coords, sp.coords)

    def test_forward_dense(self):
        dense = torch.randn(5, 4)
        linear = SparseLinear(in_features=4, out_features=8)
        assert linear(dense).shape == (5, 8)


class TestSparseBatchNorm:
    def test_init(self):
        bn = SparseBatchNorm(num_features=4)
        assert bn.num_features == 4

    def test_forward(self):
        sp = randn((16, 16, 16), batch_size=2, nnz=5, channels=4, device="cpu")
        bn = SparseBatchNorm(num_features=4)
        out = bn(sp)
        assert isinstance(out, SparseTensor)
        assert out.feats.shape == sp.feats.shape


class TestSparseLayerNorm:
    def test_init(self):
        ln = SparseLayerNorm(normalized_shape=4)
        assert ln.normalized_shape == (4,)

    def test_forward(self):
        sp = randn((16, 16, 16), batch_size=2, nnz=5, channels=4, device="cpu")
        ln = SparseLayerNorm(normalized_shape=4)
        out = ln(sp)
        assert isinstance(out, SparseTensor)
        assert out.feats.shape == sp.feats.shape


class TestActivations:
    @pytest.mark.parametrize("cls", [ReLU, LeakyReLU, SiLU, GELU, Sigmoid, Tanh])
    def test_forward(self, cls):
        sp = randn((16, 16, 16), batch_size=2, nnz=5, channels=4, device="cpu")
        act = cls(negative_slope=0.01) if cls is LeakyReLU else cls()
        out = act(sp)
        assert isinstance(out, SparseTensor)
        assert out.feats.shape == sp.feats.shape

    def test_relu_zeroes_negatives(self):
        feats = torch.tensor([[-1.0], [0.0], [1.0]])
        coords = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2]])
        out = ReLU()(SparseTensor(feats, coords))
        assert torch.all(out.F >= 0)
        assert out.F[0, 0].item() == 0.0
        assert out.F[2, 0].item() == 1.0

    def test_sigmoid_range(self):
        feats = torch.tensor([[-100.0], [0.0], [100.0]])
        coords = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2]])
        out = Sigmoid()(SparseTensor(feats, coords))
        assert torch.all((out.F >= 0) & (out.F <= 1))

    def test_tanh_range(self):
        feats = torch.tensor([[-100.0], [0.0], [100.0]])
        coords = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2]])
        out = Tanh()(SparseTensor(feats, coords))
        assert torch.all((out.F >= -1) & (out.F <= 1))


class TestSparsePooling:
    def test_init_max(self):
        pool = SparsePooling(kernel_size=2, mode="max")
        assert pool.mode == "max"

    def test_init_avg(self):
        pool = SparsePooling(kernel_size=2, mode="avg")
        assert pool.mode == "avg"

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="mode must be"):
            SparsePooling(kernel_size=2, mode="invalid")

    def test_forward_cuda(self):
        device = _cuda()
        sp = randn((16, 16, 16), batch_size=2, nnz=50, channels=4, device=device)
        pool = SparsePooling(kernel_size=2, mode="avg").to(device)
        out = pool(sp)
        assert isinstance(out, SparseTensor)


class TestSparseUpsample:
    def test_init(self):
        up = SparseUpsample(scale_factor=2)
        assert up.scale_factor == 2


class TestSparseDownsample:
    def test_init_max(self):
        ds = SparseDownsample(scale_factor=2, mode="max")
        assert ds.mode == "max"

    def test_init_avg(self):
        ds = SparseDownsample(scale_factor=2, mode="avg")
        assert ds.mode == "avg"
