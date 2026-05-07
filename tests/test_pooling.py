import pytest
import torch
from sparsetriton import SparseTensor
from sparsetriton.nn.functional import sparse_upsample
from sparsetriton.nn.modules import SparsePooling, SparseUpsample, SparseDownsample


def _cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return torch.device("cuda")


class TestSparseUpsampleFunctional:
    @pytest.mark.parametrize("scale_factor", [1, 2, 3])
    def test_forward(self, scale_factor):
        device = _cuda()
        coords = torch.tensor([[0, 0, 0, 0]], dtype=torch.int16, device=device)
        feats = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32, device=device)
        sp = SparseTensor(feats, coords, spatial_shape=(10, 10, 10), batch_size=1)

        out = sparse_upsample(sp, scale_factor=scale_factor)

        expected_count = scale_factor**3
        assert out.F.shape[0] == expected_count
        assert torch.allclose(out.F, feats.repeat(expected_count, 1))

        expected_coords = []
        for x in range(scale_factor):
            for y in range(scale_factor):
                for z in range(scale_factor):
                    expected_coords.append([0, x, y, z])
        expected_coords = torch.tensor(expected_coords, dtype=torch.int16, device=device)

        out_sorted = out.C[out.C[:, 0].argsort()]
        exp_sorted = expected_coords[expected_coords[:, 0].argsort()]
        assert torch.equal(out_sorted, exp_sorted)

        expected_shape = tuple(s * scale_factor for s in (10, 10, 10))
        assert tuple(out.spatial_shape) == expected_shape


class TestSparsePoolingModule:
    def test_init_max(self):
        pool = SparsePooling(kernel_size=2, mode="max", stride=2, padding=0)
        assert pool.kernel_size == 2
        assert pool.stride == 2

    def test_init_avg(self):
        pool = SparsePooling(kernel_size=2, mode="avg", stride=2)
        assert pool.mode == "avg"

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="mode must be"):
            SparsePooling(kernel_size=2, mode="invalid")

    def test_forward_cuda(self):
        device = _cuda()
        coords = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
                              dtype=torch.int16, device=device)
        feats = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32, device=device)
        sp = SparseTensor(feats, coords, spatial_shape=(2, 2, 2), batch_size=1)

        pool = SparsePooling(kernel_size=2, mode="avg", stride=2, padding=0)
        out = pool(sp)
        assert isinstance(out, SparseTensor)


class TestSparseUpsampleModule:
    def test_init(self):
        up = SparseUpsample(scale_factor=2)
        assert up.scale_factor == 2

    def test_forward_cuda(self):
        device = _cuda()
        coords = torch.tensor([[0, 0, 0, 0]], dtype=torch.int16, device=device)
        feats = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32, device=device)
        sp = SparseTensor(feats, coords, spatial_shape=(10, 10, 10), batch_size=1)

        up = SparseUpsample(scale_factor=2)
        out = up(sp)
        assert isinstance(out, SparseTensor)


class TestSparseDownsampleModule:
    def test_init_max(self):
        ds = SparseDownsample(scale_factor=2, mode="max")
        assert ds.kernel_size == 2
        assert ds.stride == 2

    def test_init_avg(self):
        ds = SparseDownsample(scale_factor=2, mode="avg")
        assert ds.mode == "avg"

    def test_forward_cuda(self):
        device = _cuda()
        coords = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 1]],
                              dtype=torch.int16, device=device)
        feats = torch.tensor([[1.0], [2.0]], dtype=torch.float32, device=device)
        sp = SparseTensor(feats, coords, spatial_shape=(4, 4, 4), batch_size=1)

        ds = SparseDownsample(scale_factor=2, mode="avg")
        out = ds(sp)
        assert isinstance(out, SparseTensor)
