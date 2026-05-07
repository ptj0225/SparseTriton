import pytest
import torch
from sparsetriton import SparseTensor, randn
from sparsetriton.config import set_coords_dtype, get_coords_dtype


class TestSparseTensorInit:
    def _make(self, feats=None, coords=None, **kw):
        if feats is None:
            feats = torch.randn(3, 4)
        if coords is None:
            coords = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 3, 3]])
        return SparseTensor(feats, coords, **kw)

    def test_basic(self):
        sp = self._make(feats=torch.randn(5, 3),
                        coords=torch.tensor([[0,1,2,3],[0,1,2,4],[0,1,3,3],[1,0,0,0],[1,1,1,1]]))
        assert sp.feats.shape == (5, 3)
        assert sp.batch_size == 2

    def test_explicit_spatial_shape(self):
        sp = self._make(spatial_shape=(10, 10, 10))
        assert sp.spatial_shape == torch.Size([10, 10, 10])

    def test_explicit_batch_size(self):
        sp = self._make(batch_size=5)
        assert sp.batch_size == 5

    def test_empty(self):
        sp = SparseTensor(torch.empty(0, 3), torch.empty(0, 4, dtype=torch.int16))
        assert sp.feats.shape == (0, 3)
        assert sp.batch_size == 0

    def test_shape_mismatch_raises(self):
        with pytest.raises(AssertionError, match="must match"):
            SparseTensor(torch.randn(5, 3), torch.tensor([[0, 1, 2, 3]]))

    def test_invalid_coords_ndim_raises(self):
        with pytest.raises(AssertionError, match="shape .*N, 4.*"):
            SparseTensor(torch.randn(5, 3), torch.randn(5, 3))

    def test_invalid_feats_ndim_raises(self):
        with pytest.raises(AssertionError, match="must be 2D"):
            SparseTensor(torch.randn(5, 3, 2), torch.randn(5, 4))

    def test_invalid_batch_size_raises(self):
        with pytest.raises(AssertionError, match="batch_size .* must be >"):
            SparseTensor(
                torch.randn(3, 4),
                torch.tensor([[0,1,2,3],[0,1,2,4],[5,1,3,3]]),
                batch_size=3,
            )


class TestSparseTensorProperties:
    def test_f_getter(self):
        sp = SparseTensor(torch.randn(3, 4), torch.tensor([[0,1,2,3],[0,1,2,4],[0,1,3,3]]))
        assert torch.equal(sp.F, sp.feats)

    def test_f_setter(self):
        sp = SparseTensor(torch.randn(3, 4), torch.tensor([[0,1,2,3],[0,1,2,4],[0,1,3,3]]))
        new_f = torch.randn(3, 5)
        sp.F = new_f
        assert torch.equal(sp.feats, new_f)

    def test_c_getter(self):
        sp = SparseTensor(torch.randn(3, 4), torch.tensor([[0,1,2,3],[0,1,2,4],[0,1,3,3]]))
        assert torch.equal(sp.C, sp.coords)

    def test_c_setter(self):
        sp = SparseTensor(torch.randn(3, 4), torch.tensor([[0,1,2,3],[0,1,2,4],[0,1,3,3]]))
        new_c = torch.tensor([[0,2,3,4],[0,2,3,5],[0,2,4,4]])
        sp.C = new_c
        assert torch.equal(sp.coords, new_c)


class TestSparseTensorMethods:
    def test_to_cpu(self):
        sp = SparseTensor(torch.randn(3, 4), torch.tensor([[0,1,2,3],[0,1,2,4],[0,1,3,3]]))
        assert sp.to("cpu").feats.device.type == "cpu"

    def test_cpu(self):
        sp = SparseTensor(torch.randn(3, 4), torch.tensor([[0,1,2,3],[0,1,2,4],[0,1,3,3]]))
        assert sp.cpu().feats.device.type == "cpu"

    def test_half(self):
        sp = SparseTensor(torch.randn(3, 4), torch.tensor([[0,1,2,3],[0,1,2,4],[0,1,3,3]]))
        assert sp.half().feats.dtype == torch.float16

    def test_float(self):
        sp = SparseTensor(torch.randn(3, 4, dtype=torch.float16),
                          torch.tensor([[0,1,2,3],[0,1,2,4],[0,1,3,3]]))
        assert sp.float().feats.dtype == torch.float32

    def test_replace_preserves_metadata(self):
        sp = SparseTensor(torch.randn(3, 4), torch.tensor([[0,1,2,3],[0,1,2,4],[0,1,3,3]]),
                          spatial_shape=(10, 10, 10), batch_size=2)
        new_f = torch.randn(3, 5)
        sp2 = sp.replace(new_f)
        assert torch.equal(sp2.feats, new_f)
        assert torch.equal(sp2.coords, sp.coords)
        assert sp2.spatial_shape == sp.spatial_shape
        assert sp2.batch_size == sp.batch_size

    def test_repr(self):
        sp = SparseTensor(torch.randn(3, 4), torch.tensor([[0,1,2,3],[0,1,2,4],[0,1,3,3]]))
        r = repr(sp)
        assert "SparseTensor" in r
        assert "spatial_shape=" in r


class TestRandn:
    def test_basic(self):
        sp = randn((16, 16, 16), batch_size=2, nnz=10, channels=4, device="cpu")
        assert sp.feats.shape == (10, 4)
        assert sp.batch_size == 2
        assert sp.spatial_shape == torch.Size([16, 16, 16])

    def test_unique_coords(self):
        sp = randn((16, 16, 16), batch_size=2, nnz=100, channels=4, device="cpu")
        assert torch.unique(sp.coords, dim=0).shape[0] == 100

    def test_dtype(self):
        sp = randn((16, 16, 16), batch_size=1, nnz=10, channels=4, device="cpu", dtype=torch.float64)
        assert sp.feats.dtype == torch.float64

    def test_invalid_nnz(self):
        with pytest.raises(ValueError, match="nnz .* exceeds"):
            randn((2, 2, 2), batch_size=1, nnz=10, channels=1, device="cpu")

    def test_single_voxel(self):
        sp = randn((2, 2, 2), batch_size=1, nnz=1, channels=4, device="cpu")
        assert sp.feats.shape == (1, 4)


class TestTensorCache:
    def test_init(self):
        from sparsetriton.tensor import TensorCache
        cache = TensorCache()
        assert len(cache.kmaps) == 0
        assert cache.hashtable is None

    def test_kmaps_storage(self):
        from sparsetriton.tensor import TensorCache
        cache = TensorCache()
        cache.kmaps[(3, 3, 1)] = "test"
        assert cache.kmaps[(3, 3, 1)] == "test"
