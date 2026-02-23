"""Tests for sparsetriton.tensor module."""

import pytest
import torch
from sparsetriton import SparseTensor, randn
from sparsetriton.config import set_coords_dtype, get_coords_dtype


class TestSparseTensorInit:
    """Test SparseTensor initialization."""

    def test_init_basic(self):
        """Test basic initialization."""
        feats = torch.randn(5, 3)
        coords = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 3, 3], [1, 0, 0, 0], [1, 1, 1, 1]])

        sp = SparseTensor(feats, coords)

        assert sp.feats.shape == (5, 3)
        assert sp.coords.shape == (5, 4)
        assert sp.batch_size == 2
        assert sp.spatial_shape == torch.Size([2, 4, 5])

    def test_init_with_spatial_shape(self):
        """Test initialization with explicit spatial shape."""
        feats = torch.randn(3, 4)
        coords = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 3, 3]])

        sp = SparseTensor(feats, coords, spatial_shape=(10, 10, 10))

        assert sp.spatial_shape == torch.Size([10, 10, 10])

    def test_init_with_batch_size(self):
        """Test initialization with explicit batch size."""
        feats = torch.randn(3, 4)
        coords = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 3, 3]])

        sp = SparseTensor(feats, coords, batch_size=5)

        assert sp.batch_size == 5

    def test_init_empty(self):
        """Test initialization with empty tensors."""
        feats = torch.empty(0, 3)
        coords = torch.empty(0, 4, dtype=torch.int16)

        sp = SparseTensor(feats, coords)

        assert sp.feats.shape == (0, 3)
        assert sp.coords.shape == (0, 4)
        assert sp.spatial_shape == torch.Size([0, 0, 0])
        assert sp.batch_size == 0

    def test_init_shape_mismatch(self):
        """Test initialization with shape mismatch."""
        feats = torch.randn(5, 3)
        coords = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 4]])

        with pytest.raises(AssertionError, match="must match"):
            SparseTensor(feats, coords)

    def test_init_invalid_coords_shape(self):
        """Test initialization with invalid coordinate shape."""
        feats = torch.randn(5, 3)
        coords = torch.randn(5, 3)  # Should be (N, 4)

        with pytest.raises(AssertionError, match="shape .*N, 4.*"):
            SparseTensor(feats, coords)

    def test_init_invalid_ndim(self):
        """Test initialization with invalid tensor dimensions."""
        feats = torch.randn(5, 3, 2)
        coords = torch.randn(5, 4)

        with pytest.raises(AssertionError, match="must be 2D"):
            SparseTensor(feats, coords)

    def test_init_invalid_batch_size(self):
        """Test initialization with invalid batch size."""
        feats = torch.randn(3, 4)
        coords = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 4], [5, 1, 3, 3]])  # Batch index 5

        with pytest.raises(AssertionError, match="batch_size .* must be >"):
            SparseTensor(feats, coords, batch_size=3)


class TestSparseTensorProperties:
    """Test SparseTensor properties."""

    def test_property_f(self):
        """Test F property."""
        feats = torch.randn(3, 4)
        coords = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 3, 3]])
        sp = SparseTensor(feats, coords)

        assert torch.equal(sp.F, sp.feats)

    def test_property_f_setter(self):
        """Test F property setter."""
        feats = torch.randn(3, 4)
        coords = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 3, 3]])
        sp = SparseTensor(feats, coords)

        new_feats = torch.randn(3, 5)
        sp.F = new_feats

        assert torch.equal(sp.feats, new_feats)
        assert sp.feats.shape == (3, 5)

    def test_property_c(self):
        """Test C property."""
        feats = torch.randn(3, 4)
        coords = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 3, 3]])
        sp = SparseTensor(feats, coords)

        assert torch.equal(sp.C, sp.coords)

    def test_property_c_setter(self):
        """Test C property setter."""
        feats = torch.randn(3, 4)
        coords = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 3, 3]])
        sp = SparseTensor(feats, coords)

        new_coords = torch.tensor([[0, 2, 3, 4], [0, 2, 3, 5], [0, 2, 4, 4]])
        sp.C = new_coords

        assert torch.equal(sp.coords, new_coords)


class TestSparseTensorMethods:
    """Test SparseTensor methods."""

    def test_to_cpu(self):
        """Test to() method with cpu."""
        device = torch.device("cpu")
        feats = torch.randn(3, 4)
        coords = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 3, 3]])
        sp = SparseTensor(feats, coords)

        sp_cpu = sp.to(device)

        assert sp_cpu.feats.device.type == "cpu"
        assert sp_cpu.coords.device.type == "cpu"

    def test_cpu_method(self):
        """Test cpu() method."""
        feats = torch.randn(3, 4)
        coords = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 3, 3]])
        sp = SparseTensor(feats, coords)

        sp_cpu = sp.cpu()

        assert sp_cpu.feats.device.type == "cpu"
        assert sp_cpu.coords.device.type == "cpu"

    def test_half(self):
        """Test half() method."""
        feats = torch.randn(3, 4, dtype=torch.float32)
        coords = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 3, 3]])
        sp = SparseTensor(feats, coords)

        sp_half = sp.half()

        assert sp_half.feats.dtype == torch.float16

    def test_float(self):
        """Test float() method."""
        feats = torch.randn(3, 4, dtype=torch.float16)
        coords = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 3, 3]])
        sp = SparseTensor(feats, coords)

        sp_float = sp.float()

        assert sp_float.feats.dtype == torch.float32

    def test_replace(self):
        """Test replace() method."""
        feats = torch.randn(3, 4)
        coords = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 3, 3]])
        sp = SparseTensor(feats, coords, spatial_shape=(10, 10, 10), batch_size=2)

        new_feats = torch.randn(3, 5)
        sp2 = sp.replace(new_feats)

        assert torch.equal(sp2.feats, new_feats)
        assert torch.equal(sp2.coords, sp.coords)
        assert sp2.spatial_shape == sp.spatial_shape
        assert sp2.batch_size == sp.batch_size

    def test_repr(self):
        """Test __repr__ method."""
        feats = torch.randn(3, 4)
        coords = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 3, 3]])
        sp = SparseTensor(feats, coords)

        repr_str = repr(sp)

        assert "SparseTensor" in repr_str
        assert "feats=" in repr_str
        assert "coords=" in repr_str
        assert "spatial_shape=" in repr_str


class TestRandn:
    """Test randn function."""

    def test_randn_basic(self):
        """Test basic random tensor generation."""
        sp = randn((16, 16, 16), batch_size=2, nnz=10, channels=4, device="cpu")

        assert sp.feats.shape == (10, 4)
        assert sp.coords.shape == (10, 4)
        assert sp.batch_size == 2
        assert sp.spatial_shape == torch.Size([16, 16, 16])

    def test_randn_unique_coords(self):
        """Test that generated coordinates are unique."""
        sp = randn((16, 16, 16), batch_size=2, nnz=100, channels=4, device="cpu")

        unique_coords = torch.unique(sp.coords, dim=0)
        assert unique_coords.shape[0] == 100

    def test_randn_dtype(self):
        """Test dtype parameter."""
        sp = randn((16, 16, 16), batch_size=1, nnz=10, channels=4, device="cpu", dtype=torch.float64)

        assert sp.feats.dtype == torch.float64

    def test_randn_invalid_nnz(self):
        """Test with invalid nnz (exceeds available voxels)."""
        with pytest.raises(ValueError, match="nnz .* exceeds total available voxels"):
            randn((2, 2, 2), batch_size=1, nnz=10, channels=1, device="cpu")

    def test_randn_single_voxel(self):
        """Test with single voxel."""
        sp = randn((2, 2, 2), batch_size=1, nnz=1, channels=4, device="cpu")

        assert sp.feats.shape == (1, 4)
        assert sp.coords.shape == (1, 4)


class TestTensorCache:
    """Test TensorCache class."""

    def test_cache_init(self):
        """Test TensorCache initialization."""
        from sparsetriton.tensor import TensorCache

        cache = TensorCache()

        assert len(cache.kmaps) == 0
        assert cache.hashtable is None

    def test_cache_kmaps(self):
        """Test kmaps storage."""
        from sparsetriton.tensor import TensorCache

        cache = TensorCache()
        cache.kmaps[(3, 3, 1)] = "test_value"

        assert cache.kmaps[(3, 3, 1)] == "test_value"
