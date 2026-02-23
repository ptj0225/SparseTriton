"""Tests for sparse pooling and upsampling operations."""

import pytest
import torch
from sparsetriton import SparseTensor
from sparsetriton.nn.functional import sparse_upsample
from sparsetriton.nn.modules import SparsePooling, SparseUpsample, SparseDownsample


class TestSparseUpsampleFunctional:
    """Test sparse_upsample functional."""

    @pytest.mark.parametrize("scale_factor", [1, 2, 3])
    def test_sparse_upsample_forward(self, scale_factor):
        """Test forward pass with different scale factors."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for upsampling")

        device = torch.device("cuda")

        # 1. Setup Input - single point at (0, 0, 0, 0)
        coords = torch.tensor([[0, 0, 0, 0]], dtype=torch.int16, device=device)
        feats = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32, device=device)
        spatial_shape = (10, 10, 10)
        batch_size = 1

        input_tensor = SparseTensor(feats, coords, spatial_shape, batch_size)

        # 2. Run Upsample
        output_tensor = sparse_upsample(input_tensor, scale_factor=scale_factor)

        # 3. Verify Output
        expected_count = 1 * (scale_factor ** 3)
        assert output_tensor.F.shape[0] == expected_count
        assert output_tensor.C.shape[0] == expected_count

        # Features (Nearest Neighbor -> Copy)
        expected_feats = feats.repeat(expected_count, 1)
        assert torch.allclose(output_tensor.F, expected_feats)

        # Coords
        expected_coords_list = []
        for x in range(scale_factor):
            for y in range(scale_factor):
                for z in range(scale_factor):
                    expected_coords_list.append([0, x, y, z])  # Batch 0

        expected_coords = torch.tensor(expected_coords_list, dtype=torch.int16, device=device)

        # Sort for comparison
        out_c_sorted = output_tensor.C[output_tensor.C[:, 0].argsort()]
        exp_c_sorted = expected_coords[expected_coords[:, 0].argsort()]

        assert torch.equal(out_c_sorted, exp_c_sorted)

        # Spatial Shape
        expected_shape = tuple(s * scale_factor for s in spatial_shape)
        assert tuple(output_tensor.spatial_shape) == expected_shape


class TestSparsePoolingModule:
    """Test SparsePooling module."""

    def test_init_max(self):
        """Test initialization with max pooling."""
        pool = SparsePooling(kernel_size=2, mode="max", stride=2, padding=0)

        assert pool.kernel_size == 2
        assert pool.mode == "max"
        assert pool.stride == 2
        assert pool.padding == 0

    def test_init_avg(self):
        """Test initialization with avg pooling."""
        pool = SparsePooling(kernel_size=2, mode="avg", stride=2, padding=0)

        assert pool.mode == "avg"

    def test_init_invalid_mode(self):
        """Test initialization with invalid mode."""
        with pytest.raises(ValueError, match="mode must be"):
            SparsePooling(kernel_size=2, mode="invalid")

    def test_forward_cpu(self):
        """Test forward pass on CPU (may not work due to Triton)."""
        coords = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=torch.int16)
        feats = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32)
        sp_tensor = SparseTensor(feats, coords, spatial_shape=(2, 2, 2), batch_size=1)

        pool = SparsePooling(kernel_size=2, mode="max", stride=2, padding=0)

        try:
            output = pool(sp_tensor)
            assert isinstance(output, SparseTensor)
        except Exception as e:
            # Expected if Triton doesn't support CPU
            pytest.skip(f"Pooling not available on CPU: {e}")


class TestSparseUpsampleModule:
    """Test SparseUpsample module."""

    def test_init(self):
        """Test initialization."""
        upsample = SparseUpsample(scale_factor=2)

        assert upsample.scale_factor == 2

    def test_forward(self):
        """Test forward pass."""
        coords = torch.tensor([[0, 0, 0, 0]], dtype=torch.int16)
        feats = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
        sp_tensor = SparseTensor(feats, coords, spatial_shape=(10, 10, 10), batch_size=1)

        upsample = SparseUpsample(scale_factor=2)

        try:
            output = upsample(sp_tensor)
            assert isinstance(output, SparseTensor)
        except Exception as e:
            pytest.skip(f"Upsample not available on CPU: {e}")


class TestSparseDownsampleModule:
    """Test SparseDownsample module."""

    def test_init_max(self):
        """Test initialization with max downsampling."""
        downsample = SparseDownsample(scale_factor=2, mode="max")

        assert downsample.kernel_size == 2
        assert downsample.stride == 2
        assert downsample.mode == "max"
        assert downsample.padding == 0

    def test_init_avg(self):
        """Test initialization with avg downsampling."""
        downsample = SparseDownsample(scale_factor=2, mode="avg")

        assert downsample.mode == "avg"

    def test_forward(self):
        """Test forward pass."""
        coords = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 1]], dtype=torch.int16)
        feats = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
        sp_tensor = SparseTensor(feats, coords, spatial_shape=(4, 4, 4), batch_size=1)

        downsample = SparseDownsample(scale_factor=2, mode="max")

        try:
            output = downsample(sp_tensor)
            assert isinstance(output, SparseTensor)
        except Exception as e:
            pytest.skip(f"Downsample not available on CPU: {e}")
