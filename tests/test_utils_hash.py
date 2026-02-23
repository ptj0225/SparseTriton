"""Tests for sparsetriton.utils.hash module."""

import pytest
import torch
from sparsetriton.utils.hash import HashTable, flatten_coord, unflatten_coord, hash_coords


class TestFlattenCoord:
    """Test flatten_coord function."""

    def test_flatten_coord_basic(self):
        """Test basic coordinate flattening."""
        coords = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 3, 3]])
        flat = flatten_coord(coords)

        assert flat.shape == (3,)
        assert flat.dtype == torch.int64

    def test_flatten_coord_values(self):
        """Test that flattened values are correct."""
        # (b, x, y, z) -> b*2^48 + x*2^32 + y*2^16 + z
        coords = torch.tensor([[0, 1, 2, 3]])

        flat = flatten_coord(coords)[0].item()

        expected = (0 << 48) | (1 << 32) | (2 << 16) | 3
        assert flat == expected

    def test_flatten_coord_large_values(self):
        """Test with larger coordinate values."""
        coords = torch.tensor([[1, 255, 255, 255]])
        flat = flatten_coord(coords)[0].item()

        # Should handle 16-bit values
        expected = (1 << 48) | (255 << 32) | (255 << 16) | 255
        assert flat == expected


class TestUnflattenCoord:
    """Test unflatten_coord function."""

    def test_unflatten_coord_basic(self):
        """Test basic coordinate unflattening."""
        flat = torch.tensor([((0 << 48) | (1 << 32) | (2 << 16) | 3)])
        coords = unflatten_coord(flat)

        assert coords.shape == (1, 4)
        assert coords[0, 0].item() == 0  # batch
        assert coords[0, 1].item() == 1  # x
        assert coords[0, 2].item() == 2  # y
        assert coords[0, 3].item() == 3  # z

    def test_flatten_unflatten_roundtrip(self):
        """Test that flatten and unflatten are inverses."""
        coords = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 3, 3], [1, 100, 200, 50]])

        flat = flatten_coord(coords)
        unflat = unflatten_coord(flat)

        assert torch.equal(coords, unflat)


class TestHashCoords:
    """Test hash_coords function."""

    def test_hash_coords_basic(self):
        """Test basic coordinate hashing."""
        coords = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 4]])
        hashes = hash_coords(coords)

        assert hashes.shape == (2,)
        assert hashes.dtype == torch.int32

    def test_hash_coords_non_negative(self):
        """Test that hashes are non-negative."""
        coords = torch.randn(10, 4).abs().int()
        hashes = hash_coords(coords)

        assert torch.all(hashes >= 0)

    def test_hash_coords_distribution(self):
        """Test that hashes are well-distributed."""
        # Generate many random coordinates
        coords = torch.randint(0, 100, (1000, 4))
        hashes = hash_coords(coords)

        # Check that we have many unique hashes
        unique_hashes = torch.unique(hashes)
        assert len(unique_hashes) > 500  # At least 50% unique

    def test_hash_coords_different_coords(self):
        """Test that different coordinates (usually) hash differently."""
        coords1 = torch.tensor([[0, 1, 2, 3]])
        coords2 = torch.tensor([[0, 1, 2, 4]])

        hash1 = hash_coords(coords1)[0].item()
        hash2 = hash_coords(coords2)[0].item()

        assert hash1 != hash2


class TestHashTable:
    """Test HashTable class."""

    def test_init_with_capacity(self):
        """Test initialization with capacity."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for HashTable")

        table = HashTable(capacity=1000, device="cuda")

        assert table.capacity == 1000
        assert table.table_keys.shape == (1000,)
        assert table.table_values.shape == (1000,)
        assert torch.all(table.table_keys == -1)
        assert torch.all(table.table_values == -1)

    def test_init_with_existing_tables(self):
        """Test initialization with existing tables."""
        keys = torch.full((100,), -1, dtype=torch.int32)
        values = torch.full((100,), -1, dtype=torch.int32)

        table = HashTable(table_keys=keys, table_values=values)

        assert table.capacity == 100
        assert torch.equal(table.table_keys, keys)
        assert torch.equal(table.table_values, values)

    def test_init_invalid(self):
        """Test initialization with invalid arguments."""
        with pytest.raises(AssertionError, match="must have the same shape"):
            HashTable(table_keys=torch.ones(100), table_values=torch.ones(50))

    def test_device_property(self):
        """Test device property."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for HashTable")

        table = HashTable(capacity=100, device="cuda")

        assert table.device.type == "cuda"

    def test_to_device(self):
        """Test to() method."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for HashTable")

        table = HashTable(capacity=100, device="cuda")

        table.to("cuda")
        assert table.device.type == "cuda"

    def test_cpu_method(self):
        """Test cpu() method."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for HashTable")

        table = HashTable(capacity=100, device="cuda")

        table_cpu = table.cpu()
        assert table_cpu.device.type == "cpu"

    def test_insert(self):
        """Test insert method."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for HashTable")

        table = HashTable(capacity=1000, device="cuda")

        coords = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 3, 3]], dtype=torch.int32, device="cuda")

        table.insert(coords)

        # Check that some entries are now filled (not -1)
        assert torch.any(table.table_keys != -1)

    def test_insert_invalid_capacity(self):
        """Test insert with insufficient capacity."""
        table = HashTable(capacity=10, device="cpu")

        coords = torch.randint(0, 10, (20, 4), dtype=torch.int32)

        with pytest.raises(AssertionError, match="capacity should be at least"):
            table.insert(coords)

    def test_query(self):
        """Test query method."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for HashTable")

        table = HashTable(capacity=1000, device="cuda")

        # Insert coordinates
        coords = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 3, 3]], dtype=torch.int32, device="cuda")
        table.insert(coords)

        # Query same coordinates
        query_coords = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 4]], dtype=torch.int32, device="cuda")
        indices = table.query(query_coords)

        assert indices.shape == (2,)
        # Should find indices 0 and 1
        assert indices[0].item() == 0
        assert indices[1].item() == 1

    def test_query_not_found(self):
        """Test querying coordinates that don't exist."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for HashTable")

        table = HashTable(capacity=1000, device="cuda")

        coords = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32, device="cuda")
        table.insert(coords)

        # Query different coordinate
        query_coords = torch.tensor([[0, 5, 5, 5]], dtype=torch.int32, device="cuda")
        indices = table.query(query_coords)

        assert indices[0].item() == -1

    def test_query_multiple(self):
        """Test querying multiple coordinates."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for HashTable")

        table = HashTable(capacity=1000, device="cuda")

        coords = torch.tensor(
            [[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 3, 3], [1, 0, 0, 0]], dtype=torch.int32, device="cuda"
        )
        table.insert(coords)

        # Query all coordinates
        indices = table.query(coords)

        assert torch.all(indices >= 0)
        assert indices[0].item() == 0
        assert indices[1].item() == 1
        assert indices[2].item() == 2
        assert indices[3].item() == 3
