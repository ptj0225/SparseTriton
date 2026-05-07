import pytest
import torch
from sparsetriton.utils.hash import HashTable, flatten_coord, unflatten_coord, hash_coords


class TestFlattenCoord:
    def test_shape(self):
        coords = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 3, 3]])
        assert flatten_coord(coords).shape == (3,)
        assert flatten_coord(coords).dtype == torch.int64

    def test_values(self):
        coords = torch.tensor([[0, 1, 2, 3]])
        flat = flatten_coord(coords)[0].item()
        assert flat == (0 << 48) | (1 << 32) | (2 << 16) | 3

    def test_large_values(self):
        coords = torch.tensor([[1, 255, 255, 255]])
        flat = flatten_coord(coords)[0].item()
        assert flat == (1 << 48) | (255 << 32) | (255 << 16) | 255


class TestUnflattenCoord:
    def test_basic(self):
        flat = torch.tensor([(0 << 48) | (1 << 32) | (2 << 16) | 3])
        coords = unflatten_coord(flat)
        assert coords.shape == (1, 4)
        assert coords[0].tolist() == [0, 1, 2, 3]

    def test_roundtrip(self):
        coords = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 3, 3], [1, 100, 200, 50]])
        assert torch.equal(coords, unflatten_coord(flatten_coord(coords)))


class TestHashCoords:
    def test_shape_and_dtype(self):
        hashes = hash_coords(torch.tensor([[0, 1, 2, 3], [0, 1, 2, 4]]))
        assert hashes.shape == (2,)
        assert hashes.dtype == torch.int32

    def test_non_negative(self):
        hashes = hash_coords(torch.randn(10, 4).abs().int())
        assert torch.all(hashes >= 0)

    def test_distribution(self):
        hashes = hash_coords(torch.randint(0, 100, (1000, 4)))
        assert torch.unique(hashes).shape[0] > 500

    def test_different_coords_hash_differently(self):
        h1 = hash_coords(torch.tensor([[0, 1, 2, 3]]))[0].item()
        h2 = hash_coords(torch.tensor([[0, 1, 2, 4]]))[0].item()
        assert h1 != h2


def _cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for HashTable")
    return torch.device("cuda")


class TestHashTable:
    def test_init_capacity(self):
        device = _cuda()
        table = HashTable(capacity=1000, device=device)
        assert table.capacity == 1000
        assert torch.all(table.table_keys == -1)
        assert torch.all(table.table_values == -1)

    def test_init_existing_tables(self):
        keys = torch.full((100,), -1, dtype=torch.int32)
        values = torch.full((100,), -1, dtype=torch.int32)
        table = HashTable(table_keys=keys, table_values=values)
        assert table.capacity == 100

    def test_init_mismatched_shapes(self):
        with pytest.raises(AssertionError, match="must have the same shape"):
            HashTable(table_keys=torch.ones(100), table_values=torch.ones(50))

    def test_device_property(self):
        device = _cuda()
        table = HashTable(capacity=100, device=device)
        assert table.device.type == "cuda"

    def test_to_device(self):
        device = _cuda()
        table = HashTable(capacity=100, device=device)
        table.to("cuda")
        assert table.device.type == "cuda"

    def test_cpu(self):
        device = _cuda()
        table = HashTable(capacity=100, device=device)
        assert table.cpu().device.type == "cpu"

    def test_insert_and_query(self):
        device = _cuda()
        table = HashTable(capacity=1000, device=device)
        coords = torch.tensor([[0,1,2,3],[0,1,2,4],[0,1,3,3]], dtype=torch.int32, device=device)
        table.insert(coords)
        assert torch.any(table.table_keys != -1)

        query = torch.tensor([[0,1,2,3],[0,1,2,4]], dtype=torch.int32, device=device)
        indices = table.query(query)
        assert indices[0].item() == 0
        assert indices[1].item() == 1

    def test_query_not_found(self):
        device = _cuda()
        table = HashTable(capacity=1000, device=device)
        table.insert(torch.tensor([[0,1,2,3]], dtype=torch.int32, device=device))
        result = table.query(torch.tensor([[0,5,5,5]], dtype=torch.int32, device=device))
        assert result[0].item() == -1

    def test_query_all_inserted(self):
        device = _cuda()
        table = HashTable(capacity=1000, device=device)
        coords = torch.tensor(
            [[0,1,2,3],[0,1,2,4],[0,1,3,3],[1,0,0,0]], dtype=torch.int32, device=device
        )
        table.insert(coords)
        indices = table.query(coords)
        assert torch.all(indices >= 0)
        assert indices.tolist() == [0, 1, 2, 3]

    def test_insert_insufficient_capacity(self):
        table = HashTable(capacity=10, device="cpu")
        coords = torch.randint(0, 10, (20, 4), dtype=torch.int32)
        with pytest.raises(AssertionError, match="capacity should be at least"):
            table.insert(coords)
