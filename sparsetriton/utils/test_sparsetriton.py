import torch
import sys
import os

# Add parent directory to path to import sparsetriton
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sparsetriton.tensor import SparseTensor
from sparsetriton.config import set_coords_dtype
from sparsetriton.utils.hash import HashTable


def test_sparse_tensor_basic():
    print("Testing SparseTensor basic...")
    if not torch.cuda.is_available():
        print("Skipping CUDA tests")
        return

    device = torch.device("cuda")

    # Test int16 coords
    set_coords_dtype(torch.int16)
    feats = torch.randn(10, 16, device=device)
    coords = torch.randint(0, 10, (10, 4), device=device)

    st = SparseTensor(feats, coords)
    assert st.coords.dtype == torch.int16
    assert st.feats.device.type == "cuda"

    # Test int32 coords
    set_coords_dtype(torch.int32)
    st = SparseTensor(feats, coords)
    assert st.coords.dtype == torch.int32
    print("Passed")


def test_to_dense():
    print("Testing to_dense...")
    if not torch.cuda.is_available():
        return

    device = torch.device("cuda")
    set_coords_dtype(torch.int16)

    # 3 points in 3D space
    feats = torch.tensor([[1.0], [2.0], [3.0]], device=device)
    # coords: (N, 3) for 3D spatial range
    coords = torch.tensor([[0, 0, 0], [1, 1, 1], [0, 1, 0]], device=device)

    st = SparseTensor(feats, coords)

    spatial_range = (2, 2, 2)
    dense = st.dense(spatial_range)

    # Dense shape: (2, 2, 2, 1)
    assert dense.shape == (2, 2, 2, 1)

    assert dense[0, 0, 0, 0] == 1.0
    assert dense[1, 1, 1, 0] == 2.0
    assert dense[0, 1, 0, 0] == 3.0
    assert dense[0, 0, 1, 0] == 0.0  # Empty

    print("Passed")


def test_hash_table():
    print("Testing HashTable...")
    if not torch.cuda.is_available():
        return

    device = torch.device("cuda")
    set_coords_dtype(torch.int32)

    capacity = 1
    ht = HashTable(capacity, device)

    keys = torch.tensor([
        [10, 0, 0, 0],
        [20, 0, 0, 0],
        [30, 0, 0, 0]
    ], dtype=torch.int16, device=device)
    values = torch.tensor([1124, 2123, 3123], dtype=torch.int32, device=device)

    ht.insert(keys)
    out = ht.query(keys)
    print(out)

    assert torch.all(values[out] == values)
    print("Passed")

if __name__ == "__main__":
    test_sparse_tensor_basic()
    test_to_dense()
    test_hash_table()
