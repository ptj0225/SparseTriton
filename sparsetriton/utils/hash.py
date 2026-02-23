"""Hash table utilities for sparse coordinate indexing.

This module provides hash-based data structures for efficient coordinate
lookups in sparse tensors. Uses Triton kernels for GPU-accelerated operations.

Example:
    >>> import torch
    >>> from sparsetriton.utils.hash import HashTable, flatten_coord, unflatten_coord, hash_coords
    >>> coords = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 3, 3]])
    >>> flat = flatten_coord(coords)
    >>> unflat = unflatten_coord(flat)
    >>> torch.allclose(unflat, coords)
    True
"""

import torch
import triton
import triton.language as tl

from sparsetriton.config import get_coords_dtype, get_h_table_max_p

__all__ = ["HashTable", "flatten_coord", "hash_coords", "unflatten_coord"]


def flatten_coord(coords: torch.Tensor) -> torch.Tensor:
    """Flatten 4D coordinates to a single 64-bit integer.

    Packs (batch, x, y, z) coordinates into a single 64-bit integer using
    bit shifts. Each coordinate dimension uses 16 bits.

    Args:
        coords: Coordinate tensor of shape (N, 4) in (batch, x, y, z) format

    Returns:
        torch.Tensor: Flattened coordinate tensor of shape (N,) with dtype int64

    Raises:
        AssertionError: If coords doesn't have shape (N, 4)

    Example:
        >>> import torch
        >>> from sparsetriton.utils.hash import flatten_coord, unflatten_coord
        >>> coords = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 3, 3]])
        >>> flat = flatten_coord(coords)
        >>> unflat = unflatten_coord(flat)
        >>> torch.allclose(unflat, coords)
        True
    """
    return (
        (coords[:, 0].to(torch.int64) << 48)
        | (coords[:, 1].to(torch.int64) << 32)
        | (coords[:, 2].to(torch.int64) << 16)
        | coords[:, 3].to(torch.int64)
    )


def unflatten_coord(flat_coords: torch.Tensor) -> torch.Tensor:
    """Unflatten a single 64-bit integer back to 4D coordinates.

    Extracts (batch, x, y, z) coordinates from a packed 64-bit integer.
    Each dimension is extracted from 16-bit chunks.

    Args:
        flat_coords: Flattened coordinate tensor of shape (N,) with dtype int64

    Returns:
        torch.Tensor: Unflattened coordinate tensor of shape (N, 4) in
                     (batch, x, y, z) format with coords_dtype

    Example:
        >>> import torch
        >>> from sparsetriton.utils.hash import flatten_coord, unflatten_coord
        >>> coords = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 3, 3]])
        >>> flat = flatten_coord(coords)
        >>> unflat = unflatten_coord(flat)
        >>> torch.allclose(unflat, coords)
        True
    """
    b = (flat_coords >> 48) & 0xFFFF
    x = (flat_coords >> 32) & 0xFFFF
    y = (flat_coords >> 16) & 0xFFFF
    z = flat_coords & 0xFFFF
    return torch.stack([b, x, y, z], dim=1).to(get_coords_dtype())


def hash_coords(coords: torch.Tensor) -> torch.Tensor:
    """Hash 4D coordinates to 32-bit integer values.

    Uses a spatial hashing function that combines all four coordinate
    dimensions with prime number multipliers for good distribution.

    Args:
        coords: Coordinate tensor of shape (N, 4) in (batch, x, y, z) format

    Returns:
        torch.Tensor: Hash values of shape (N,) with dtype int32 (non-negative)

    Example:
        >>> import torch
        >>> from sparsetriton.utils.hash import hash_coords
        >>> coords = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 3, 3]])
        >>> hashes = hash_coords(coords)
        >>> hashes.shape
        torch.Size([3])
        >>> torch.all(hashes >= 0)
        tensor(True)
    """
    return (
        (coords[:, 1].to(torch.int32) * 73856093)
        ^ (coords[:, 2].to(torch.int32) * 19349663)
        ^ (coords[:, 3].to(torch.int32) * 83492791)
        ^ (coords[:, 0].to(torch.int32) * 1000003)
    ) & 0x7FFFFFFF


@triton.jit
def hash_coords_kernel(b, x, y, z):
    """Triton kernel for hashing 4D coordinates.

    Packs B, X, Y, Z (each 16-bit) into a 64-bit integer with
    optimal bit arrangement for spatial locality. Uses the upper
    24 bits for batch to ensure locality within the same batch.

    Args:
        b: Batch indices
        x: X coordinates
        y: Y coordinates
        z: Z coordinates

    Returns:
        int32: Hashed value
    """
    h = (
        (b.to(tl.uint64) << 24)
        | ((x.to(tl.uint64) & 0xFF) << 16)
        | ((y.to(tl.uint64) & 0xFF) << 8)
        | (z.to(tl.uint64) & 0xFF)
    )
    return h.to(tl.int32) & 0x7FFFFFFF


@triton.jit
def hash_coords_kernel2(b, x, y, z):
    """Alternative Triton kernel for spatial hashing.

    Uses different prime multipliers for collision resistance.

    Args:
        b: Batch indices
        x: X coordinates
        y: Y coordinates
        z: Z coordinates

    Returns:
        int32: Hashed value
    """
    h = (
        (x.to(tl.uint64) * 982451653)
        ^ (y.to(tl.uint64) * 701000767)
        ^ (z.to(tl.uint64) * 1610612741)
        ^ (b.to(tl.uint64) * 67867979)
    )
    return h.to(tl.int32) & 0x7FFFFFFF


@triton.jit
def flatten_coords_kernel(b, x, y, z):
    """Triton kernel for flattening 4D coordinates.

    Packs (batch, x, y, z) into a 64-bit integer with full 16-bit
    allocation for each dimension.

    Args:
        b: Batch indices
        x: X coordinates
        y: Y coordinates
        z: Z coordinates

    Returns:
        int64: Flattened coordinate
    """
    return (b.to(tl.int64) << 48) | (x.to(tl.int64) << 32) | (y.to(tl.int64) << 16) | z.to(tl.int64)


@triton.jit
def get_probe_offsets_impl(hashes, probe_step, table_size):
    """Calculate probe offset for hash table collision resolution.

    Implements linear probing with a hash function on the probe step.

    Args:
        hashes: Base hash values
        probe_step: Current probe step (starting from 0)
        table_size: Size of the hash table

    Returns:
        uint32: The index to probe
    """
    curr_hash = (hashes + 83492791 * (probe_step // 8) + probe_step).to(tl.uint32)
    curr_hash %= tl.cast(table_size, tl.uint32)
    return curr_hash


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
    ],
    key=["tune_N"],
)
@triton.jit
def build_hash_table_kernel(
    coords_ptr,
    hash_keys_ptr,
    hash_vals_ptr,
    table_size,
    N,
    tune_N,
    BLOCK_SIZE: tl.constexpr,
    max_probe_step: tl.constexpr = 128,
):
    """Build a hash table mapping packed coordinates to voxel indices.

    Uses linear probing for collision resolution. Each coordinate gets
    hashed, and we probe until finding an empty slot or an existing
    entry with the same key.

    Args:
        coords_ptr: Pointer to coordinate data (N, 4) flattened
        hash_keys_ptr: Pointer to hash table keys array
        hash_vals_ptr: Pointer to hash table values array
        table_size: Size of the hash table
        N: Number of coordinates to insert
        tune_N: Autotuning key
        BLOCK_SIZE: Block size for kernel execution
        max_probe_step: Maximum number of probe attempts
    """
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < N

    coords_base_idx = idx * 4
    b = tl.load(coords_ptr + coords_base_idx + 0, mask=mask)
    x = tl.load(coords_ptr + coords_base_idx + 1, mask=mask)
    y = tl.load(coords_ptr + coords_base_idx + 2, mask=mask)
    z = tl.load(coords_ptr + coords_base_idx + 3, mask=mask)

    hashes = hash_coords_kernel(b, x, y, z)
    keys = hash_coords_kernel2(b, x, y, z)

    active_mask = mask

    probe_step = 0
    while (tl.max(active_mask.to(tl.int32), axis=0) > 0) & (probe_step < max_probe_step):
        curr_hash = get_probe_offsets_impl(hashes, probe_step, table_size)
        compare_vals = tl.where(active_mask, -1, -2)
        old_key = tl.atomic_cas(hash_keys_ptr + curr_hash, compare_vals, keys)

        # Success if slot was empty (-1) or already had our key
        success = active_mask & ((old_key == -1) | (old_key == keys))

        # Store only on successful insertion
        tl.store(hash_vals_ptr + curr_hash, idx, mask=success)

        # Deactivate successful threads
        active_mask = active_mask & (~success)
        probe_step += 1


@triton.jit
def query_hash_table_impl(
    hashes, keys, table_keys_ptr, table_values_ptr, table_size, idx, N, BLOCK_SIZE, max_probe_step: tl.constexpr = 128
):
    """Query hash table for coordinate indices.

    Probes the hash table until finding a matching key or determining
    the coordinate doesn't exist.

    Args:
        hashes: Base hash values
        keys: Hash keys to match
        table_keys_ptr: Pointer to hash table keys array
        table_values_ptr: Pointer to hash table values array
        table_size: Size of the hash table
        idx: Thread indices
        N: Number of queries
        BLOCK_SIZE: Block size
        max_probe_step: Maximum number of probe attempts

    Returns:
        Tensor of found indices (-1 if not found)
    """
    active_mask = idx < N
    probe_step = 0
    result = tl.full((BLOCK_SIZE,), -1, dtype=tl.int32)
    while (tl.max(active_mask.to(tl.int1), axis=0) > 0) & (probe_step < max_probe_step):
        curr_hash = get_probe_offsets_impl(hashes, probe_step, table_size)
        loaded_key = tl.load(table_keys_ptr + curr_hash, mask=active_mask, other=-1)
        found_mask = active_mask & (loaded_key == keys)
        empty_mask = loaded_key == -1
        val = tl.load(table_values_ptr + curr_hash, mask=found_mask, other=-1)
        result = tl.where(found_mask, val, result)
        active_mask = active_mask & (~found_mask) & (~empty_mask)
        probe_step += 1
    return result


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
    ],
    key=["tune_N"],
)
@triton.jit
def query_hash_table_kernel(
    coords_ptr,
    out_values_ptr,
    table_keys_ptr,
    table_values_ptr,
    table_size,
    N,
    tune_N,
    BLOCK_SIZE: tl.constexpr,
    max_probe_step: tl.constexpr = 128,
):
    """Query kernel for hash table lookups.

    Loads coordinates, computes hashes, and queries the hash table
    to find corresponding indices.

    Args:
        coords_ptr: Pointer to coordinate data
        out_values_ptr: Pointer to output values array
        table_keys_ptr: Pointer to hash table keys array
        table_values_ptr: Pointer to hash table values array
        table_size: Size of the hash table
        N: Number of queries
        tune_N: Autotuning key
        BLOCK_SIZE: Block size for kernel execution
        max_probe_step: Maximum number of probe attempts
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    idx = block_start + tl.arange(0, BLOCK_SIZE)
    mask = idx < N

    coords_base_idx = idx * 4
    b = tl.load(coords_ptr + coords_base_idx + 0, mask=mask)
    x = tl.load(coords_ptr + coords_base_idx + 1, mask=mask)
    y = tl.load(coords_ptr + coords_base_idx + 2, mask=mask)
    z = tl.load(coords_ptr + coords_base_idx + 3, mask=mask)

    hash = hash_coords_kernel(b, x, y, z) % table_size
    keys = hash_coords_kernel2(b, x, y, z)
    query_out = query_hash_table_impl(
        hashes=hash,
        keys=keys,
        table_keys_ptr=table_keys_ptr,
        table_values_ptr=table_values_ptr,
        table_size=table_size,
        idx=idx,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
        max_probe_step=max_probe_step,
    )
    store_mask = (query_out != -1) & mask
    tl.store(out_values_ptr + idx, query_out, mask=store_mask)


class HashTable:
    """GPU-accelerated hash table for sparse coordinate lookups.

    Uses Triton kernels for fast insertion and query operations on GPU.
    Implements linear probing for collision resolution.

    Attributes:
        table_keys: Array of hash keys (int32)
        table_values: Array of values (voxel indices) (int32)

    Example:
        >>> import torch
        >>> from sparsetriton.utils.hash import HashTable
        >>> # Create hash table with capacity 1000
        >>> table = HashTable(capacity=1000, device="cpu")
        >>> # Insert coordinates
        >>> coords = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 3, 3]])
        >>> table.insert(coords)
        >>> # Query coordinates
        >>> query_coords = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 4]])
        >>> indices = table.query(query_coords)
        >>> print(indices)
        tensor([0, 1])
    """

    def __init__(
        self,
        capacity: int = None,
        device: torch.device = "cpu",
        table_keys: torch.Tensor = None,
        table_values: torch.Tensor = None,
    ):
        """Initialize a HashTable.

        Args:
            capacity: Number of slots in the hash table. Required if not loading existing tables.
            device: Device for the hash table (default: "cpu")
            table_keys: Pre-existing keys array for loading (optional)
            table_values: Pre-existing values array for loading (optional)

        Raises:
            AssertionError: If neither capacity nor both table_keys/table_values are provided
            AssertionError: If table_keys and table_values have different shapes

        Example:
            >>> import torch
            >>> from sparsetriton.utils.hash import HashTable
            >>> table = HashTable(capacity=1000, device="cpu")
            >>> table.capacity
            1000
        """
        assert (
            capacity is not None or (table_keys is not None and table_values is not None)
        ), "Either capacity or both table_keys and table_values must be provided."

        if table_keys is not None and table_values is not None:
            assert table_keys.shape == table_values.shape, "table_keys and table_values must have the same shape."
            self.table_keys, self.table_values = table_keys, table_values
        else:
            self.table_keys = torch.full((capacity,), -1, dtype=torch.int32, device=device)
            self.table_values = torch.full((capacity,), -1, dtype=torch.int32, device=device)

    @property
    def device(self) -> torch.device:
        """Get the device of the hash table.

        Returns:
            torch.device: Current device

        Example:
            >>> import torch
            >>> from sparsetriton.utils.hash import HashTable
            >>> table = HashTable(capacity=1000, device="cpu")
            >>> table.device
            device(type='cpu')
        """
        return self.table_keys.device

    @device.setter
    def device(self, device: torch.device):
        """Set the device of the hash table.

        Args:
            device: Target device

        Example:
            >>> import torch
            >>> from sparsetriton.utils.hash import HashTable
            >>> table = HashTable(capacity=1000, device="cpu")
            >>> table.device = torch.device("cuda:0")  # if CUDA available
        """
        self.table_keys = self.table_keys.to(device, non_blocking=True)
        self.table_values = self.table_values.to(device, non_blocking=True)

    def cpu(self) -> "HashTable":
        """Move hash table to CPU.

        Returns:
            HashTable: self after moving to CPU

        Example:
            >>> import torch
            >>> from sparsetriton.utils.hash import HashTable
            >>> table = HashTable(capacity=1000, device="cpu")
            >>> table_cpu = table.cpu()
            >>> table_cpu.device.type
            'cpu'
        """
        self.device = torch.device("cpu")
        return self

    def to(self, device: torch.device) -> "HashTable":
        """Move hash table to specified device.

        Args:
            device: Target device

        Returns:
            HashTable: self after moving

        Example:
            >>> import torch
            >>> from sparsetriton.utils.hash import HashTable
            >>> table = HashTable(capacity=1000, device="cpu")
            >>> table_cpu = table.to("cpu")
            >>> table_cpu.device.type
            'cpu'
        """
        self.device = device
        return self

    @property
    def capacity(self) -> int:
        """Get the hash table capacity.

        Returns:
            int: Number of slots in the hash table

        Example:
            >>> import torch
            >>> from sparsetriton.utils.hash import HashTable
            >>> table = HashTable(capacity=1000, device="cpu")
            >>> table.capacity
            1000
        """
        return self.table_keys.shape[0]

    def insert(self, keys: torch.Tensor) -> None:
        """Insert coordinates into the hash table.

        Args:
            keys: Coordinate tensor of shape (N, 4) to insert

        Raises:
            AssertionError: If capacity is less than number of keys

        Example:
            >>> import torch
            >>> from sparsetriton.utils.hash import HashTable
            >>> table = HashTable(capacity=1000, device="cpu")
            >>> coords = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 3, 3]])
            >>> table.insert(coords)
        """
        keys = keys.contiguous()
        N = keys.shape[0]
        assert self.capacity >= keys.shape[0], "Hash table capacity should be at least twice the number of keys to ensure low collision rate."
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)

        build_hash_table_kernel[grid](
            keys,  # coords_ptr
            self.table_keys,  # hash_keys_ptr
            self.table_values,  # hash_vals_ptr
            self.capacity,  # table_size
            N,  # N
            triton.next_power_of_2(N),
        )

    def query(self, keys: torch.Tensor) -> torch.Tensor:
        """Query the hash table for coordinate indices.

        Args:
            keys: Coordinate tensor of shape (N, 4) to query

        Returns:
            torch.Tensor: Indices of matching coordinates, or -1 if not found

        Example:
            >>> import torch
            >>> from sparsetriton.utils.hash import HashTable
            >>> table = HashTable(capacity=1000, device="cpu")
            >>> coords = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 3, 3]])
            >>> table.insert(coords)
            >>> query_coords = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 4], [0, 5, 5, 5]])
            >>> indices = table.query(query_coords)
            >>> print(indices)
            tensor([ 0,  1, -1])
        """
        keys = keys.contiguous()
        N = keys.shape[0]
        out_values = torch.full((N,), -1, dtype=torch.int32, device=self.device)
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)

        query_hash_table_kernel[grid](
            keys,
            out_values,
            self.table_keys,
            self.table_values,
            self.capacity,
            N,
            triton.next_power_of_2(N),
        )
        return out_values
