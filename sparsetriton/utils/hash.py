import torch
import triton
import triton.language as tl

from sparsetriton.config import get_coords_dtype

__all__ = ["HashTable", "flatten_coord", "hash_coords", "unflatten_coord"]

def flatten_coord(coords:torch.Tensor) -> torch.Tensor:
    return (coords[:, 0].to(torch.int64) << 48) | (coords[:, 1].to(torch.int64) << 32) | (coords[:, 2].to(torch.int64) << 16) | coords[:, 3].to(torch.int64)

def unflatten_coord(flat_coords:torch.Tensor) -> torch.Tensor:
    b = (flat_coords >> 48) & 0xFFFF
    x = (flat_coords >> 32) & 0xFFFF
    y = (flat_coords >> 16) & 0xFFFF
    z = flat_coords & 0xFFFF
    return torch.stack([b, x, y, z], dim=1).to(get_coords_dtype())

def hash_coords(coords:torch.Tensor) -> torch.Tensor:
    return (coords[:, 0].to(torch.int64) * 73856093) ^ (coords[:, 1].to(torch.int64) * 19349663) ^ (coords[:, 2].to(torch.int64) * 83492791) ^ (coords[:, 3].to(torch.int64) * 1000003)

@triton.jit
def hash_coords_kernel(b, x, y, z):
    """Simple spatial hashing function."""
    return (x.to(tl.int64) * 73856093) ^ (y.to(tl.int64) * 19349663) ^ (z.to(tl.int64) * 83492791) ^ (b.to(tl.int64) * 1000003)

@triton.jit
def flatten_coords_kernel(b, x, y, z):
    return (b.to(tl.int64) << 48) | (x.to(tl.int64) << 32) | (y.to(tl.int64) << 16) | z.to(tl.int64)

@triton.jit
def build_hash_table_kernel(
    coords_ptr, hash_keys_ptr, hash_vals_ptr,
    table_size, N,
    failure_ptr,
    BLOCK_SIZE: tl.constexpr
):
    """
    Builds a hash table mapping packed coordinates to voxel indices.
    Uses linear probing for collision resolution.
    """
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < N

    # Load coordinates (Batch, X, Y, Z)
    b = tl.load(coords_ptr + idx * 4 + 0, mask=mask)
    x = tl.load(coords_ptr + idx * 4 + 1, mask=mask)
    y = tl.load(coords_ptr + idx * 4 + 2, mask=mask)
    z = tl.load(coords_ptr + idx * 4 + 3, mask=mask)

    # Pack coordinates into a unique key
    key = flatten_coords_kernel(b, x, y, z)
    hash_val = hash_coords_kernel(b, x, y, z) % table_size

    active_mask = mask 

    probe_step = 0
    while (tl.max(active_mask.to(tl.int32), axis=0) > 0) & (probe_step < 256):
        # 활성화된 스레드만 해시 계산 및 atomic 연산 수행
        curr_hash = (hash_val + probe_step) % table_size
        
        # tl.atomic_cas는 마스크를 직접 지원하지 않으므로 
        compare_vals = tl.where(active_mask, -1, -2).to(tl.int64)
        old_key = tl.atomic_cas(hash_keys_ptr + curr_hash, compare_vals, key)
        
        # 삽입 성공 조건: 빈 자리(-1)였거나, 이미 내 키가 들어있거나
        success = (old_key == -1) | (old_key == key)
        
        # 이번 루프에서 성공했고, 아직 처리 중이었던 스레드만 store
        write_mask = active_mask & success
        tl.store(hash_vals_ptr + curr_hash, idx, mask=write_mask)
        
        # 성공한 스레드는 다음 루프부터 제외
        active_mask = active_mask & (~success)
        probe_step += 1
    tl.store(failure_ptr + pid, tl.max(active_mask))

@triton.jit
def query_hash_table_kernel(
    keys_ptr,
    out_values_ptr,
    table_keys_ptr,
    table_values_ptr,
    table_size,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    idx = block_start + tl.arange(0, BLOCK_SIZE)
    mask = idx < N

    b = tl.load(keys_ptr + idx * 4 + 0, mask=mask)
    x = tl.load(keys_ptr + idx * 4 + 1, mask=mask)
    y = tl.load(keys_ptr + idx * 4 + 2, mask=mask)
    z = tl.load(keys_ptr + idx * 4 + 3, mask=mask)
    


    hash = hash_coords_kernel(
        b, x, y, z
    ) % table_size

    probe_step = 0
    active_mask = mask 
    key = flatten_coords_kernel(b, x, y, z)
    while (tl.max(active_mask.to(tl.int32), axis=0) > 0) & (probe_step < 256):
        curr_hash = (hash + probe_step) % table_size
        loaded_key = tl.load(table_keys_ptr + curr_hash, mask=active_mask)
        write_mask = active_mask & (loaded_key == key)
        tl.store(out_values_ptr + idx, tl.load(table_values_ptr + curr_hash), mask=write_mask)
        active_mask = active_mask & (~write_mask)
        probe_step += 1

@triton.jit
def coalesce_coords_kernel(
    coords_ptr, # (N, 4)
    valids_ptr, # (N)
    hash_keys_ptr, # (2 * N)
    table_size, 
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < N
    b = tl.load(coords_ptr + idx * 4 + 0, mask=mask)
    x = tl.load(coords_ptr + idx * 4 + 1, mask=mask)
    y = tl.load(coords_ptr + idx * 4 + 2, mask=mask)
    z = tl.load(coords_ptr + idx * 4 + 3, mask=mask)

    hash_vals = hash_coords_kernel(b, x, y, z) % table_size
    hash_keys = flatten_coords_kernel(b, x, y, z)
    step = 0
    active_mask = mask
    duplicated = mask
    while (tl.max(active_mask.to(tl.int32), axis=0) > 0) and (step < 128):
        curr_hashs = (hash_vals + step) % table_size
        compare_val = tl.where(active_mask, -1, -2).to(tl.int64)
        old_keys = tl.atomic_cas(hash_keys_ptr + curr_hashs, compare_val, hash_keys)
        duplicated = duplicated | ((old_keys == hash_keys) & active_mask)
        success = (old_keys == -1) |  (old_keys == hash_keys)
        active_mask = active_mask & (~success)
        step += 1

    tl.store(valids_ptr + idx, False, mask=duplicated)
        
def coalesce_coords(coords: torch.Tensor):
    coords = coords.contiguous()
    N = coords.shape[0]
    BLOCK_SIZE = 128
    meta = (triton.cdiv(N, BLOCK_SIZE),)
    valids = torch.full((N,), True, device=coords.device, dtype=torch.bool)
    hash_keys = torch.full((N * 4,), -1, device=coords.device, dtype=torch.int64)
    table_size = hash_keys.shape[0]
    coalesce_coords_kernel[meta](
        coords,
        valids,
        hash_keys,
        table_size,
        N,
        BLOCK_SIZE
    )
    return valids

class HashTable:
    def __init__(self, capacity: int = None, device: torch.device = "cpu", table_keys: torch.Tensor = None, table_values: torch.Tensor = None):
        assert capacity is not None or (table_keys is not None and table_values is not None), "Either capacity or both table_keys and table_values must be provided."
        assert not (capacity is not None and (table_keys is not None and table_values is not None)), "If both capacity and table tensors are provided, capacity is ignored."
        assert (table_keys is None and table_values is None) or (table_keys.shape == table_values.shape), "table_keys and table_values must have the same shape."

        self.table_keys = table_keys if table_keys is not None else torch.full((capacity,), -1, dtype=torch.int64, device=device)
        self.table_values = table_values if table_values is not None else torch.full((capacity,), -1, dtype=torch.int64, device=device)

    @property
    def device(self) -> torch.device:
        return self.table_keys.device
    
    @device.setter
    def device(self, device: torch.device):
        self.table_keys = self.table_keys.to(device, non_blocking=True)
        self.table_values = self.table_values.to(device, non_blocking=True)

    def cpu(self):
        self.device = torch.device("cpu")
        return self
    
    def to(self, device: torch.device):
        self.device = device
        return self

    @property
    def capacity(self) -> int:
        return self.table_keys.shape[0]
    
    def insert(self, keys: torch.Tensor):
        N = keys.shape[0]
        assert self.capacity >= keys.shape[0], "Hash table capacity should be at least twice the number of keys to ensure low collision rate."

        # 1. BLOCK_SIZE를 상수로 정의
        BLOCK_SIZE = 128
        # 2. grid를 람다가 아닌 튜플로 직접 전달 (가장 깔끔한 방법)
        grid = (triton.cdiv(N, BLOCK_SIZE),)
        failure = torch.zeros((grid[0],), dtype=torch.int32, device=self.device)

        build_hash_table_kernel[grid](
            keys,           # coords_ptr
            self.table_keys,   # hash_keys_ptr
            self.table_values, # hash_vals_ptr
            self.capacity,     # table_size
            N,                 # N
            failure,
            BLOCK_SIZE=BLOCK_SIZE # 명시적 전달
        )
        failure_n = torch.sum(failure).item()
        if (failure_n > 0):
            print(f"Warning: {failure_n} keys could not be inserted into the hash table due to excessive collisions.")

    def query(self, keys: torch.Tensor) -> torch.Tensor:
        N = keys.shape[0]
        BLOCK_SIZE = 256
        out_values = torch.full((N,), -1, dtype=torch.int64, device=self.device)
        grid = (triton.cdiv(N, BLOCK_SIZE),)
        
        query_hash_table_kernel[grid](
            keys, out_values, self.table_keys, self.table_values, 
            self.capacity, N, 
            BLOCK_SIZE=BLOCK_SIZE
        )
        return out_values