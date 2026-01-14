import torch
import triton
import triton.language as tl

from sparsetriton.config import get_coords_dtype

__all__ = ["HashTable"]

@triton.jit
def _hash_coords(b, x, y, z):
    """Simple spatial hashing function."""
    return (x.to(tl.int64) * 73856093) ^ (y.to(tl.int64) * 19349663) ^ (z.to(tl.int64) * 83492791) ^ (b.to(tl.int64) * 1000003)

@triton.jit
def _flatten_coords(b, x, y, z):
    return (b.to(tl.int64) << 36) | (x.to(tl.int64) << 24) | (y.to(tl.int64) << 12) | z.to(tl.int64)

@triton.jit
def build_hash_table_kernel(
    coords_ptr, hash_keys_ptr, hash_vals_ptr,
    table_size, N,
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
    key = _flatten_coords(b, x, y, z)
    hash_val = _hash_coords(b, x, y, z) % table_size

    active_mask = mask 

    dummy_ptr = hash_keys_ptr + table_size
    probe_step = 0
    while (tl.max(active_mask.to(tl.int32), axis=0) > 0) & (probe_step < 32):
        # 활성화된 스레드만 해시 계산 및 atomic 연산 수행
        curr_hash = (hash_val + probe_step) % table_size
        
        # tl.atomic_cas는 마스크를 직접 지원하지 않으므로 
        target_ptr = tl.where(active_mask, hash_keys_ptr + curr_hash, dummy_ptr)
        old_key = tl.atomic_cas(target_ptr, tl.full((BLOCK_SIZE,), -1, dtype=tl.int64), key)
        
        # 삽입 성공 조건: 빈 자리(-1)였거나, 이미 내 키가 들어있거나
        success = (old_key == -1) | (old_key == key)
        
        # 이번 루프에서 성공했고, 아직 처리 중이었던 스레드만 store
        write_mask = active_mask & success
        tl.store(hash_vals_ptr + curr_hash, idx, mask=write_mask)
        
        # 성공한 스레드는 다음 루프부터 제외
        active_mask = active_mask & (~success)
        probe_step += 1

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
    


    hash = _hash_coords(
        b, x, y, z
    ) % table_size

    probe_step = 0
    active_mask = mask 
    key = _flatten_coords(b, x, y, z)
    while (tl.max(active_mask.to(tl.int32), axis=0) > 0) & (probe_step < 32):
        curr_hash = (hash + probe_step) % table_size
        loaded_key = tl.load(table_keys_ptr + curr_hash, mask=active_mask)
        tl.store(out_values_ptr + idx, tl.load(table_values_ptr + curr_hash), mask=active_mask)
        active_mask = active_mask & (loaded_key != key)
        probe_step += 1


class HashTable:
    def __init__(self, capacity: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.table_keys = torch.full((capacity + 1,), -1, dtype=torch.int64, device=device)
        self.table_values = torch.full((capacity + 1,), -1, dtype=torch.int64, device=device)

    def insert(self, keys: torch.Tensor):
        N = keys.shape[0]
        # 1. BLOCK_SIZE를 상수로 정의
        BLOCK_SIZE = 128
        # 2. grid를 람다가 아닌 튜플로 직접 전달 (가장 깔끔한 방법)
        grid = (triton.cdiv(N, BLOCK_SIZE),)
        
        build_hash_table_kernel[grid](
            keys,           # coords_ptr
            self.table_keys,   # hash_keys_ptr
            self.table_values, # hash_vals_ptr
            self.capacity,     # table_size
            N,                 # N
            BLOCK_SIZE=BLOCK_SIZE # 명시적 전달
        )

    def query(self, keys: torch.Tensor) -> torch.Tensor:
        N = keys.shape[0]
        BLOCK_SIZE = 128
        out_values = torch.full((N,), -1, dtype=torch.int64, device=self.device)
        grid = (triton.cdiv(N, BLOCK_SIZE),)
        
        query_hash_table_kernel[grid](
            keys, out_values, self.table_keys, self.table_values, 
            self.capacity, N, 
            BLOCK_SIZE=BLOCK_SIZE
        )
        return out_values
