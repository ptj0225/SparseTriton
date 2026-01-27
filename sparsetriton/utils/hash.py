import torch
import triton
import triton.language as tl

from sparsetriton.config import get_coords_dtype, get_h_table_max_p

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
    return ((coords[:, 0].to(torch.int32) * 73856093) ^ (coords[:, 1].to(torch.int32) * 19349663) ^ (coords[:, 2].to(torch.int32) * 83492791) ^ (coords[:, 3].to(torch.int32) * 1000003)) & 0x7FFFFFFF

@triton.jit
def hash_coords_kernel(b, x, y, z):
    """Simple spatial hashing function."""
    h = ((x.to(tl.int32) * 73856093) ^ (y.to(tl.int32) * 19349663) ^ (z.to(tl.int32) * 83492791) ^ (b.to(tl.int32) * 1000003))
    return h & 0x7FFFFFFF

@triton.jit
def hash_coords_kernel2(b, x, y, z):
    """Simple spatial hashing function."""
    h = ((x.to(tl.int32) * 982451653) ^ (y.to(tl.int32) * 701000767) ^ (z.to(tl.int32) * 1610612741) ^ (b.to(tl.int32) * 67867979))
    return h & 0x7FFFFFFF

@triton.jit
def flatten_coords_kernel(b, x, y, z):
    return (b.to(tl.int64) << 48) | (x.to(tl.int64) << 32) | (y.to(tl.int64) << 16) | z.to(tl.int64)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['N'],
    cache_results=True
)
@triton.jit
def build_hash_table_kernel(
    coords_ptr, hash_keys_ptr, hash_vals_ptr,
    table_size, N,
    failure_ptr,
    BLOCK_SIZE: tl.constexpr,
    max_probe_step: tl.constexpr=get_h_table_max_p()
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
    hash_val = hash_coords_kernel(b, x, y, z)
    keys = hash_coords_kernel2(b, x, y, z)

    active_mask = mask 

    probe_step = 0
    while (tl.max(active_mask.to(tl.int32), axis=0) > 0) & (probe_step < max_probe_step):
        # 활성화된 스레드만 해시 계산 및 atomic 연산 수행
        curr_hash = (hash_val + probe_step) % table_size
        
        # tl.atomic_cas는 마스크를 직접 지원하지 않으므로 
        compare_vals = tl.where(active_mask, tl.full((BLOCK_SIZE,), -1, dtype=tl.int32), tl.full((BLOCK_SIZE,), -2, dtype=tl.int32))
        old_key = tl.atomic_cas(hash_keys_ptr + curr_hash, compare_vals, keys)
        
        # 삽입 성공 조건: 빈 자리(-1)였거나, 이미 내 키가 들어있거나
        success = active_mask & ((old_key == -1) | (old_key == keys))
        
        # 이번 루프에서 성공했고, 아직 처리 중이었던 스레드만 store
        tl.store(hash_vals_ptr + curr_hash, idx, mask=success)
        
        # 성공한 스레드는 다음 루프부터 제외
        active_mask = active_mask & (~success)
        probe_step += 1
    tl.store(failure_ptr + pid, tl.sum(active_mask).to(tl.int32))

@triton.jit
def query_hash_table_impl(
    hashes,
    keys,
    table_keys_ptr,
    table_values_ptr,
    table_size,
    idx,
    N,
    BLOCK_SIZE,
    max_probe_step: tl.constexpr = get_h_table_max_p()
):
    active_mask = idx < N 
    probe_step = 0
    result = tl.full((BLOCK_SIZE,), -1, dtype=tl.int32)
    while (tl.max(active_mask.to(tl.int32), axis=0) > 0) & (probe_step < max_probe_step):
        curr_hash = (hashes + probe_step) % table_size
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
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def query_hash_table_kernel(
    coords_ptr,
    out_values_ptr,
    table_keys_ptr,
    table_values_ptr,
    table_size,
    N,
    BLOCK_SIZE: tl.constexpr,
    max_probe_step: tl.constexpr = get_h_table_max_p(),
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    idx = block_start + tl.arange(0, BLOCK_SIZE)
    mask = idx < N

    b = tl.load(coords_ptr + idx * 4 + 0, mask=mask)
    x = tl.load(coords_ptr + idx * 4 + 1, mask=mask)
    y = tl.load(coords_ptr + idx * 4 + 2, mask=mask)
    z = tl.load(coords_ptr + idx * 4 + 3, mask=mask)

    hash = hash_coords_kernel(
        b, x, y, z
    ) % table_size
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
        max_probe_step= max_probe_step
    )
    store_mask = (query_out != -1) & mask
    tl.store(out_values_ptr + idx, query_out, mask=store_mask)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def coalesce_coords_kernel(
    coords_ptr, # (N, 4)
    valids_ptr, # (N)
    hash_keys_ptr, # (2 * N)
    table_size, 
    N,
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
    valid_mask = tl.zeros((BLOCK_SIZE,), dtype=tl.int1)
    while (tl.max(active_mask.to(tl.int32), axis=0) > 0) and (step < 32):
        curr_hashs = (hash_vals + step) % table_size
        compare_val = tl.where(active_mask, -1, -2).to(tl.int64)
        old_keys = tl.atomic_cas(hash_keys_ptr + curr_hashs, compare_val, hash_keys)
        duplicated = old_keys == hash_keys
        valid_mask = valid_mask | (duplicated & active_mask)
        success = (old_keys == -1) |  duplicated
        active_mask = active_mask & (~success)
        step += 1

    tl.store(valids_ptr + idx, 0, mask=valid_mask & mask)
        
def coalesce_coords(coords: torch.Tensor):
    coords = coords.contiguous()
    N = coords.shape[0]
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    valids = torch.full((N,), True, device=coords.device, dtype=torch.bool)
    hash_keys = torch.full((N * 2,), -1, device=coords.device, dtype=torch.int64)
    table_size = hash_keys.shape[0]
    coalesce_coords_kernel[grid](
        coords,
        valids,
        hash_keys,
        table_size,
        N
    )
    return valids

class HashTable:
    def __init__(self, capacity: int = None, device: torch.device = "cpu", table_keys: torch.Tensor = None, table_values: torch.Tensor = None):
        assert capacity is not None or (table_keys is not None and table_values is not None), "Either capacity or both table_keys and table_values must be provided."
        
        if table_keys is not None and table_values is not None:
            assert table_keys.shape == table_values.shape, "table_keys and table_values must have the same shape."
            self.table_keys, self.table_values = table_keys, table_values
        else:
            self.table_keys = torch.full((capacity,), -1, dtype=torch.int32, device=device)
            self.table_values = torch.full((capacity,), -1, dtype=torch.int32, device=device)

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

        grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
        # 최소 BLOCK_SIZE가 128이므로 안전하게 N // 128 + 1 크기로 할당
        failure = torch.zeros((triton.cdiv(N, 128),), dtype=torch.int32, device=self.device)

        build_hash_table_kernel[grid](
            keys,           # coords_ptr
            self.table_keys,   # hash_keys_ptr
            self.table_values, # hash_vals_ptr
            self.capacity,     # table_size
            N,                 # N
            failure,
            max_probe_step = get_h_table_max_p()
        )
        failure_n = torch.sum(failure).item()
        if (failure_n > 0):
            print(f"Warning: {failure_n} keys could not be inserted into the hash table due to excessive collisions.")

    def query(self, keys: torch.Tensor) -> torch.Tensor:
        N = keys.shape[0]
        out_values = torch.full((N,), -1, dtype=torch.int32, device=self.device)
        grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
        
        query_hash_table_kernel[grid](
            keys, out_values, self.table_keys, self.table_values, 
            self.capacity, N, max_probe_step = get_h_table_max_p()
        )
        return out_values