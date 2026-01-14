import torch
import triton
import triton.language as tl



@triton.jit
def get_neighbor_map_kernel(
    coords_ptr, hash_keys_ptr, hash_vals_ptr, neighbor_map_ptr,
    N, table_size, 
    kernel_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """
    For each voxel, find indices of its 27 neighbors using the hash table.
    """
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < N

    b = tl.load(coords_ptr + idx * 4 + 0, mask=mask)
    x = tl.load(coords_ptr + idx * 4 + 1, mask=mask)
    y = tl.load(coords_ptr + idx * 4 + 2, mask=mask)
    z = tl.load(coords_ptr + idx * 4 + 3, mask=mask)

    # Iterate over 3x3x3 kernel
    for dk in range(kernel_size ** 3):
        dx = (dk // kernel_size**2) - kernel_size // 2
        dy = ((dk // kernel_size) % kernel_size) - kernel_size // 2
        dz = (dk % kernel_size) - kernel_size // 2

        nx, ny, nz = x + dx, y + dy, z + dz
        n_key = (b.to(tl.int64) << 30) | (nx.to(tl.int64) << 20) | (ny.to(tl.int64) << 10) | nz.to(tl.int64)
        n_hash = _hash_coords(b, nx, ny, nz) % table_size

        # Probe hash table
        found_idx = -1
        for i in range(32): # Max probing steps
            curr_hash = (n_hash + i) % table_size
            k = tl.load(hash_keys_ptr + curr_hash)
            if k == n_key:
                found_idx = tl.load(hash_vals_ptr + curr_hash)
                break
            if k == -1: break

        tl.store(neighbor_map_ptr + idx * kernel_size**3 + dk, found_idx, mask=mask)


@triton.jit
def build_kmap_kernel(
    coords_ptr,
    kmap_ptr,
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

    b = tl.load(coords_ptr + idx * 4 + 0, mask=mask)
    x = tl.load(coords_ptr + idx * 4 + 1, mask=mask)
    y = tl.load(coords_ptr + idx * 4 + 2, mask=mask)
    z = tl.load(coords_ptr + idx * 4 + 3, mask=mask)

    hash = _hash_coords(
        b, x, y, z
    ) % table_size

    probe_step = 0
    active_mask = mask
    key = _flatten_coords(b, x, y, z)
    dummy_ptr = tl.nullptr(tl.int64)

    while (tl.max(active_mask.to(tl.int32), axis=0) > 0) & (probe_step < 32):
        curr_hash = (hash + probe_step) % table_size
        # tl.atomic_cas는 마스크를 직접 지원하지 않으므로 
        target_ptr = tl.where(active_mask, table_keys_ptr + curr_hash, dummy_ptr)
        old_key = tl.atomic_cas(target_ptr, tl.full((BLOCK_SIZE,), -1, dtype=tl.int64), key)

        # 삽입 성공 조건: 빈 자리(-1)였거나, 이미 내 키가 들어있거나
        success = (old_key == -1) | (old_key == key)

        # 이번 루프에서 성공했고, 아직 처리 중이었던 스레드만 store
        write_mask = active_mask & success
        tl.store(table_keys_ptr + curr_hash, key, mask=write_mask)
        tl.store(table_values_ptr + curr_hash, idx, mask=write_mask)
        tl.store(kmap_ptr + idx, curr_hash, mask=write_mask)

        active_mask = active_mask & (~success)
        probe_step += 1