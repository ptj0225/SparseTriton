import torch
import time
from sparsetriton.utils.hash import HashTable
from sparsetriton.config import get_h_table_f

def create_random_coords(num_keys, spatial_dim, batch_size, device='cuda'):
    """
    Generates a tensor of unique random 4D coordinates.
    """
    print(f"Generating {num_keys} unique random coordinates...")
    # Use torch operations for better performance with large N
    # This might generate fewer than num_keys if there are many collisions,
    # but it's much faster than iterating.
    
    # Generate more than needed to account for duplicates
    n_generate = int(num_keys * 1.2)
    if n_generate == 0:
        return torch.empty((0, 4), dtype=torch.int32, device=device)
    
    b = torch.randint(0, batch_size, (n_generate, 1), device=device)
    x = torch.randint(0, spatial_dim, (n_generate, 1), device=device)
    y = torch.randint(0, spatial_dim, (n_generate, 1), device=device)
    z = torch.randint(0, spatial_dim, (n_generate, 1), device=device)
    
    coords = torch.cat([b, x, y, z], dim=1)
    unique_coords = torch.unique(coords, dim=0)
    
    # If not enough unique coords, take what we have
    if unique_coords.shape[0] < num_keys:
        print(f"Warning: Could only generate {unique_coords.shape[0]} unique coordinates.")
        return unique_coords.to(torch.int32)

    return unique_coords[:num_keys].to(torch.int32)


def benchmark_hash_table(num_keys, capacity, spatial_dim, batch_size, warmup_runs=3, bench_runs=100):
    """
    Benchmarks the insert and query operations of the HashTable with warm-up and averaging.
    """
    print("-" * 50)
    print(f"Starting benchmark with:")
    print(f"  Number of Keys: {num_keys}")
    print(f"  Table Capacity: {capacity}")
    print(f"  Warm-up Runs: {warmup_runs}")
    print(f"  Benchmark Runs: {bench_runs}")
    print("-" * 50)

    # 1. Create keys
    keys = create_random_coords(num_keys, spatial_dim, batch_size, device='cuda')
    actual_num_keys = keys.shape[0]
    if actual_num_keys == 0:
        print("No keys generated, skipping benchmark.")
        return
        
    print(f"Actual number of keys created: {actual_num_keys}")

    # 2. Warm-up
    print(f"\nPerforming {warmup_runs} warm-up runs...")
    for i in range(warmup_runs):
        # We must re-create the hash table for each insert benchmark
        hash_table = HashTable(capacity=capacity, device='cuda')
        hash_table.insert(keys)
        _ = hash_table.query(keys)
    torch.cuda.synchronize()
    print("Warm-up complete.")

    # 3. Actual Benchmark
    insert_times = []
    query_times = []
    print(f"\nPerforming {bench_runs} benchmark runs...")
    for i in range(bench_runs):
        torch.cuda.synchronize()

        # Benchmark Insert
        start_time = time.perf_counter() 
        hash_table = HashTable(capacity=capacity, device='cuda')
        hash_table.insert(keys)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        insert_times.append((end_time - start_time) * 1000)

        # Benchmark Query
        start_time = time.perf_counter()
        query_results = hash_table.query(keys)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        query_times.append((end_time - start_time) * 1000)

    # 4. Results
    avg_insert_time = sum(insert_times) / len(insert_times)
    avg_query_time = sum(query_times) / len(query_times)
    
    print("\n" + "=" * 50)
    print(f"Benchmark Results (Averaged over {bench_runs} runs)")
    print(f"Average Insert Time: {avg_insert_time:.4f} ms")
    print(f"Average Query Time:  {avg_query_time:.4f} ms")
    print("=" * 50)

    # Optional: Verify correctness of query on the last run's result
    expected_results = torch.arange(actual_num_keys, device='cuda', dtype=torch.int32)
    correct_queries = (query_results == expected_results).sum().item()
    accuracy = (correct_queries / actual_num_keys) * 100
    print(f"\nQuery Accuracy (last run): {accuracy:.2f}% ({correct_queries}/{actual_num_keys} correct)")


if __name__ == '__main__':
    # Configuration
    NUM_KEYS = 100_000
    CAPACITY = int(NUM_KEYS * get_h_table_f())
    SPATIAL_DIM = 369
    BATCH_SIZE = 16

    benchmark_hash_table(
        num_keys=NUM_KEYS,
        capacity=CAPACITY,
        spatial_dim=SPATIAL_DIM,
        batch_size=BATCH_SIZE
    )
