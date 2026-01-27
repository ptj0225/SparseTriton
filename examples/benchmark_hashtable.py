import torch
import triton
import time
# Import HashTable class
from sparsetriton.utils.hash import HashTable
from sparsetriton.config import set_h_table_f, set_h_table_max_p, get_h_table_max_p
 
def benchmark_hashtable_class():
    if not torch.cuda.is_available():
        return

    device = torch.device("cuda")
    N = 2_000_0000
    num_repeats = 10
    print(f"\nBenchmarking HashTable class insert with N={N}, repeats={num_repeats}...")
    
    coords = torch.randint(0, 20000, (N, 4), device=device, dtype=torch.int16)
    coords[:, 0] = 0
    
    factors = [1.05, 1.1, 1.2, 1.5, 2.0, 3.0, 4.0, 5.0, 8.0]
    
    print(f"{'Capacity Factor':<20} | {'Table Size':<15} | {'Avg Time (ms)':<15}")
    print("-" * 60)
    set_h_table_max_p(4096)
    for factor in factors:
        table_size = int(N * factor)
        total_time_ms = 0.0
        set_h_table_f(factor)

        for _ in range(10):
            ht = HashTable(table_size, device=device)
            ht.insert(coords)

        for _ in range(num_repeats):
            # Initialize HashTable
            ht = HashTable(table_size, device=device)
            
            # Timing
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            torch.cuda.synchronize()
            start_event.record()
            ht.insert(coords)
            end_event.record()
            torch.cuda.synchronize()
            total_time_ms += start_event.elapsed_time(end_event)
            
        avg_time_ms = total_time_ms / num_repeats
        print(f"{factor:<20} | {table_size:<15} | {avg_time_ms:<15.4f}")

if __name__ == "__main__":
    print("=== Benchmark: HashTable Class ===")
    try:
        benchmark_hashtable_class()
    except Exception as e:
        print(f"Skipping HashTable class benchmark: {e}")