"""Benchmark: spconv vs SparseTriton sparse convolution performance."""

import time
import torch
import sys
sys.path.insert(0, "/home/ptj0225/.openclaw/workspace/SparseTriton")

from sparsetriton import SparseTensor
from sparsetriton.nn.modules import SubMConv3D
from sparsetriton.config import set_conv_algo, ConvAlgo

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Parameters
BATCH_SIZE = 2
SPATIAL_SIZE = (64, 64, 64)  # D, H, W
IN_CHANNELS = 16
OUT_CHANNELS = 32
KERNEL_SIZE = 3
NNZ_LIST = [50000, 100000, 200000, 500000]
WARMUP = 10
ITERATIONS = 100


def create_sparse_data(batch_size, spatial_size, nnz, in_channels, device):
    """Create random sparse tensor data."""
    D, H, W = spatial_size
    
    # Random coordinates
    x = torch.randint(0, W, (nnz,), device=device)
    y = torch.randint(0, H, (nnz,), device=device)
    z = torch.randint(0, D, (nnz,), device=device)
    batch = torch.randint(0, batch_size, (nnz,), device=device)
    
    coords = torch.stack([batch, x, y, z], dim=1).to(torch.int32)
    features = torch.randn(nnz, in_channels, device=device)
    
    return coords, features


def benchmark_spconv(coords, features, spatial_size, in_channels, out_channels, kernel_size, warmup, iterations, batch_size):
    """Benchmark spconv SubMConv3d."""
    import spconv.pytorch as spconv
    
    indices = coords[:, [0, 3, 2, 1]].contiguous()
    sp_tensor = spconv.SparseConvTensor(features, indices, list(spatial_size), batch_size)
    conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False).to(device)
    
    for _ in range(warmup):
        _ = conv(sp_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(iterations):
        _ = conv(sp_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    return (time.perf_counter() - start) / iterations * 1000


def benchmark_sparsetriton(coords, features, spatial_size, in_channels, out_channels, kernel_size, warmup, iterations, batch_size, algo):
    """Benchmark SparseTriton with specified algorithm."""
    set_conv_algo(algo)
    
    sp_tensor = SparseTensor(
        feats=features,
        coords=coords,
        spatial_shape=spatial_size,
        batch_size=batch_size,
    )
    conv = SubMConv3D(in_channels, out_channels, kernel_size, bias=False).to(device)
    
    for _ in range(warmup):
        _ = conv(sp_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(iterations):
        _ = conv(sp_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    return (time.perf_counter() - start) / iterations * 1000


def main():
    print(f"\n{'='*100}")
    print(f"SubMConv3d Benchmark: spconv vs SparseTriton (HashFly vs Precomputed vs Im2Col)")
    print(f"In={IN_CHANNELS}, Out={OUT_CHANNELS}, Kernel={KERNEL_SIZE}")
    print(f"{'='*100}\n")
    
    print(f"{'NNZ':>8} | {'spconv':>10} | {'HashFly':>10} | {'Precomp':>10} | {'Im2Col':>10} | {'HF/sp':>8} | {'PC/sp':>8} | {'I2C/sp':>8}")
    print("-" * 100)
    
    for nnz in NNZ_LIST:
        coords, features = create_sparse_data(BATCH_SIZE, SPATIAL_SIZE, nnz, IN_CHANNELS, device)
        
        # spconv
        try:
            spconv_time = benchmark_spconv(
                coords, features, SPATIAL_SIZE, IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE,
                WARMUP, ITERATIONS, BATCH_SIZE
            )
        except Exception as e:
            print(f"spconv error: {e}")
            spconv_time = float('nan')
        
        # HashFly
        try:
            hashfly_time = benchmark_sparsetriton(
                coords, features, SPATIAL_SIZE, IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE,
                WARMUP, ITERATIONS, BATCH_SIZE, ConvAlgo.ImplicitHashFlyGEMM
            )
        except Exception as e:
            print(f"HashFly error: {e}")
            hashfly_time = float('nan')
        
        # Precomputed
        try:
            precomp_time = benchmark_sparsetriton(
                coords, features, SPATIAL_SIZE, IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE,
                WARMUP, ITERATIONS, BATCH_SIZE, ConvAlgo.PrecomputedNeighborGEMM
            )
        except Exception as e:
            print(f"Precomputed error: {e}")
            precomp_time = float('nan')
        
        # Im2Col
        try:
            im2col_time = benchmark_sparsetriton(
                coords, features, SPATIAL_SIZE, IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE,
                WARMUP, ITERATIONS, BATCH_SIZE, ConvAlgo.Im2ColGEMM
            )
        except Exception as e:
            print(f"Im2Col error: {e}")
            im2col_time = float('nan')
        
        # Speedups
        hf_sp = spconv_time / hashfly_time if not torch.isnan(torch.tensor(hashfly_time)) else float('nan')
        pc_sp = spconv_time / precomp_time if not torch.isnan(torch.tensor(precomp_time)) else float('nan')
        i2c_sp = spconv_time / im2col_time if not torch.isnan(torch.tensor(im2col_time)) else float('nan')
        
        print(f"{nnz:>8} | {spconv_time:>10.3f} | {hashfly_time:>10.3f} | {precomp_time:>10.3f} | {im2col_time:>10.3f} | {hf_sp:>8.2f}x | {pc_sp:>8.2f}x | {i2c_sp:>8.2f}x")


if __name__ == "__main__":
    main()
