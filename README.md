# SparseTriton

A high-performance, **hardware-agnostic** 3D Sparse Convolution library implemented purely in **Triton**. Provides a `torch.nn` compatible interface that runs on any device supported by the Triton compiler (NVIDIA, AMD ROCm), eliminating the dependency on proprietary CUDA/C++ extensions.

## Features

- **Submanifold Sparse Convolution** — Preserves input sparsity patterns for deep architectures
- **Standard Sparse Convolution** — Supports stride, padding, and dilation for downsampling
- **Transposed Sparse Convolution** — Upsampling with stride support
- **GPU Hash Table** — Parallel hash table with Triton kernels for coordinate lookup
- **Autograd Support** — Full backward pass for end-to-end training
- **torch.nn Compatible Modules** — Drop-in replacements: `SparseConv3D`, `SubMConv3D`, `SparseBatchNorm`, etc.

## Installation

```bash
pip install -e .
```

Requires Python >= 3.9, PyTorch >= 2.1, and Triton >= 2.1.

## Quick Start

```python
import torch
from sparsetriton import SparseTensor, randn
from sparsetriton.nn.modules import SubMConv3D, SparseConv3D, SparseBatchNorm, ReLU

# Create a random sparse tensor
sp = randn(spatial_shape=(64, 64, 64), batch_size=2, nnz=4096, channels=16, device="cuda")

# Build a sparse conv network
conv1 = SubMConv3D(16, 32, kernel_size=3).cuda()
conv2 = SparseConv3D(32, 64, kernel_size=3, stride=2).cuda()
bn = SparseBatchNorm(32).cuda()
relu = ReLU()

# Forward pass
x = relu(bn(conv1(sp)))
out = conv2(x)
```

## Architecture

Two convolution algorithms are available:

| Algorithm | Description | Use Case |
|---|---|---|
| `PrecomputedNeighborGEMM` | Pre-computes neighbor indices, then runs fused GEMM kernel | Default, best performance |
| `ImplicitHashFlyGEMM` | On-the-fly hash lookup inside the GEMM kernel | Lower memory, no precomputation |

```python
from sparsetriton.config import set_conv_algo, ConvAlgo

set_conv_algo(ConvAlgo.PrecomputedNeighborGEMM)  # default
set_conv_algo(ConvAlgo.ImplicitHashFlyGEMM)
```

## Modules

| Module | Description |
|---|---|
| `SparseConv3D` | Standard sparse 3D convolution (stride, padding, dilation) |
| `SubMConv3D` | Submanifold sparse convolution (preserves sparsity) |
| `SparseConvTransposed3D` | Transposed sparse convolution |
| `SparseLinear` | Linear layer for sparse features |
| `SparseBatchNorm` | Batch normalization with per-batch statistics |
| `SparseLayerNorm` | Layer normalization |
| `ReLU`, `LeakyReLU`, `SiLU`, `GELU`, `Sigmoid`, `Tanh` | Activations |
| `SparsePooling` | Max / Average pooling |
| `SparseUpsample` | Nearest-neighbor upsampling |
| `SparseDownsample` | Downsampling via pooling |

## Testing

```bash
pytest tests/ -v
```

## Known Limitations

- **Max pooling on CPU**: Triton atomic operations are not supported on CPU
- **Coordinate range**: Coordinates should fit within `int16` range (recommended). The config can be changed via `set_coords_dtype(torch.int32)`
- **Triton TF32 precision**: `tl.dot` internally uses TF32, resulting in ~0.001 relative error in forward/backward compared to dense PyTorch convolutions

## License

MIT
