# Triton-based 3D Sparse Convolution

A high-performance, **hardware-agnostic** 3D Sparse Convolution library implemented purely in **Triton**. This project aims to provide a seamless `torch.nn` compatible interface that runs on any device supported by the Triton compiler (NVIDIA, AMD, etc.), eliminating the dependency on proprietary CUDA/C++ extensions.

## 🌟 Key Objectives
* **Vendor Agnostic**: Zero CUDA C++ code. Fully compatible with NVIDIA and AMD (ROCm) via Triton.
* **Memory Efficiency**: Utilizes sparse data structures to handle large-scale 3D point clouds or medical volumes.
* **Performance**: Highly optimized kernels for Gather-GEMM-Scatter operations.

---

## 🚧 Work in Progress
This project is currently under active development. While core functionalities are implemented, some features may be unstable, undocumented, or subject to change. Performance optimizations and comprehensive testing are ongoing. Your contributions and feedback are highly welcome!

---

## 🚀 Features
* **Submanifold Sparse Convolution**: Preserves input sparsity patterns for deep architectures.
* **Standard Sparse Convolution**: Supports stride and padding for downsampling.
* **GPU-based Rule Generation**: Fast index mapping and rulebook generation using Triton-based hashing.
* **Autograd Support**: Full backward pass implementation for end-to-end training.

---

## 🛠 Architecture

1.  **Coordinate Hashing**: Map 3D coordinates to linear indices using a parallel hash table.
2.  **Rulebook Generation**: Identify active neighbor pairs for each kernel offset.
3.  **Gather-GEMM-Scatter**:
    * **Gather**: Collect features based on the rulebook.
    * **GEMM**: Perform matrix multiplication using Triton's fused kernels.
    * **Scatter**: Distribute results back to the output sparse tensor.

---

## 📋 TODO

### Known Issues
- [ ] Backward pass weight gradient incorrect (99.5% mismatch) - `DEBUG_SUMMARY.md`
- [ ] Transposed convolution forward fails for stride=2 cases
- [ ] Pooling operations fail on CPU (atomic operations not supported)

### CPU Support
- [ ] CPU implementation or fallback (many tests skipped on CPU)

### Performance Optimization
- [ ] Autotuner config tuning (C_in=1, 64 show issues)
- [ ] Memory efficiency for large-scale point clouds

### Features
- [ ] More activation functions
- [ ] Batch normalization for CPU
- [ ] Documentation & API reference
