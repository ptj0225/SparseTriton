# Triton-based 3D Sparse Convolution

A high-performance, **hardware-agnostic** 3D Sparse Convolution library implemented purely in **Triton**. This project aims to provide a seamless `torch.nn` compatible interface that runs on any device supported by the Triton compiler (NVIDIA, AMD, etc.), eliminating the dependency on proprietary CUDA/C++ extensions.

## 🌟 Key Objectives
* **Vendor Agnostic**: Zero CUDA C++ code. Fully compatible with NVIDIA and AMD (ROCm) via Triton.
* **Memory Efficiency**: Utilizes sparse data structures to handle large-scale 3D point clouds or medical volumes.
* **Performance**: Highly optimized kernels for Gather-GEMM-Scatter operations.

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

## 📦 Installation

```bash
# Requirements: Python 3.9+, PyTorch 2.1+, Triton 2.1+
git clone [https://github.com/your-repo/triton-3d-sparse-conv.git](https://github.com/your-repo/triton-3d-sparse-conv.git)
cd triton-3d-sparse-conv
pip install -e .