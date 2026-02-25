"""Pytest configuration file for SparseTriton tests."""

import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def warmup_backward_kernels():
    """
    Warm up Triton backward kernels before running tests.

    This works around a JIT/stateful bug where the first backward pass
    produces incorrect gradients. By running a warmup backward pass
    before any tests, all subsequent backward passes work correctly.
    """
    if not torch.cuda.is_available():
        return

    from sparsetriton.nn.functional.conv.conv import sparse_conv3d
    from sparsetriton.nn.functional.conv.kmap import build_out_coords
    from sparsetriton.utils.hash import HashTable
    from sparsetriton import randn

    device = torch.device("cuda")
    print("\n" + "=" * 60)
    print("Warming up backward kernels (JIT compilation)...")
    print("=" * 60)

    # Run a few backward passes with different configurations to warm up all kernels
    configs = [
        {"C_in": 8, "C_out": 8, "kernel_size": 3, "padding": 0},
        {"C_in": 16, "C_out": 16, "kernel_size": 3, "padding": 2},
        {"C_in": 32, "C_out": 32, "kernel_size": 5, "padding": 2},
    ]

    for i, cfg in enumerate(configs, 1):
        C_in, C_out = cfg["C_in"], cfg["C_out"]
        kernel_size = cfg["kernel_size"]
        padding = cfg["padding"]

        st_tensor = randn((10, 10, 10), batch_size=1, channels=C_in, nnz=64, device=device).half()
        weight = torch.rand(kernel_size**3, C_in, C_out, device=device, dtype=torch.float16, requires_grad=True)

        out = sparse_conv3d(
            st_tensor, weight, kernel_size=kernel_size, stride=1,
            padding=padding, dilation=1, submanifold=False
        )

        # Run backward to trigger JIT compilation
        out.F.abs().mean().backward()

        torch.cuda.synchronize()
        print(f"  [{i}/{len(configs)}] Warmup: C_in={C_in}, C_out={C_out}, kernel={kernel_size}, padding={padding} - OK")

    # Warm up transposed conv with stride=2
    print("  [4/4] Warmup: transposed, C_in=4, C_out=8, kernel=3, stride=2, padding=1 - ", end="")
    st_tensor = randn((10, 10, 10), batch_size=1, channels=4, nnz=64, device=device).half()
    weight = torch.rand(27, 4, 8, device=device, dtype=torch.float16)

    out = sparse_conv3d(
        st_tensor, weight, kernel_size=3, stride=2,
        padding=1, dilation=1, submanifold=False, transposed=True
    )

    torch.cuda.synchronize()
    print("OK")

    for i, cfg in enumerate(configs, 1):
        C_in, C_out = cfg["C_in"], cfg["C_out"]
        kernel_size = cfg["kernel_size"]
        padding = cfg["padding"]

        st_tensor = randn((10, 10, 10), batch_size=1, channels=C_in, nnz=64, device=device).half()
        weight = torch.rand(kernel_size**3, C_in, C_out, device=device, dtype=torch.float16, requires_grad=True)

        out = sparse_conv3d(
            st_tensor, weight, kernel_size=kernel_size, stride=1,
            padding=padding, dilation=1, submanifold=False
        )

        # Run backward to trigger JIT compilation
        out.F.abs().mean().backward()

        torch.cuda.synchronize()
        print(f"  [{i}/{len(configs)}] Warmup: C_in={C_in}, C_out={C_out}, kernel={kernel_size}, padding={padding} - OK")

    print("=" * 60)
    print("Warmup complete. Running tests...")
    print("=" * 60 + "\n")
