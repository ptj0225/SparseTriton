import pytest
import torch


def _warmup_backward():
    if not torch.cuda.is_available():
        return

    from sparsetriton.nn.functional.conv.conv import sparse_conv3d
    from sparsetriton import randn

    device = torch.device("cuda")

    configs = [
        {"C_in": 8, "C_out": 8, "kernel_size": 3, "padding": 0},
        {"C_in": 16, "C_out": 16, "kernel_size": 3, "padding": 2},
        {"C_in": 32, "C_out": 32, "kernel_size": 5, "padding": 2},
    ]

    for cfg in configs:
        C_in, C_out = cfg["C_in"], cfg["C_out"]
        ks, pad = cfg["kernel_size"], cfg["padding"]
        st = randn((10, 10, 10), batch_size=1, channels=C_in, nnz=64, device=device).half()
        w = torch.rand(ks**3, C_in, C_out, device=device, dtype=torch.float16, requires_grad=True)
        out = sparse_conv3d(st, w, kernel_size=ks, stride=1, padding=pad, dilation=1, submanifold=False)
        out.F.abs().mean().backward()
        torch.cuda.synchronize()

    st = randn((10, 10, 10), batch_size=1, channels=4, nnz=64, device=device).half()
    w = torch.rand(27, 4, 8, device=device, dtype=torch.float16)
    out = sparse_conv3d(st, w, kernel_size=3, stride=2, padding=1, dilation=1, submanifold=False, transposed=True)
    torch.cuda.synchronize()


@pytest.fixture(scope="session", autouse=True)
def warmup_backward_kernels():
    _warmup_backward()
