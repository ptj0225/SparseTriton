import torch
import torch.nn.functional as F
import pytest
from sparsetriton.tensor import SparseTensor, randn
from sparsetriton.nn.functional import sparse_conv3d

@pytest.mark.parametrize("C_in, C_out, kernel_size, stride, padding, dilation", [
    (8, 16, 3, 1, 1, 1),
    (4, 8, 3, 2, 1, 1),
    (16, 16, 5, 1, 2, 1),
])
def test_sparse_conv3d_vs_torch_dense(C_in, C_out, kernel_size, stride, padding, dilation):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Create input sparse tensor
    spatial_shape = (3, 3, 3)
    st_tensor = randn(spatial_shape, batch_size=1, channels=C_in, nnz=27, device=device)

    # st_tensor.F = torch.ones_like(st_tensor.F)

    # 2. Create convolution weight (K, C_in, C_out)
    weight = torch.rand(kernel_size**3, C_in, C_out, device=device)
    weight.requires_grad = True
    

    # 3. Run sparsetriton convolution (submanifold=False for full comparison)
    st_out_tensor = sparse_conv3d(
        st_tensor.half(),
        weight.half(),
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        submanifold=False,
        transposed=True
    ).float()

    st_out_tensor.dense().sum().backward()
    grad1 = weight.grad.clone()
    print(grad1[:1])

    weight.grad = None
    st_out_tensor = sparse_conv3d(
        st_tensor.half(),
        weight.half(),
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        submanifold=False,
        transposed=True
    ).float()
    st_out_tensor.F.sum().backward()
    grad1 = weight.grad.clone()
    print(grad1[:1])
    weight.grad = None

    # 4. Run torch dense convolution
    # Weight: (K, C_in, C_out) -> (C_out, C_in, k, k, k)
    k = kernel_size
    weight_torch = weight.view(k, k, k, C_in, C_out).permute(3, 4, 0, 1, 2).contiguous()
    # weight_torch = weight.view(k, k, k, C_in, C_out).permute(4, 3, 1, 0, 2).contiguous()
    # weight_torch = weight.view(k, k, k, C_in, C_out).permute(4, 3, 0, 2, 1).contiguous()
    # weight_torch = weight.view(k, k, k, C_in, C_out).permute(4, 3, 2, 1, 0).contiguous()
    # weight_torch = weight.view(k, k, k, C_in, C_out).permute(4, 3, 2, 0, 1).contiguous()
    # weight_torch = weight.view(k, k, k, C_in, C_out).permute(4, 3, 1, 2, 0).contiguous()
    
    # Input: Sparse -> Dense (N, D, H, W, C) -> (N, C, D, H, W)
    dense_input = st_tensor.dense().permute(0, 4, 1, 2, 3).contiguous()
    
    dense_output = F.conv_transpose3d(
        dense_input.half(),
        weight_torch.half(),
        stride=stride,
        padding=padding,
        dilation=dilation
    ).float()
    dense_output.sum().backward()
    grad2 = weight.grad.clone()
    # 5. Compare dense results
    # st_out_tensor.dense() is (N, D_out, H_out, W_out, C_out)
    # dense_output is (N, C_out, D_out, H_out, W_out)
    st_dense_output = st_out_tensor.dense()
    torch_dense_output = dense_output.permute(0, 2, 3, 4, 1).contiguous()
    # print(st_dense_output.nonzero())
    # print(torch_dense_output.nonzero())
    # print(st_dense_output[..., -1])
    # print(torch_dense_output[..., -1])
    print((st_dense_output - torch_dense_output).abs().max())
    assert st_dense_output.shape == torch_dense_output.shape, \
        f"Shape mismatch: {st_dense_output.shape} vs {torch_dense_output.shape}"
        
    assert torch.allclose(st_dense_output, torch_dense_output, atol=1e-3, rtol=1e-3), \
        f"Feature values mismatch. Max diff: {(st_dense_output - torch_dense_output).abs().max()}"
    
test_sparse_conv3d_vs_torch_dense(* (16, 16, 3, 2, 1, 1))
