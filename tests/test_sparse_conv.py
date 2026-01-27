import torch
import torch.nn.functional as F
import pytest
from sparsetriton.tensor import SparseTensor, randn
from sparsetriton.nn.functional import sparse_conv3d

import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# @pytest.mark.parametrize("C_in, C_out, kernel_size, stride, padding, dilation", [
#     (8, 16, 3, 1, 1, 1),
#     (4, 8, 3, 2, 1, 1),
#     (16, 16, 5, 1, 2, 1),
# ])
# def test_sparse_conv3d_forward(C_in, C_out, kernel_size, stride, padding, dilation):
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     # 1. Create input sparse tensor
#     spatial_shape = (10, 10, 10)
#     st_tensor = randn(spatial_shape, batch_size=1, channels=C_in, nnz=27, device=device).half()

#     # 2. Create convolution weight (K, C_in, C_out)
#     weight = torch.rand(kernel_size**3, C_in, C_out, device=device, dtype=torch.float16, requires_grad=True)

#     # 3. Run sparsetriton convolution (submanifold=False)
#     st_out_tensor = sparse_conv3d(
#         st_tensor,
#         weight,
#         kernel_size=kernel_size,
#         stride=stride,
#         padding=padding,
#         dilation=dilation,
#         submanifold=False,
#         transposed=False
#     ).float()
    
#     # 4. Run torch dense convolution
#     # Weight: (K, C_in, C_out) -> (C_out, C_in, k, k, k)
#     k = kernel_size
#     weight_torch = weight.view(k, k, k, C_in, C_out).permute(4, 3, 0, 1, 2).contiguous().float()
    
#     # Input: Sparse -> Dense (N, D, H, W, C) -> (N, C, D, H, W)
#     dense_input = st_tensor.dense().permute(0, 4, 1, 2, 3).contiguous().float()
    
#     dense_output = F.conv3d(
#         dense_input,
#         weight_torch,
#         stride=stride,
#         padding=padding,
#         dilation=dilation
#     )

#     # 5. Compare dense results
#     st_dense_output = st_out_tensor.dense()
#     torch_dense_output = dense_output.permute(0, 2, 3, 4, 1).contiguous()
    
#     assert st_dense_output.shape == torch_dense_output.shape, \
#         f"Shape mismatch: {st_dense_output.shape} vs {torch_dense_output.shape}"
        
#     assert torch.allclose(st_dense_output, torch_dense_output, atol=1e-3, rtol=1e-3), \
#         f"Feature values mismatch. Max diff: {(st_dense_output - torch_dense_output).abs().max()}"


# @pytest.mark.parametrize("C_in, C_out, kernel_size, stride, padding, dilation", [
#     (8, 16, 3, 1, 1, 1),
#     (4, 8, 3, 2, 1, 1),
#     (16, 16, 5, 1, 2, 1),
# ])
# def test_sparse_conv_transpose3d_forward(C_in, C_out, kernel_size, stride, padding, dilation):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     spatial_shape = (10, 10, 10)
#     st_tensor = randn(spatial_shape, batch_size=1, channels=C_in, nnz=27, device=device).half()

#     # 2. Create convolution weight (K, C_in, C_out)
#     weight = torch.rand(kernel_size**3, C_in, C_out, device=device, dtype=torch.float16, requires_grad=True)

#     # 3. Run sparsetriton transposed convolution (submanifold=False)
#     st_out_tensor = sparse_conv3d(
#         st_tensor,
#         weight,
#         kernel_size=kernel_size,
#         stride=stride,
#         padding=padding,
#         dilation=dilation,
#         submanifold=False,
#         transposed=True
#     ).float()
    
#     # 4. Run torch dense transposed convolution
#     # Weight: (K, C_in, C_out) -> (C_in, C_out, k, k, k)
#     k = kernel_size
#     weight_torch = weight.view(k, k, k, C_in, C_out).permute(3, 4, 0, 1, 2).contiguous().float()
    
#     # Input: Sparse -> Dense (N, D, H, W, C) -> (N, C, D, H, W)
#     dense_input = st_tensor.dense().permute(0, 4, 1, 2, 3).contiguous().float()
    
#     dense_output = F.conv_transpose3d(
#         dense_input,
#         weight_torch,
#         stride=stride,
#         padding=padding,
#         dilation=dilation
#     )

#     # 5. Compare dense results
#     st_dense_output = st_out_tensor.dense()
#     torch_dense_output = dense_output.permute(0, 2, 3, 4, 1).contiguous()
    
#     assert st_dense_output.shape == torch_dense_output.shape, \
#         f"Shape mismatch: {st_dense_output.shape} vs {torch_dense_output.shape}"
        
#     assert torch.allclose(st_dense_output, torch_dense_output, atol=1e-3, rtol=1e-3), \
#         f"Feature values mismatch. Max diff: {(st_dense_output - torch_dense_output).abs().max()}"

# @pytest.mark.parametrize("C_in, C_out, kernel_size, stride, padding, dilation", [
#     (16, 16, 3, 1, 0, 1),
#     (64, 64, 3, 1, 1, 1),
#     (512, 512, 5, 1, 1, 1)
# ])
# def test_sparse_conv3d_backward(C_in, C_out, kernel_size, stride, padding, dilation):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     spatial_shape = (64, 64, 64)# 빠른 테스트를 위해 크기 축소
    
#     # 1. Input 준비 (Gradients 체크를 위해 float32 권장)
#     # Sparse Input
#     st_input = randn(spatial_shape, batch_size=1, channels=C_in, nnz=32**3, device=device)
#     st_input.F = st_input.F.half()
#     st_input.F.requires_grad_(True)
    
#     # Weight
#     weight = torch.randn(kernel_size**3, C_in, C_out, device=device, dtype=torch.float16)
#     weight.requires_grad_(True)

#     st_out = sparse_conv3d(
#         st_input, weight, kernel_size=kernel_size, stride=stride, 
#         padding=padding, dilation=dilation, submanifold=False
#     )
#     st_out_dense = st_out.dense()
#     st_out_dense.abs().sum().backward()
#     spconv_w_grad = weight.grad.clone()
#     spconv_f_grad = st_input.F.grad.clone()

#     weight.grad = None 
#     st_input.F.grad = None
#     k = kernel_size
#     weight_torch = weight.view(k, k, k, C_in, C_out).permute(4, 3, 0, 1, 2).contiguous()
#     dense_input = st_input.dense().permute(0, 4, 1, 2, 3).contiguous()
#     dense_input.requires_grad_(True)
#     dense_out = F.conv3d(
#         dense_input, weight_torch, stride=stride, padding=padding, dilation=dilation
#     )

#     dense_out = dense_out.permute(0, 2, 3, 4, 1)
#     dense_out.abs().sum().backward()

#     tconv_w_grad = weight.grad.clone()
#     tconv_f_grad = st_input.F.grad.clone()
#     torch.testing.assert_close(st_out_dense, dense_out, atol=1e-3, rtol=1e-3)
#     torch.testing.assert_close(tconv_w_grad, spconv_w_grad, atol=1e-3, rtol=1e-3)
#     torch.testing.assert_close(spconv_f_grad, tconv_f_grad, atol=1e-3, rtol=1e-3)
#     torch.testing.assert_close(tconv_w_grad, spconv_w_grad, atol=1e-3, rtol=1e-3)
#     torch.testing.assert_close(spconv_f_grad, tconv_f_grad, atol=1e-3, rtol=1e-3)
#     torch.testing.assert_close(tconv_w_grad, spconv_w_grad, atol=1e-3, rtol=1e-3)
#     torch.testing.assert_close(spconv_f_grad, tconv_f_grad, atol=1e-3, rtol=1e-3)