from typing import Literal
import torch
import pytest

# sparsetriton imports
from sparsetriton.tensor import SparseTensor, randn
from sparsetriton.nn.functional import sparse_conv3d
from sparsetriton.utils.to_dense import ToDenseFunction

# torchsparse imports
try:
    import torchsparse.nn.functional as F_st
    from torchsparse import SparseTensor as SparseTensor_st
    TORCHSPARSE_AVAILABLE = True
except ImportError:
    TORCHSPARSE_AVAILABLE = False

# Helper function for lexicographical sort
def sort_sparse_tensor(coords, feats):
    if coords.shape[0] == 0:
        return coords, feats
    
    # Argsort on the last column first, then propagate
    # Assuming coords are (batch_idx, x, y, z)
    order = torch.arange(coords.shape[0], device=coords.device)
    for i in range(coords.shape[1] - 1, -1, -1): # Iterate from last column (z) to first (batch_idx)
        order = order[coords[order, i].argsort(stable=True)]
    
    return coords[order], feats[order]


@pytest.mark.skipif(not TORCHSPARSE_AVAILABLE, reason="torchsparse is not installed")
@pytest.mark.parametrize("C_in, C_out, kernel_size, stride, padding, dilation", [
    (512, 512, 3, 1, 3, 1), # Basic case, stride 1, padding 0
    (4, 8, 5, 1, 0, 1),   # 1x1 kernel, stride 1, padding 0
    (8, 8, 5, 1, 1, 1),   # Basic case, stride 1, padding 0
])
def test_sparse_conv3d_vs_torchsparse(C_in: Literal[8], C_out: Literal[16], kernel_size: Literal[3], stride: Literal[1], padding: Literal[0], dilation: Literal[1]):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Create input sparse tensor for sparsetriton
    spatial_shape = (3, 3, 3)
    batch_size = 2
    nnz = 10
    feats = torch.rand(nnz, C_in, device=device)
    
    # Create non-overlapping coordinates for each batch
    coords_b0 = torch.cat([
        torch.zeros(nnz // 2, 1, device=device, dtype=torch.int),
        torch.randint(0, spatial_shape[0], (nnz // 2, 1), device=device),
        torch.randint(0, spatial_shape[1], (nnz // 2, 1), device=device),
        torch.randint(0, spatial_shape[2], (nnz // 2, 1), device=device),
    ], dim=1)
    coords_b1 = torch.cat([
        torch.ones(nnz - nnz // 2, 1, device=device, dtype=torch.int),
        torch.randint(0, spatial_shape[0], (nnz - nnz // 2, 1), device=device),
        torch.randint(0, spatial_shape[1], (nnz - nnz // 2, 1), device=device),
        torch.randint(0, spatial_shape[2], (nnz - nnz // 2, 1), device=device),
    ], dim=1)
    coords = torch.cat([coords_b0, coords_b1], dim=0).to(torch.int32)
    
    st_tensor = randn((10, 10, 10), 2, C_in, 10 ** 3 // 2, device)

    # 2. Create convolution weight
    weight = torch.rand(kernel_size * kernel_size * kernel_size, C_in, C_out, device=device)

    # 3. Run sparsetriton convolution
    st_out_tensor = sparse_conv3d(
        st_tensor,
        weight,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    
    # 4. Create input sparse tensor for torchsparse
    # torchsparse expects coords as (x, y, z, batch_idx)
    # torchsparse.SparseTensor constructor takes spatial_size, not spatial_shape (tuple)
    # spatial_size must be a list of int
    stt_tensor = SparseTensor_st(st_tensor.F, st_tensor.C.to(torch.int32), list(st_tensor.spatial_shape)) 

    # 5. Run torchsparse convolution
    # The weight format for torchsparse is (Kx, Ky, Kz, Cin, Cout)

    stt_out_tensor = F_st.conv3d(
        stt_tensor,
        weight,
        kernel_size=kernel_size,
        stride=stride,
        # padding=padding, # torchsparse conv3d does not have padding arg in this functional
        dilation=dilation * 2,
    )
    
    # --- Compare sparse outputs directly by sorting ---
    # sparsetriton output
    st_out_coords = st_out_tensor.C
    st_out_feats = st_out_tensor.F

    # torchsparse output
    # Convert coords from (x, y, z, batch_idx) to (batch_idx, x, y, z) for comparison
    stt_out_coords = stt_out_tensor.C
    stt_out_feats = stt_out_tensor.F

    # Handle cases where output might be empty
    if st_out_coords.shape[0] == 0 and stt_out_coords.shape[0] == 0:
        return # Both empty, test passes

    if st_out_coords.shape[0] != stt_out_coords.shape[0]:
        print(f"\n--- Coordinate Shape Mismatch for C_in={C_in}, C_out={C_out}, kernel={kernel_size}, stride={stride}, padding={padding}, dilation={dilation} ---")
        print(f"sparsetriton output shape: {st_out_coords.shape[0]}")
        print(f"torchsparse output shape: {stt_out_coords.shape[0]}")
        print(f"sparsetriton output coords (first 5):\n{st_out_coords[:5]}")
        print(f"torchsparse output coords (first 5):\n{stt_out_coords[:5]}")
        
    assert st_out_coords.shape[0] == stt_out_coords.shape[0], \
        f"Number of non-zero elements mismatch: sparsetriton {st_out_coords.shape[0]}, torchsparse {stt_out_coords.shape[0]}"

    # Lexicographical sort both sparse tensors
    # sparsetriton
    sorted_st_out_coords, sorted_st_out_feats = sort_sparse_tensor(st_out_coords, st_out_feats)

    # torchsparse
    sorted_stt_out_coords, sorted_stt_out_feats = sort_sparse_tensor(stt_out_coords, stt_out_feats)
    # Compare sorted coordinates
    assert torch.equal(sorted_st_out_coords, sorted_stt_out_coords), \
        f"Output coordinates mismatch:\nsparsetriton:\n{sorted_st_out_coords}\ntorchsparse:\n{sorted_stt_out_coords}"
    diff = (sorted_st_out_feats - sorted_stt_out_feats).abs()
    max_diff_idx = diff.max(dim=1)[0].argmax()
    print(f"Max diff at coordinate: {sorted_st_out_coords[max_diff_idx]}")
    # Compare sorted features
    assert torch.allclose(sorted_stt_out_feats, sorted_st_out_feats, atol=1e-3, rtol=1e-3), \
        f"Output features mismatch:\nsparsetriton:\n{sorted_st_out_feats.shape}\n{sorted_st_out_feats[: 10]}\ntorchsparse:\n{sorted_stt_out_feats.shape}\n{sorted_stt_out_feats[:10]}\nError max diff: {(sorted_st_out_feats - sorted_stt_out_feats).abs().max()} \
        \n {sorted_st_out_coords}"