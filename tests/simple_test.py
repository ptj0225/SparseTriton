import torch
import pytest
from typing import Literal

# sparsetriton imports
from sparsetriton.tensor import SparseTensor
from sparsetriton.nn.functional import sparse_conv3d

# torchsparse imports
try:
    import torchsparse.nn.functional as F_st
    from torchsparse import SparseTensor as SparseTensor_st
    TORCHSPARSE_AVAILABLE = True
except ImportError:
    TORCHSPARSE_AVAILABLE = False

def sort_sparse_tensor(coords, feats):
    if coords.shape[0] == 0:
        return coords, feats
    order = torch.arange(coords.shape[0], device=coords.device)
    for i in range(coords.shape[1] - 1, -1, -1):
        order = order[coords[order, i].argsort(stable=True)]
    return coords[order], feats[order]

@pytest.mark.skipif(not TORCHSPARSE_AVAILABLE, reason="torchsparse is not installed")
def test_simple_conv3d():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    C_in, C_out, kernel_size, stride, padding, dilation = 2, 2, 3, 1, 1, 1

    # Single input point at the center of a 5x5x5 space
    spatial_shape = (5, 5, 5)
    batch_size = 1
    feats = torch.ones(1, C_in, device=device)
    coords = torch.tensor([[0, 2, 2, 2]], dtype=torch.int32, device=device)
    
    st_tensor = SparseTensor(feats, coords, spatial_shape, batch_size)

    # Weight tensor: an identity-like kernel where each output channel is a copy of an input channel
    weight = torch.rand(kernel_size**3, C_in, C_out, device=device)
    # At the center of the kernel (k=13 for 3x3x3), set a simple weight
    # This should copy input channel 0 to output channel 0, and input 1 to output 1
    center_k_idx = 13 # (3*3*3) // 2
    weight[center_k_idx, 0, 0] = 1.0
    weight[center_k_idx, 1, 1] = 1.0

    # --- sparsetriton convolution ---
    # With submanifold=False and padding=1, a single input point should produce kernel_size**3 output points
    st_out_tensor = sparse_conv3d(
        st_tensor,
        weight,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        submanifold=True,
    )
    
    # --- torchsparse convolution ---
    coords_stt_input = coords[:, [1, 2, 3, 0]]
    stt_tensor = SparseTensor_st(feats, coords_stt_input, list(st_tensor.spatial_shape))
    weight_stt = weight.reshape(kernel_size, kernel_size, kernel_size, C_in, C_out)
    
    stt_out_tensor = F_st.conv3d(
        stt_tensor,
        weight_stt,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    
    st_out_coords, st_out_feats = sort_sparse_tensor(st_out_tensor.C, st_out_tensor.F)
    
    stt_out_coords_raw = stt_out_tensor.C
    stt_out_feats_raw = stt_out_tensor.F
    stt_out_coords_swapped = stt_out_coords_raw[:, [3, 0, 1, 2]]
    stt_out_coords, stt_out_feats = sort_sparse_tensor(stt_out_coords_swapped, stt_out_feats_raw)

    print("\n--- sparsetriton output ---")
    print("Coords:\n", st_out_coords)
    print("Features:\n", st_out_feats)

    print("\n--- torchsparse output ---")
    print("Coords:\n", stt_out_coords)
    print("Features:\n", stt_out_feats)

    assert st_out_coords.shape[0] == stt_out_coords.shape[0], "Coordinate count mismatch"
    assert torch.equal(st_out_coords, stt_out_coords), "Coordinate values mismatch"
    assert torch.allclose(st_out_feats, stt_out_feats, atol=1e-5), "Feature values mismatch"
