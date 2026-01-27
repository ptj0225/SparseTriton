import torch
import pytest
from sparsetriton import SparseTensor, randn
from sparsetriton.nn.functional import sparse_upsample

@pytest.mark.parametrize("scale_factor", [1, 2, 3])
def test_sparse_upsample_forward(scale_factor):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    device = torch.device("cuda")
    
    # 1. Setup Input
    # Create a single point at (0, 0, 0, 0)
    # Batch=0, X=0, Y=0, Z=0
    coords = torch.tensor([[0, 0, 0, 0]], dtype=torch.int16, device=device)
    feats = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32, device=device)
    spatial_shape = (10, 10, 10)
    batch_size = 1
    
    input_tensor = SparseTensor(feats, coords, spatial_shape, batch_size)
    
    # 2. Run Upsample
    output_tensor = sparse_upsample(input_tensor, scale_factor=scale_factor)
    
    # 3. Verify Output
    # Count
    expected_count = 1 * (scale_factor ** 3)
    assert output_tensor.F.shape[0] == expected_count
    assert output_tensor.C.shape[0] == expected_count
    
    # Features (Nearest Neighbor -> Copy)
    expected_feats = feats.repeat(expected_count, 1)
    assert torch.allclose(output_tensor.F, expected_feats)
    
    # Coords
    # Expected coords: (0, 0..s-1, 0..s-1, 0..s-1)
    expected_coords_list = []
    for x in range(scale_factor):
        for y in range(scale_factor):
            for z in range(scale_factor):
                expected_coords_list.append([0, x, y, z]) # Batch 0
    
    expected_coords = torch.tensor(expected_coords_list, dtype=torch.int16, device=device)
    
    # Helper to sort coords for comparison
    def sort_coords(c):
        c_list = c.tolist()
        c_list.sort()
        return torch.tensor(c_list, dtype=torch.int16, device=device)

    out_c_sorted = sort_coords(output_tensor.C)
    exp_c_sorted = sort_coords(expected_coords)
    
    assert torch.equal(out_c_sorted, exp_c_sorted)
    
    # Spatial Shape
    expected_shape = tuple(s * scale_factor for s in spatial_shape)
    assert tuple(output_tensor.spatial_shape) == expected_shape