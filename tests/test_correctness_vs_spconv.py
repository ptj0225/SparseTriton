import torch
import pytest

try:
    import spconv.pytorch as spconv
    HAS_SPCONV = True
except ImportError:
    HAS_SPCONV = False

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(not HAS_SPCONV, reason="spconv not installed"),
]

from sparsetriton import SparseTensor
from sparsetriton.nn.modules import SubMConv3D, SparseConv3D
from sparsetriton.config import set_conv_algo, ConvAlgo

RTOL = 1e-3
ATOL = 1e-3


def _create_sparse_data(batch_size, spatial_size, nnz, in_channels, device, seed=42):
    D, H, W = spatial_size
    torch.manual_seed(seed)

    coords_set = set()
    coords_list = []
    while len(coords_list) < nnz:
        b = torch.randint(0, batch_size, (1,)).item()
        x = torch.randint(0, W, (1,)).item()
        y = torch.randint(0, H, (1,)).item()
        z = torch.randint(0, D, (1,)).item()
        key = (b, x, y, z)
        if key not in coords_set:
            coords_set.add(key)
            coords_list.append([b, x, y, z])

    coords = torch.tensor(coords_list, device=device, dtype=torch.int32)
    features = torch.randn(nnz, in_channels, device=device)
    return coords, features


def _convert_spconv_weight(weight_spconv):
    C_out, Kz, Ky, Kx, C_in = weight_spconv.shape
    K = Kz
    K_vol = K**3
    weight_st = torch.zeros((K_vol, C_in, C_out), device=weight_spconv.device, dtype=weight_spconv.dtype)

    for k in range(K_vol):
        dx = k // (K * K) - K // 2
        dy = (k % (K * K)) // K - K // 2
        dz = k % K - K // 2
        Kz_idx = dz + K // 2
        Ky_idx = dy + K // 2
        Kx_idx = dx + K // 2
        weight_st[k] = weight_spconv[:, Kz_idx, Ky_idx, Kx_idx, :].permute(1, 0)

    return weight_st


class TestSubMConv3DVsSpconv:
    @pytest.fixture(autouse=True)
    def _use_precomputed(self):
        original = ConvAlgo.PrecomputedNeighborGEMM
        set_conv_algo(ConvAlgo.PrecomputedNeighborGEMM)
        yield
        set_conv_algo(original)

    def _make_inputs(self, batch_size=2, spatial_size=(32, 32, 32), nnz=1000,
                     in_channels=8, out_channels=16, kernel_size=3, seed=123):
        device = torch.device("cuda")
        coords, features = _create_sparse_data(
            batch_size, spatial_size, nnz, in_channels, device, seed=seed
        )

        indices_spconv = coords[:, [0, 3, 2, 1]].contiguous()
        sp_tensor_spconv = spconv.SparseConvTensor(
            features, indices_spconv, list(spatial_size), batch_size
        )
        conv_spconv = spconv.SubMConv3d(
            in_channels, out_channels, kernel_size, bias=False
        ).to(device)

        sp_tensor_st = SparseTensor(
            feats=features.clone(),
            coords=coords,
            spatial_shape=spatial_size,
            batch_size=batch_size,
        )
        conv_st = SubMConv3D(
            in_channels, out_channels, kernel_size, bias=False
        ).to(device)

        with torch.no_grad():
            conv_st.weight.copy_(_convert_spconv_weight(conv_spconv.weight))

        return sp_tensor_spconv, conv_spconv, sp_tensor_st, conv_st

    def test_forward(self):
        sp_spconv, conv_spconv, sp_st, conv_st = self._make_inputs()
        out_spconv = conv_spconv(sp_spconv)
        out_st = conv_st(sp_st)
        torch.testing.assert_close(
            out_st.feats, out_spconv.features, atol=ATOL, rtol=RTOL
        )

    def test_backward_input_grad(self):
        sp_spconv, conv_spconv, sp_st, conv_st = self._make_inputs()

        feat_spconv = sp_spconv.features.detach().clone().requires_grad_(True)
        sp_spconv = spconv.SparseConvTensor(
            feat_spconv, sp_spconv.indices, sp_spconv.spatial_shape, sp_spconv.batch_size
        )
        feat_st = sp_st.feats.detach().clone().requires_grad_(True)
        sp_st = sp_st.replace(feat_st)

        out_spconv = conv_spconv(sp_spconv)
        out_st = conv_st(sp_st)

        grad = torch.randn_like(out_spconv.features)
        out_spconv.features.backward(grad)
        out_st.feats.backward(grad)

        torch.testing.assert_close(
            feat_st.grad, feat_spconv.grad, atol=ATOL, rtol=RTOL
        )

    @pytest.mark.parametrize("kernel_size", [3, 5])
    def test_kernel_sizes(self, kernel_size):
        sp_spconv, conv_spconv, sp_st, conv_st = self._make_inputs(
            batch_size=1, spatial_size=(16, 16, 16), nnz=500,
            in_channels=4, out_channels=8, kernel_size=kernel_size,
        )
        out_spconv = conv_spconv(sp_spconv)
        out_st = conv_st(sp_st)
        torch.testing.assert_close(
            out_st.feats, out_spconv.features, atol=ATOL, rtol=RTOL
        )
