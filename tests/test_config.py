import pytest
import torch
from sparsetriton.config import (
    ConvAlgo,
    set_coords_dtype,
    get_coords_dtype,
    set_conv_algo,
    get_conv_algo,
    set_h_table_f,
    get_h_table_f,
    set_h_table_max_p,
    get_h_table_max_p,
    _STATE,
)

_DEFAULT_STATE = dict(_STATE)


@pytest.fixture(autouse=True)
def _restore_config():
    yield
    _STATE.update(_DEFAULT_STATE)


class TestConvAlgo:
    def test_all_enum_values(self):
        assert ConvAlgo.ImplicitHashFlyGEMM.value == "Implicit_hashfly_gemm"
        assert ConvAlgo.PrecomputedNeighborGEMM.value == "Precomputed_neighbor_gemm"

    def test_enum_membership(self):
        assert len(ConvAlgo) == 2


class TestCoordsDtype:
    def test_default(self):
        assert get_coords_dtype() == torch.int16

    def test_set_and_restore(self):
        set_coords_dtype(torch.int32)
        assert get_coords_dtype() == torch.int32

        set_coords_dtype(torch.int8)
        assert get_coords_dtype() == torch.int8

    @pytest.mark.parametrize("invalid", ["int32", 32, None])
    def test_invalid_type(self, invalid):
        with pytest.raises(TypeError):
            set_coords_dtype(invalid)


class TestConvAlgoConfig:
    def test_default(self):
        assert get_conv_algo() == ConvAlgo.PrecomputedNeighborGEMM

    @pytest.mark.parametrize("algo", list(ConvAlgo))
    def test_set_each_algo(self, algo):
        set_conv_algo(algo)
        assert get_conv_algo() == algo

    @pytest.mark.parametrize("invalid", ["Implicit_hashmap_gemm", 1, None])
    def test_invalid_type(self, invalid):
        with pytest.raises(TypeError):
            set_conv_algo(invalid)


class TestHashTableFactor:
    def test_default(self):
        assert get_h_table_f() == 1.5

    def test_set_float(self):
        set_h_table_f(2.0)
        assert get_h_table_f() == 2.0

    def test_set_int(self):
        set_h_table_f(2)
        assert get_h_table_f() == 2.0

    def test_set_string(self):
        set_h_table_f("3.5")
        assert get_h_table_f() == 3.5

    @pytest.mark.parametrize("invalid", [0.5, 0, -1])
    def test_invalid_range(self, invalid):
        with pytest.raises(AssertionError, match="factor must be >= 1"):
            set_h_table_f(invalid)

    @pytest.mark.parametrize("invalid", ["not_a_number", None])
    def test_invalid_type(self, invalid):
        with pytest.raises(TypeError, match="factor must be a number"):
            set_h_table_f(invalid)


class TestHashTableMaxProbe:
    def test_default(self):
        assert get_h_table_max_p() == 16

    def test_set_int(self):
        set_h_table_max_p(32)
        assert get_h_table_max_p() == 32

    def test_set_string(self):
        set_h_table_max_p("64")
        assert get_h_table_max_p() == 64

    @pytest.mark.parametrize("invalid", [0, -1])
    def test_invalid_range(self, invalid):
        with pytest.raises(AssertionError, match="probe_n must be positive"):
            set_h_table_max_p(invalid)

    @pytest.mark.parametrize("invalid", ["not_a_number", None])
    def test_invalid_type(self, invalid):
        with pytest.raises(TypeError, match="probe_n must be an integer"):
            set_h_table_max_p(invalid)
