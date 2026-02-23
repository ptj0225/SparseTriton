"""Tests for sparsetriton.config module."""

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


class TestConvAlgo:
    """Test ConvAlgo enum."""

    def test_enum_values(self):
        """Test that enum has correct values."""
        assert ConvAlgo.ImplicitHashMapGEMM.value == "Implicit_hashmap_gemm"
        assert ConvAlgo.ImplicitHashFlyGEMM.value == "Implicit_hashfly_gemm"

    def test_enum_members(self):
        """Test enum membership."""
        assert ConvAlgo.ImplicitHashMapGEMM in ConvAlgo
        assert ConvAlgo.ImplicitHashFlyGEMM in ConvAlgo


class TestCoordsDtype:
    """Test coordinate dtype configuration."""

    def test_default_coords_dtype(self):
        """Test default coordinate dtype."""
        assert get_coords_dtype() == torch.int16

    def test_set_coords_dtype(self):
        """Test setting coordinate dtype."""
        original = get_coords_dtype()

        set_coords_dtype(torch.int32)
        assert get_coords_dtype() == torch.int32

        set_coords_dtype(torch.int8)
        assert get_coords_dtype() == torch.int8

        # Restore original
        set_coords_dtype(original)

    def test_set_coords_dtype_invalid(self):
        """Test setting invalid coordinate dtype."""
        with pytest.raises(TypeError):
            set_coords_dtype("int32")

        with pytest.raises(TypeError):
            set_coords_dtype(32)


class TestConvAlgoConfig:
    """Test convolution algorithm configuration."""

    def test_default_conv_algo(self):
        """Test default convolution algorithm."""
        assert get_conv_algo() == ConvAlgo.ImplicitHashFlyGEMM

    def test_set_conv_algo(self):
        """Test setting convolution algorithm."""
        original = get_conv_algo()

        set_conv_algo(ConvAlgo.ImplicitHashMapGEMM)
        assert get_conv_algo() == ConvAlgo.ImplicitHashMapGEMM

        set_conv_algo(ConvAlgo.ImplicitHashFlyGEMM)
        assert get_conv_algo() == ConvAlgo.ImplicitHashFlyGEMM

        # Restore original
        set_conv_algo(original)

    def test_set_conv_algo_invalid(self):
        """Test setting invalid convolution algorithm."""
        with pytest.raises(TypeError):
            set_conv_algo("Implicit_hashmap_gemm")

        with pytest.raises(TypeError):
            set_conv_algo(1)


class TestHashTableFactor:
    """Test hash table factor configuration."""

    def test_default_h_table_f(self):
        """Test default hash table factor."""
        assert get_h_table_f() == 1.5

    def test_set_h_table_f(self):
        """Test setting hash table factor."""
        original = get_h_table_f()

        set_h_table_f(2.0)
        assert get_h_table_f() == 2.0

        set_h_table_f(3.5)
        assert get_h_table_f() == 3.5

        # Restore original
        set_h_table_f(original)

    def test_set_h_table_f_integer(self):
        """Test setting hash table factor with integer."""
        original = get_h_table_f()

        set_h_table_f(2)
        assert get_h_table_f() == 2.0

        # Restore original
        set_h_table_f(original)

    def test_set_h_table_f_invalid(self):
        """Test setting invalid hash table factor."""
        with pytest.raises(AssertionError, match="factor must be >= 1"):
            set_h_table_f(0.5)

        with pytest.raises(AssertionError, match="factor must be >= 1"):
            set_h_table_f(0)

        with pytest.raises(AssertionError, match="factor must be >= 1"):
            set_h_table_f(-1)

        with pytest.raises(TypeError, match="factor must be a number"):
            set_h_table_f("not_a_number")

        with pytest.raises(TypeError, match="factor must be a number"):
            set_h_table_f(None)


class TestHashTableMaxProbe:
    """Test hash table max probe configuration."""

    def test_default_h_table_max_p(self):
        """Test default max probe count."""
        assert get_h_table_max_p() == 16

    def test_set_h_table_max_p(self):
        """Test setting max probe count."""
        original = get_h_table_max_p()

        set_h_table_max_p(32)
        assert get_h_table_max_p() == 32

        set_h_table_max_p(64)
        assert get_h_table_max_p() == 64

        # Restore original
        set_h_table_max_p(original)

    def test_set_h_table_max_p_invalid(self):
        """Test setting invalid max probe count."""
        with pytest.raises(AssertionError, match="probe_n must be positive"):
            set_h_table_max_p(0)

        with pytest.raises(AssertionError, match="probe_n must be positive"):
            set_h_table_max_p(-1)

        with pytest.raises(TypeError, match="probe_n must be an integer"):
            set_h_table_max_p("not_a_number")

        with pytest.raises(TypeError, match="probe_n must be an integer"):
            set_h_table_max_p(None)

    def test_set_h_table_max_p_string(self):
        """Test that string numbers are accepted."""
        original = get_h_table_max_p()

        set_h_table_max_p("32")
        assert get_h_table_max_p() == 32

        # Restore original
        set_h_table_max_p(original)

    def test_set_h_table_f_string(self):
        """Test that string numbers are accepted."""
        original = get_h_table_f()

        set_h_table_f("2.0")
        assert get_h_table_f() == 2.0

        # Restore original
        set_h_table_f(original)
