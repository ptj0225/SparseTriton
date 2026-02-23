"""Tests for sparsetriton.utils module."""

import pytest
import torch
from sparsetriton.utils import make_ntuple


class TestMakeNtuple:
    """Test make_ntuple function."""

    def test_make_ntuple_int(self):
        """Test make_ntuple with integer input."""
        result = make_ntuple(3, 4)

        assert result == (3, 3, 3, 3)
        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_make_ntuple_float(self):
        """Test make_ntuple with float input."""
        result = make_ntuple(2.5, 3)

        assert result == (2.5, 2.5, 2.5)

    def test_make_ntuple_tuple_exact(self):
        """Test make_ntuple with tuple of exact length."""
        result = make_ntuple((1, 2, 3, 4), 4)

        assert result == (1, 2, 3, 4)

    def test_make_ntuple_tuple_longer(self):
        """Test make_ntuple with longer tuple."""
        result = make_ntuple((1, 2, 3, 4, 5), 4)

        assert result == (1, 2, 3, 4)

    def test_make_ntuple_tuple_shorter(self):
        """Test make_ntuple with shorter tuple."""
        result = make_ntuple((1, 2), 4)

        assert result == (1, 2, 2, 2)

    def test_make_ntuple_list(self):
        """Test make_ntuple with list input."""
        result = make_ntuple([1, 2, 3], 3)

        assert result == (1, 2, 3)

    def test_make_ntuple_invalid_type(self):
        """Test make_ntuple with invalid type."""
        with pytest.raises(TypeError, match="Expected int, float, tuple, or list"):
            make_ntuple("3", 4)

        with pytest.raises(TypeError, match="Expected int, float, tuple, or list"):
            make_ntuple(None, 4)

    def test_make_ntuple_zero_length(self):
        """Test make_ntuple with zero length."""
        result = make_ntuple(5, 0)

        assert result == ()
