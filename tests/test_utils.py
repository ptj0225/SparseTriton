import pytest
from sparsetriton.utils import make_ntuple


class TestMakeNtuple:
    def test_int(self):
        assert make_ntuple(3, 4) == (3, 3, 3, 3)

    def test_float(self):
        assert make_ntuple(2.5, 3) == (2.5, 2.5, 2.5)

    def test_tuple_exact(self):
        assert make_ntuple((1, 2, 3, 4), 4) == (1, 2, 3, 4)

    def test_tuple_longer(self):
        assert make_ntuple((1, 2, 3, 4, 5), 4) == (1, 2, 3, 4)

    def test_tuple_shorter(self):
        assert make_ntuple((1, 2), 4) == (1, 2, 2, 2)

    def test_list(self):
        assert make_ntuple([1, 2, 3], 3) == (1, 2, 3)

    @pytest.mark.parametrize("invalid", ["3", None])
    def test_invalid_type(self, invalid):
        with pytest.raises(TypeError, match="Expected int, float, tuple, or list"):
            make_ntuple(invalid, 4)

    def test_zero_length(self):
        assert make_ntuple(5, 0) == ()
