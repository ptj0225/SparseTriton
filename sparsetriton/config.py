import torch

__all__ = ["set_coords_dtype", "get_coords_dtype"]

_coords_dtype = torch.int16


def set_coords_dtype(dtype: torch.dtype) -> None:
    global _coords_dtype
    _coords_dtype = dtype


def get_coords_dtype() -> torch.dtype:
    return _coords_dtype