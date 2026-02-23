"""Global configuration for SparseTriton.

This module provides a centralized configuration system for SparseTriton,
controlling data types, algorithm choices, and hash table parameters.

Example:
    >>> import torch
    >>> from sparsetriton.config import set_coords_dtype, get_coords_dtype, set_conv_algo, ConvAlgo
    >>> set_coords_dtype(torch.int32)
    >>> get_coords_dtype() == torch.int32
    True
    >>> set_conv_algo(ConvAlgo.ImplicitHashFlyGEMM)
    >>> get_conv_algo() == ConvAlgo.ImplicitHashFlyGEMM
    True
"""

import torch
from enum import Enum
from typing import Optional

class ConvAlgo(Enum):
    """Enumeration of available sparse convolution algorithms.

    Attributes:
        ImplicitHashMapGEMM: Implicit Hash Map based GEMM algorithm
        ImplicitHashFlyGEMM: Implicit Hash Fly based GEMM algorithm (default, usually faster)

    Example:
        >>> from sparsetriton.config import ConvAlgo
        >>> algo = ConvAlgo.ImplicitHashFlyGEMM
        >>> algo.value
        'Implicit_hashfly_gemm'
    """
    ImplicitHashMapGEMM = "Implicit_hashmap_gemm"
    ImplicitHashFlyGEMM = "Implicit_hashfly_gemm"


_STATE: dict = {
    "coords_dtype": torch.int16,
    "algo": ConvAlgo.ImplicitHashFlyGEMM,
    "h_tbl_factor": 1.5,
    "h_tbl_max_probe_n": 16,
}


def set_coords_dtype(dtype: torch.dtype) -> None:
    """Set the data type for coordinate tensors.

    Args:
        dtype: PyTorch data type for coordinates (e.g., torch.int16, torch.int32)

    Raises:
        TypeError: If dtype is not a valid torch.dtype

    Example:
        >>> import torch
        >>> from sparsetriton.config import set_coords_dtype, get_coords_dtype
        >>> set_coords_dtype(torch.int32)
        >>> get_coords_dtype() == torch.int32
        True
    """
    if not isinstance(dtype, torch.dtype):
        raise TypeError(f"Expected torch.dtype, got {type(dtype)}")
    _STATE["coords_dtype"] = dtype


def get_coords_dtype() -> torch.dtype:
    """Get the current data type for coordinate tensors.

    Returns:
        torch.dtype: The current coordinate data type

    Example:
        >>> from sparsetriton.config import get_coords_dtype
        >>> dtype = get_coords_dtype()
        >>> isinstance(dtype, torch.dtype)
        True
    """
    return _STATE["coords_dtype"]


def set_conv_algo(algo: ConvAlgo) -> None:
    """Set the sparse convolution algorithm.

    Args:
        algo: The convolution algorithm to use

    Raises:
        TypeError: If algo is not a ConvAlgo enum

    Example:
        >>> from sparsetriton.config import set_conv_algo, get_conv_algo, ConvAlgo
        >>> set_conv_algo(ConvAlgo.ImplicitHashFlyGEMM)
        >>> get_conv_algo() == ConvAlgo.ImplicitHashFlyGEMM
        True
    """
    if not isinstance(algo, ConvAlgo):
        raise TypeError(f"Expected ConvAlgo, got {type(algo)}")
    _STATE["algo"] = algo


def get_conv_algo() -> ConvAlgo:
    """Get the current sparse convolution algorithm.

    Returns:
        ConvAlgo: The current convolution algorithm

    Example:
        >>> from sparsetriton.config import get_conv_algo
        >>> algo = get_conv_algo()
        >>> isinstance(algo, ConvAlgo)
        True
    """
    return _STATE["algo"]


def set_h_table_f(factor: float) -> None:
    """Set the hash table size factor.

    The hash table size is calculated as: table_size = factor * nnz

    Args:
        factor: Size factor, must be >= 1.0 (int, float, or string that can be cast to float)

    Raises:
        AssertionError: If factor is less than 1
        TypeError: If factor cannot be converted to float

    Example:
        >>> from sparsetriton.config import set_h_table_f, get_h_table_f
        >>> set_h_table_f(2.0)
        >>> get_h_table_f()
        2.0
    """
    try:
        factor_float = float(factor)
    except (TypeError, ValueError):
        raise TypeError("factor must be a number")
    assert factor_float >= 1, "factor must be >= 1"
    _STATE["h_tbl_factor"] = factor_float


def get_h_table_f() -> float:
    """Get the current hash table size factor.

    Returns:
        float: The current hash table size factor

    Example:
        >>> from sparsetriton.config import get_h_table_f
        >>> factor = get_h_table_f()
        >>> isinstance(factor, float)
        True
        >>> factor >= 1.0
        True
    """
    return _STATE["h_tbl_factor"]


def set_h_table_max_p(probe_n: int) -> None:
    """Set the maximum number of probe attempts for hash table collisions.

    When inserting into the hash table, this determines how many
    alternative positions to try before giving up.

    Args:
        probe_n: Maximum probe attempts, must be positive (int or string that can be cast to int)

    Raises:
        AssertionError: If probe_n is not positive
        TypeError: If probe_n cannot be converted to int

    Example:
        >>> from sparsetriton.config import set_h_table_max_p, get_h_table_max_p
        >>> set_h_table_max_p(32)
        >>> get_h_table_max_p()
        32
    """
    try:
        probe_n_int = int(probe_n)
    except (TypeError, ValueError):
        raise TypeError("probe_n must be an integer")
    assert probe_n_int > 0, "probe_n must be positive"
    _STATE["h_tbl_max_probe_n"] = probe_n_int


def get_h_table_max_p() -> int:
    """Get the current maximum number of probe attempts.

    Returns:
        int: The current maximum probe attempts

    Example:
        >>> from sparsetriton.config import get_h_table_max_p
        >>> max_probe = get_h_table_max_p()
        >>> isinstance(max_probe, int)
        True
        >>> max_probe > 0
        True
    """
    return _STATE["h_tbl_max_probe_n"]
