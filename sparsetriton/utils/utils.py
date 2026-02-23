"""General utility functions for SparseTriton.

This module provides common helper functions used throughout the library.

Example:
    >>> from sparsetriton.utils import make_ntuple
    >>> make_ntuple(3, 4)
    (3, 3, 3, 3)
    >>> make_ntuple((1, 2, 3, 4), 4)
    (1, 2, 3, 4)
"""

import torch
from typing import Union, Tuple, List, Any


def make_ntuple(value: Union[int, float, Tuple[Any, ...], List[Any]], n: int) -> Tuple[Any, ...]:
    """Convert a single value or sequence to an n-tuple.

    If a single value is provided, repeats it n times.
    If a tuple or list is provided with length >= n, returns first n elements.

    Args:
        value: Single value or sequence (tuple/list)
        n: Desired length of output tuple

    Returns:
        tuple: Tuple of length n

    Raises:
        TypeError: If value is not int, float, tuple, or list

    Example:
        >>> from sparsetriton.utils import make_ntuple
        >>> make_ntuple(3, 4)
        (3, 3, 3, 3)
        >>> make_ntuple((1, 2, 3, 4, 5), 4)
        (1, 2, 3, 4)
        >>> make_ntuple(2.5, 3)
        (2.5, 2.5, 2.5)
    """
    if isinstance(value, (list, tuple)):
        return tuple(value[:n]) if len(value) >= n else tuple(value) + (value[-1],) * (n - len(value))
    elif isinstance(value, (int, float)):
        return tuple(value for _ in range(n))
    else:
        raise TypeError(f"Expected int, float, tuple, or list, got {type(value)}")
