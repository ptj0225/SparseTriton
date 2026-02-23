"""Utility functions for sparse tensor operations.

This module provides helper functions for coordinate transformations,
hashing, and spatial operations.

Example:
    >>> from sparsetriton.utils import HashTable, to_dense, make_ntuple
    >>> make_ntuple(3, 4)
    (3, 3, 3, 3)
"""

from .hash import HashTable
from .to_dense import to_dense
from .utils import make_ntuple
from .spatial import mask_spatial_range

__all__ = ["HashTable", "to_dense", "make_ntuple", "mask_spatial_range"]
