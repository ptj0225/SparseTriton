"""Sparse tensor implementation for 3D sparse operations.

This module provides the core SparseTensor class and related utilities for
handling sparse 3D point cloud and volumetric data with PyTorch compatibility.

Example:
    >>> import torch
    >>> from sparsetriton import SparseTensor, randn
    >>> # Create random sparse tensor
    >>> sp_tensor = randn((32, 32, 32), batch_size=2, nnz=10, device="cpu", channels=4)
    >>> print(sp_tensor.feats.shape)
    torch.Size([10, 4])
    >>> print(sp_tensor.coords.shape)
    torch.Size([10, 4])
"""

import torch
from typing import Optional, Tuple, Union, Dict, Any, List

from sparsetriton.config import get_coords_dtype
from sparsetriton.utils.to_dense import to_dense
from sparsetriton.utils.hash import HashTable

__all__ = ["SparseTensor", "randn"]


class TensorCache:
    """Cache for storing tensor-related computed data.

    This class stores intermediate computations like kernel maps (kmaps)
    and hash tables to avoid redundant calculations.

    Attributes:
        kmaps: Dictionary storing kernel mapping tensors
        hashtable: Cached hash table for coordinate lookups

    Example:
        >>> from sparsetriton.tensor import TensorCache
        >>> cache = TensorCache()
        >>> cache.kmaps = {}
        >>> cache.hashtable is None
        True
    """

    def __init__(self) -> None:
        """Initialize an empty TensorCache."""
        self.kmaps: Dict[Tuple[Any, ...], Any] = {}
        self.hashtable: Optional[HashTable] = None


class SparseTensor:
    """Sparse tensor for 3D spatial data.

    A sparse tensor stores only active (non-zero) points with their features
    and 3D coordinates. Coordinates are stored as (batch, x, y, z) format.

    Attributes:
        feats: Feature tensor of shape (N, C) where N is number of active points
        coords: Coordinate tensor of shape (N, 4) in (batch, x, y, z) format
        spatial_shape: Tuple (D, H, W) representing the spatial dimensions
        batch_size: Number of batches in the tensor
        F: Property alias for feats
        C: Property alias for coords

    Example:
        >>> import torch
        >>> from sparsetriton import SparseTensor
        >>> feats = torch.randn(5, 3)
        >>> coords = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 3, 3], [1, 0, 0, 0], [1, 1, 1, 1]])
        >>> sp = SparseTensor(feats, coords)
        >>> print(sp.spatial_shape)
        torch.Size([2, 4, 5])
        >>> print(sp.batch_size)
        2
    """

    def __init__(
        self,
        feats: torch.Tensor,
        coords: torch.Tensor,
        spatial_shape: Optional[Tuple[int, ...]] = None,
        batch_size: Optional[int] = None,
        cache: Optional[TensorCache] = None,
    ) -> None:
        """Initialize a SparseTensor.

        Args:
            feats: Feature tensor of shape (N, C) where N is number of active points
            coords: Coordinate tensor of shape (N, 4) in (batch, x, y, z) format.
                   Will be converted to coords_dtype and moved to feats.device.
            spatial_shape: Optional spatial dimensions (D, H, W). If None, computed from coords.
            batch_size: Optional number of batches. If None, computed from coords.
            cache: Optional TensorCache for sharing across operations

        Raises:
            AssertionError: If feats and coords have different lengths
            AssertionError: If coords doesn't have shape (N, 4)
            AssertionError: If coords and feats are not 2D tensors

        Example:
            >>> import torch
            >>> from sparsetriton import SparseTensor
            >>> feats = torch.randn(3, 4)
            >>> coords = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 3, 3]])
            >>> sp = SparseTensor(feats, coords)
            >>> print(sp.feats.shape)
            torch.Size([3, 4])
        """
        self.feats = feats
        self.coords = coords.to(device=feats.device, dtype=get_coords_dtype())

        assert self.feats.shape[0] == self.coords.shape[0], "The number of features and coordinates must match."
        assert self.coords.shape[1] == 4, "Coordinates must have shape (N, 4)."
        assert self.feats.ndim == 2 and self.coords.ndim == 2, "Features and coordinates must be 2D tensors."

        if spatial_shape is None:
            if self.coords.shape[0] > 0:
                self.spatial_shape = torch.Size(self.coords[:, 1:].max(dim=0).values + 1)
            else:
                self.spatial_shape = torch.Size([0, 0, 0])
        else:
            self.spatial_shape = torch.Size(spatial_shape)

        if batch_size is None:
            if self.coords.shape[0] > 0:
                self.batch_size = int(self.coords[:, 0].max().item() + 1)
            else:
                self.batch_size = 0
        else:
            assert batch_size > self.coords[:, 0].max().item(), f"batch_size {batch_size} must be > max batch index {self.coords[:, 0].max().item()}"
            self.batch_size = batch_size

        self._cache = TensorCache() if cache is None else cache

    @property
    def F(self) -> torch.Tensor:
        """Get feature tensor.

        Returns:
            torch.Tensor: Feature tensor of shape (N, C)

        Example:
            >>> import torch
            >>> from sparsetriton import SparseTensor
            >>> feats = torch.randn(3, 4)
            >>> coords = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 3, 3]])
            >>> sp = SparseTensor(feats, coords)
            >>> sp.F.shape == sp.feats.shape
            True
        """
        return self.feats

    @F.setter
    def F(self, feats: torch.Tensor) -> None:
        """Set feature tensor.

        Args:
            feats: New feature tensor

        Example:
            >>> import torch
            >>> from sparsetriton import SparseTensor
            >>> feats = torch.randn(3, 4)
            >>> coords = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 3, 3]])
            >>> sp = SparseTensor(feats, coords)
            >>> new_feats = torch.randn(3, 5)
            >>> sp.F = new_feats
            >>> sp.feats.shape
            torch.Size([3, 5])
        """
        self.feats = feats

    @property
    def C(self) -> torch.Tensor:
        """Get coordinate tensor.

        Returns:
            torch.Tensor: Coordinate tensor of shape (N, 4)

        Example:
            >>> import torch
            >>> from sparsetriton import SparseTensor
            >>> feats = torch.randn(3, 4)
            >>> coords = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 3, 3]])
            >>> sp = SparseTensor(feats, coords)
            >>> sp.C.shape == sp.coords.shape
            True
        """
        return self.coords

    @C.setter
    def C(self, coords: torch.Tensor) -> None:
        """Set coordinate tensor.

        Args:
            coords: New coordinate tensor, will be converted to coords_dtype

        Example:
            >>> import torch
            >>> from sparsetriton import SparseTensor
            >>> feats = torch.randn(3, 4)
            >>> coords = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 3, 3]])
            >>> sp = SparseTensor(feats, coords)
            >>> new_coords = torch.tensor([[0, 2, 3, 4], [0, 2, 3, 5], [0, 2, 4, 4]])
            >>> sp.C = new_coords
            >>> sp.coords.shape
            torch.Size([3, 4])
        """
        self.coords = coords.to(get_coords_dtype())

    def to(self, device: Union[str, torch.device], non_blocking: bool = False) -> "SparseTensor":
        """Move tensor to specified device.

        Args:
            device: Target device (e.g., "cpu", "cuda", torch.device)
            non_blocking: Whether to perform transfer asynchronously

        Returns:
            SparseTensor: self after moving

        Example:
            >>> import torch
            >>> from sparsetriton import SparseTensor
            >>> feats = torch.randn(3, 4)
            >>> coords = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 3, 3]])
            >>> sp = SparseTensor(feats, coords)
            >>> sp_cpu = sp.to("cpu")
            >>> sp_cpu.feats.device.type
            'cpu'
        """
        self.feats = self.feats.to(device, non_blocking=non_blocking)
        self.coords = self.coords.to(device, non_blocking=non_blocking)
        return self

    def cpu(self) -> "SparseTensor":
        """Move tensor to CPU.

        Returns:
            SparseTensor: self after moving to CPU

        Example:
            >>> import torch
            >>> from sparsetriton import SparseTensor
            >>> feats = torch.randn(3, 4)
            >>> coords = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 3, 3]])
            >>> sp = SparseTensor(feats, coords)
            >>> sp_cpu = sp.cpu()
            >>> sp_cpu.feats.device.type
            'cpu'
        """
        return self.to("cpu")

    def half(self) -> "SparseTensor":
        """Convert features to half precision (float16).

        Returns:
            SparseTensor: self after conversion

        Example:
            >>> import torch
            >>> from sparsetriton import SparseTensor
            >>> feats = torch.randn(3, 4)
            >>> coords = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 3, 3]])
            >>> sp = SparseTensor(feats, coords)
            >>> sp_half = sp.half()
            >>> sp_half.feats.dtype == torch.float16
            True
        """
        self.feats = self.feats.half()
        return self

    def float(self) -> "SparseTensor":
        """Convert features to single precision (float32).

        Returns:
            SparseTensor: self after conversion

        Example:
            >>> import torch
            >>> from sparsetriton import SparseTensor
            >>> feats = torch.randn(3, 4, dtype=torch.float16)
            >>> coords = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 3, 3]])
            >>> sp = SparseTensor(feats, coords)
            >>> sp_float = sp.float()
            >>> sp_float.feats.dtype == torch.float32
            True
        """
        self.feats = self.feats.float()
        return self

    def dense(self) -> torch.Tensor:
        """Convert sparse tensor to dense tensor.

        Returns:
            torch.Tensor: Dense tensor of shape (B, C, D, H, W)

        Example:
            >>> import torch
            >>> from sparsetriton import SparseTensor
            >>> feats = torch.tensor([[1.0], [2.0], [3.0]])
            >>> coords = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]])
            >>> sp = SparseTensor(feats, coords)
            >>> dense = sp.dense()
            >>> dense.shape
            torch.Size([2, 1, 1, 1, 2])
        """
        return to_dense(self.feats, self.coords, self.spatial_shape)

    def replace(self, feats: torch.Tensor) -> "SparseTensor":
        """Create a new SparseTensor with replaced features.

        Keeps the same coordinates, spatial_shape, batch_size, and cache.

        Args:
            feats: New feature tensor

        Returns:
            SparseTensor: New SparseTensor with new features

        Example:
            >>> import torch
            >>> from sparsetriton import SparseTensor
            >>> feats = torch.randn(3, 4)
            >>> coords = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 3, 3]])
            >>> sp = SparseTensor(feats, coords)
            >>> new_feats = torch.randn(3, 4)
            >>> sp2 = sp.replace(new_feats)
            >>> torch.allclose(sp2.feats, new_feats)
            True
            >>> torch.allclose(sp2.coords, sp.coords)
            True
        """
        return SparseTensor(feats, self.coords, self.spatial_shape, self.batch_size, self._cache)

    def __repr__(self) -> str:
        """String representation of the SparseTensor.

        Returns:
            str: Formatted string representation
        """
        return (
            f"SparseTensor(\n"
            f"  feats=tensor(shape={self.feats.shape}, dtype={self.feats.dtype}, device={self.feats.device}),\n"
            f"  coords=tensor(shape={self.coords.shape}, dtype={self.coords.dtype}, device={self.coords.device}),\n"
            f"  spatial_shape={self.spatial_shape}\n"
            f")"
        )


def randn(
    spatial_shape: Tuple[int, ...],
    batch_size: int = 1,
    channels: int = 1,
    nnz: int = 100,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
) -> SparseTensor:
    """Generate a random SparseTensor.

    Creates a sparse tensor with random features and unique random coordinates.
    Uses efficient coordinate generation with collision handling.

    Args:
        spatial_shape: Spatial dimensions (D, H, W)
        batch_size: Number of batches (default: 1)
        channels: Number of feature channels (default: 1)
        nnz: Number of non-zero points to generate (default: 100)
        device: Target device (default: "cpu")
        dtype: Data type for features (default: torch.float32)

    Returns:
        SparseTensor: Random sparse tensor with specified properties

    Raises:
        ValueError: If nnz > batch_size * D * H * W

    Example:
        >>> from sparsetriton import randn
        >>> sp = randn((16, 16, 16), batch_size=2, nnz=10, channels=4, device="cpu")
        >>> sp.feats.shape
        torch.Size([10, 4])
        >>> sp.coords.shape
        torch.Size([10, 4])
        >>> len(torch.unique(sp.coords, dim=0))
        10
    """
    # Check if nnz is feasible
    total_voxels = batch_size * spatial_shape[0] * spatial_shape[1] * spatial_shape[2]
    if nnz > total_voxels:
        raise ValueError(f"nnz ({nnz}) exceeds total available voxels ({total_voxels})")

    # Generate unique coordinates
    all_coords: List[torch.Tensor] = []
    current_nnz = 0

    c_dtype = get_coords_dtype()

    while current_nnz < nnz:
        # Generate more coordinates than needed (account for collisions)
        needed = nnz - current_nnz
        sample_size = int(needed * 1.2) if current_nnz > 0 else nnz

        temp_list = [
            torch.randint(0, batch_size, (sample_size, 1), device=device, dtype=c_dtype)
        ]
        for dim_size in spatial_shape:
            temp_list.append(torch.randint(0, dim_size, (sample_size, 1), device=device, dtype=c_dtype))

        new_coords = torch.cat(temp_list, dim=1)

        # Merge and deduplicate
        if len(all_coords) > 0:
            all_coords = [torch.cat([all_coords[0], new_coords], dim=0)]
        else:
            all_coords = [new_coords]

        all_coords[0] = torch.unique(all_coords[0], dim=0)
        current_nnz = all_coords[0].shape[0]

    # Slice to exactly nnz coordinates
    coords = all_coords[0][:nnz].contiguous()

    # Generate random features
    feats = torch.randn(nnz, channels, device=device, dtype=dtype).contiguous()

    return SparseTensor(feats, coords, spatial_shape=spatial_shape)
