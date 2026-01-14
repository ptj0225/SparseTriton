import torch
from typing import Tuple, Union

from sparsetriton.config import get_coords_dtype
from sparsetriton.utils.to_dense import to_dense

__all__ = ["SparseTensor"]


class SparseTensor:
    def __init__(
        self,
        feats: torch.Tensor,
        coords: torch.Tensor,
        stride: Union[int, Tuple[int, ...]] = 1,
    ) -> None:
        self.feats = feats
        self.coords = coords.to(get_coords_dtype())

        if isinstance(stride, int):
            self.stride = tuple(stride for _ in range(coords.shape[1] - 1))
        else:
            self.stride = stride

    @property
    def F(self) -> torch.Tensor:
        return self.feats

    @F.setter
    def F(self, feats: torch.Tensor) -> None:
        self.feats = feats

    @property
    def C(self) -> torch.Tensor:
        return self.coords

    @C.setter
    def C(self, coords: torch.Tensor) -> None:
        self.coords = coords.to(get_coords_dtype())

    @property
    def s(self) -> Tuple[int, ...]:
        return self.stride

    @s.setter
    def s(self, stride: Union[int, Tuple[int, ...]]) -> None:
        if isinstance(stride, int):
            self.stride = tuple(stride for _ in range(self.coords.shape[1] - 1))
        else:
            self.stride = stride

    def to(self, device: Union[str, torch.device], non_blocking: bool = False):
        self.feats = self.feats.to(device, non_blocking=non_blocking)
        self.coords = self.coords.to(device, non_blocking=non_blocking)
        return self

    def cpu(self):
        return self.to("cpu")

    def half(self):
        self.feats = self.feats.half()
        return self

    def float(self):
        self.feats = self.feats.float()
        return self

    def dense(self, spatial_range: Tuple[int, ...]) -> torch.Tensor:
        return to_dense(self.feats, self.coords, spatial_range)

    def __repr__(self):
        return (
            f"SparseTensor(\n"
            f"  feats=tensor(shape={self.feats.shape}, dtype={self.feats.dtype}, device={self.feats.device}),\n"
            f"  coords=tensor(shape={self.coords.shape}, dtype={self.coords.dtype}, device={self.coords.device}),\n"
            f"  stride={self.stride}\n"
            f")"
        )