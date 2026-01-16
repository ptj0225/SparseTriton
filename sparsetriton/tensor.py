from numpy import size
import torch
from typing import *

from sparsetriton.config import get_coords_dtype
from sparsetriton.utils.to_dense import to_dense

__all__ = ["SparseTensor"]


class SparseTensor:
    def __init__(
        self,
        feats: torch.Tensor,
        coords: torch.Tensor,
        spatial_shape: Tuple[int, ...] = None,
        batch_size: int = None,
    ) -> None:
        self.feats = feats
        self.coords = coords.to(get_coords_dtype())
        if spatial_shape is None:
            self.spatial_shape = torch.Size(coords[:, 1:].max(dim=0).values + 1)
        else:
            self.spatial_shape = torch.Size(spatial_shape)
        
        if batch_size is None:
            self.batch_size = int(coords[:, 0].max().item() + 1)
        else:
            assert batch_size > coords[:, 0].max().item()
            self.batch_size = batch_size

        self._cache = TensorCache()

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

    def dense(self) -> torch.Tensor:
        return to_dense(self.feats, self.coords, self.spatial_shape)

    def __repr__(self):
        return (
            f"SparseTensor(\n"
            f"  feats=tensor(shape={self.feats.shape}, dtype={self.feats.dtype}, device={self.feats.device}),\n"
            f"  coords=tensor(shape={self.coords.shape}, dtype={self.coords.dtype}, device={self.coords.device}),\n"
            f"  spatial_shape={self.spatial_shape}\n"
            f")"
        )
    
class TensorCache:
    def __init__(
        self,
    ) -> None:
        self.cmaps: Dict[Tuple[int, ...], Tuple[torch.Tensor, Tuple[int, ...]]] = {}
        self.kmaps: Dict[Tuple[Any, ...], Any] = {}
        self.hashmaps: Dict[Tuple[int, ...], Tuple[Any, ...]] = {}

def randn(
    spatial_shape: Tuple[int, ...],
    batch_size: int = 1,
    channels: int = 1,
    nnz: int = 100,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
) -> SparseTensor:
    
    feats = torch.randn(nnz, channels, device=device, dtype=dtype)

    coords_list = [torch.randint(0, batch_size, (nnz, 1), device=device, dtype=get_coords_dtype())]
    for dim_size in spatial_shape:
        coords_list.append(torch.randint(0, dim_size, (nnz, 1), device=device, dtype=get_coords_dtype()))
    coords = torch.cat(coords_list, dim=1)

    return SparseTensor(feats, coords, spatial_shape=spatial_shape)

if __name__ == "__main__":
    from tqdm import tqdm
    # Simple test

    sp_tensor = randn((256, 256, 256), nnz=1, device="cuda", channels=16, batch_size=1)
    sp_tensor.feats.requires_grad = True
    sp_tensor.feats.requires_grad = True
    for _ in tqdm(range(1)):
        (sp_tensor.dense()**2).sum().backward()
        print(sp_tensor.feats.grad.norm())
        

    print(sp_tensor.dense().shape)
    print(sp_tensor.feats)
    print(sp_tensor.coords)

    sp_tensor = SparseTensor(feats=torch.tensor([[1.0, 2.0], [2.0, 2.0], [3.0, 3.0]], device="cuda"),
                             coords=torch.tensor([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2]], device="cuda"))
    print(sp_tensor.dense())
    print(sp_tensor.dense().shape)