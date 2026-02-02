import torch
from typing import *

from sparsetriton.config import get_coords_dtype
from sparsetriton.utils.to_dense import to_dense
from sparsetriton.utils.hash import HashTable

__all__ = ["SparseTensor"]

    
class TensorCache:
    def __init__(
        self,
    ) -> None:
        self.kmaps: Dict[Tuple[Any, ...], Any] = {}
        self.hashtable: Optional[HashTable] = None


class SparseTensor:
    def __init__(
        self,
        feats: torch.Tensor,
        coords: torch.Tensor,
        spatial_shape: Tuple[int, ...] = None,
        batch_size: int = None,
        cache: TensorCache = None
    ) -> None:
        self.feats = feats
        self.coords = coords.to(device=feats.device, dtype=get_coords_dtype())
        assert self.feats.shape[0] == self.coords.shape[0], "The number of features and coordinates must match."
        assert self.coords.shape[1] == 4, "Coordinates must have shape (N, 4)."
        assert self.feats.ndim == 2 and self.coords.ndim == 2, "Features and coordinates must be 2D tensors."
        
        if spatial_shape is None:
            self.spatial_shape = torch.Size(self.coords[:, 1:].max(dim=0).values + 1) if self.coords.shape[0] > 0 else torch.Size([0, 0, 0])
        else:
            self.spatial_shape = torch.Size(spatial_shape)
        
        if batch_size is None:
            self.batch_size = int(self.coords[:, 0].max().item() + 1) if self.coords.shape[0] > 0 else 0
        else:
            assert batch_size > self.coords[:, 0].max().item()
            self.batch_size = batch_size

        self._cache = TensorCache() if cache is None else cache

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
    
    def replace(self, feats):
        return SparseTensor(feats, self.coords, self.spatial_shape, self.batch_size, self._cache)

    def __repr__(self):
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
    
    # 1. 충분한 양의 유니크 좌표가 확보될 때까지 루프
    # (randint 후 unique를 거는 방식이 공간이 클 때 permutation보다 훨씬 빠름)
    all_coords = []
    current_nnz = 0
    
    # 좌표 타입 (get_coords_dtype() 대신 직접 지정하거나 호출)
    c_dtype = get_coords_dtype()
    
    while current_nnz < nnz:
        # 부족한 개수만큼 랜덤 생성 (충돌 대비 1.2배 더 생성)
        needed = (nnz - current_nnz)
        sample_size = int(needed * 1.2) if current_nnz > 0 else nnz
        
        temp_list = [torch.randint(0, batch_size, (sample_size, 1), device=device, dtype=c_dtype)]
        for dim_size in spatial_shape:
            temp_list.append(torch.randint(0, dim_size, (sample_size, 1), device=device, dtype=c_dtype))
        
        new_coords = torch.cat(temp_list, dim=1)
        
        # 기존 좌표와 합친 후 중복 제거
        if len(all_coords) > 0:
            all_coords = torch.cat([all_coords, new_coords], dim=0)
        else:
            all_coords = new_coords
            
        all_coords = torch.unique(all_coords, dim=0)
        current_nnz = all_coords.shape[0]

    # 2. 정확히 nnz 개수만큼 슬라이싱
    coords = all_coords[:nnz].contiguous()
    
    # 3. 피처 생성 (좌표 개수에 맞춤)
    feats = torch.randn(nnz, channels, device=device, dtype=dtype).contiguous()

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