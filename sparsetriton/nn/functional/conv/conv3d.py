import math
import torch
import torch.nn as nn
from typing import Union, Tuple, List, Optional

from ..tensor import SparseTensor
from ..utils.hash import HashTable

class SparseConv3d(nn.Module):
    """
    Sparse 3D Convolution using Triton-based HashTable for neighbor search.
    Currently supports Submanifold Sparse Convolution (stride=1, same input/output coords).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        bias: bool = True,
        device=None,
        dtype=None
    ):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        def to_3d(x):
            return x if isinstance(x, (list, tuple)) else [x] * 3
            
        self.kernel_size = to_3d(kernel_size)
        self.stride = to_3d(stride)
        self.padding = to_3d(padding)
        self.dilation = to_3d(dilation)
        
        assert all(s == 1 for s in self.stride), "Only stride=1 (Submanifold) is currently supported."
        
        self.kernel_volume = self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
        
        self.weight = nn.Parameter(torch.empty((self.kernel_volume, in_channels, out_channels), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
            
        self.register_buffer('offsets', self._compute_offsets())
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def _compute_offsets(self):
        kx, ky, kz = self.kernel_size
        offsets = []
        # Order matches the weight dimension 0
        for z in range(kz):
            for y in range(ky):
                for x in range(kx):
                    dx = (x - kx // 2) * self.dilation[0]
                    dy = (y - ky // 2) * self.dilation[1]
                    dz = (z - kz // 2) * self.dilation[2]
                    offsets.append([0, dx, dy, dz])
        return torch.tensor(offsets, dtype=torch.int32)

    def forward(self, input: SparseTensor) -> SparseTensor:
        coords = input.coords
        feats = input.feats
        
        N = coords.shape[0]
        device = coords.device
        
        # Build Hash Table
        # Capacity should be power of 2 and sufficiently large to minimize collisions
        if N > 0:
            capacity = 2 ** math.ceil(math.log2(N * 2))
        else:
            capacity = 1024
            
        ht = HashTable(capacity, device)
        ht.insert(coords)
        
        out_feats = torch.zeros((N, self.out_channels), dtype=feats.dtype, device=device)
        
        # Ensure offsets are on correct device
        if self.offsets.device != device:
            self.offsets = self.offsets.to(device)
            
        # Iterate over kernel weights (implicit GEMM approach via gather/scatter)
        for i in range(self.kernel_volume):
            offset = self.offsets[i]
            
            # Calculate neighbor coordinates: p + offset
            target_coords = coords + offset.to(coords.dtype)
            
            # Query Hash Table to find if neighbor exists in input
            neighbor_indices = ht.query(target_coords)
            
            # -1 means no neighbor at that offset
            mask = neighbor_indices != -1
            
            if mask.any():
                valid_neighbor_indices = neighbor_indices[mask]
                gathered_feats = feats[valid_neighbor_indices]
                w = self.weight[i]
                weighted_feats = gathered_feats @ w
                
                # Accumulate result into the output features
                active_output_indices = torch.nonzero(mask).squeeze(1)
                out_feats.index_add_(0, active_output_indices, weighted_feats)
                
        if self.bias is not None:
            out_feats += self.bias
            
        return SparseTensor(out_feats, coords)
