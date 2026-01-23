import torch
from enum import Enum

class ConvAlgo(Enum):
    ImplicitHashMapGEMM = "Implicit_hashmap_gemm"
    ImplicitHashFlyGEMM = "Implicit_hashfly_gemm"

_STATE = {
    "coords_dtype": torch.int16,
    "algo": ConvAlgo.ImplicitHashFlyGEMM
}

def set_coords_dtype(dtype: torch.dtype):
    _STATE["coords_dtype"] = dtype

def get_coords_dtype():
    return _STATE["coords_dtype"]

def set_conv_algo(algo: ConvAlgo):
    _STATE["algo"] = algo

def get_conv_algo() -> ConvAlgo:
    return _STATE["algo"]