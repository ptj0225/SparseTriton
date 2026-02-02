import torch
from enum import Enum

class ConvAlgo(Enum):
    ImplicitHashMapGEMM = "Implicit_hashmap_gemm"
    ImplicitHashFlyGEMM = "Implicit_hashfly_gemm"

_STATE = {
    "coords_dtype": torch.int16,
    "algo": ConvAlgo.ImplicitHashFlyGEMM,
    "h_tbl_factor": 1.5,
}

def set_coords_dtype(dtype: torch.dtype):
    _STATE["coords_dtype"] = dtype

def get_coords_dtype():
    return _STATE["coords_dtype"]

def set_conv_algo(algo: ConvAlgo):
    _STATE["algo"] = algo

def get_conv_algo() -> ConvAlgo:
    return _STATE["algo"]

def set_h_table_f(factor: float): 
    assert factor >= 1, "factor must be >= 1"
    _STATE["h_tbl_factor"] = factor

def get_h_table_f():
    return _STATE["h_tbl_factor"]

def set_h_table_max_p(probe_n: int):
    _STATE["h_tbl_max_probe_n"] = probe_n

def get_h_table_max_p():
    return _STATE["h_tbl_max_probe_n"]
    
