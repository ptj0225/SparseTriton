import torch
from typing import Union, Tuple

def make_ntuple(value: Union[int, Tuple[int, ...]], n: int):
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return tuple(value for _ in range(n))