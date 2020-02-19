import numpy as np
from numpy import ndarray

from typing import Any


def make_1d_array(array: Any) -> ndarray:
    try:
        arr = np.array(array, dtype=float)
    except BaseException as e:
        raise ValueError("Could not convert passed in value to numpy ndarray.") from e

    flat = arr.ravel()
    if len(flat) != len(arr):
        raise ValueError("Could not convert passed in value to 1-dimensional array.")

    return np.copy(flat)
