from typing import Any, cast

import numpy as np
from numpy import float64 as f64

from empyricalRMT._types import fArr


def make_1d_array(array: Any) -> fArr:
    try:
        arr = np.array(array, dtype=f64)
    except Exception as e:
        raise ValueError(
            f"Could not convert passed in values: {array} to numpy NDArray[float64]."
        ) from e
    flat = arr.ravel()
    if len(flat) != len(arr):
        raise ValueError("Could not convert passed in value to 1-dimensional array.")
    return cast(fArr, np.copy(flat))
