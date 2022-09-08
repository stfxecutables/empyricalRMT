from typing import cast

import numpy as np
from numba import njit, prange

from empyricalRMT._types import bArr, fArr


def get_signals(arr2d: fArr, threshold: float = 0.01) -> fArr:
    if len(arr2d.shape) != 2:
        raise Exception("get_signals() requires a 2D array of the time-series")

    mask = np.ones(arr2d.shape[0], dtype=np.bool_)
    fill_mask(mask, arr2d, threshold)
    # print("mask")
    # print(mask)
    # make array large enough to hold just the actual signals
    arr = arr2d[mask == 1]
    return cast(fArr, arr)


@njit(parallel=True, fastmath=True)
def fill_mask(mask: bArr, arr2d: fArr, threshold: float) -> None:
    """Remove voxels that will cause undefined correlations"""
    for i in prange(mask.shape[0]):
        if np.var(arr2d[i]) <= threshold:
            mask[i] = 0
        if np.linalg.norm(arr2d[i]) <= threshold:
            mask[i] = 0
