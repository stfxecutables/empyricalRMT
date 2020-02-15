import numpy as np

from numba import jit, prange
from typing import Any


def get_signals(arr2d: np.array, threshold: float = 0.01) -> np.array:
    if len(arr2d.shape) != 2:
        raise Exception("get_signals() requires a 2D array of the time-series")

    mask = np.ones(arr2d.shape[0], dtype=np.int8)
    fill_mask(mask, arr2d, threshold)
    # print("mask")
    # print(mask)
    # make array large enough to hold just the actual signals
    arr = arr2d[mask == 1]
    return arr


@jit(nopython=True, parallel=True, fastmath=True)
def fill_mask(mask: Any, arr2d: Any, threshold: Any) -> Any:
    """Remove voxels that will cause undefined correlations"""
    for i in prange(mask.shape[0]):
        if np.var(arr2d[i]) <= threshold:
            mask[i] = 0
        if np.linalg.norm(arr2d[i]) <= threshold:
            mask[i] = 0


# @jit(nopython=True, fastmath=True, cache=True)
# def fill_arr_from_mask(arr, arr2d, mask):
#     current = 0
#     for i in range(arr.shape[0]):
#         if mask[i] == 0:
#             arr[current, :] = arr2d[i, :]
#             current += 1
