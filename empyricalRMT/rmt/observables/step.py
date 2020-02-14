import numpy as np
from numpy import ndarray

from numba import jit, prange
from typing import Any


# slow, but guaranteed to be correct
@jit(nopython=True, fastmath=True, cache=True, parallel=True)
def stepFunctionCorrect(eigs: ndarray, x: ndarray) -> ndarray:
    ret = np.empty((len(x)), dtype=np.float64)
    for i in prange(len(x)):
        ret[i] = np.sum(eigs <= x[i])
    return ret


# this function is equivalent to `return len(eigs[eigs <= x])`
# in particular, since our eigenvalues are always sorted, we can simply
# return the index of the eigenvalue in eigs, e.g. if we have eigs
#   [0.1, 0.2, 0.3, 0.4]
# then stepFunctionG(eigs, 0.20) is
# this function could be improved with binary search and memoization if
# necessary or if it becomes a bottleneck
@jit(nopython=True, fastmath=True, cache=True)
def stepFunctionG(eigs: ndarray, x: float) -> int:
    """Count the number of eigenvalues <= x."""
    cumulative = 0
    for eig in eigs:
        if x <= eig:
            break
        else:
            cumulative += 1
    return cumulative


# Currently this is extremely slow since we don't use the fact that
# `eigs` and `x` will be sorted for our use cases.
@jit(nopython=True, fastmath=True, cache=True)
def stepFunctionVectorized(eigs: ndarray, x: ndarray) -> ndarray:
    ret = np.empty((len(x)), dtype=np.float64)
    for i in prange(len(x)):
        ret[i] = stepFunctionG(eigs, x[i])
    return ret


@jit(nopython=True, fastmath=True, cache=True)
def stepFunctionFast(eigs: ndarray, x: ndarray) -> ndarray:
    ret = np.zeros((len(x)), dtype=np.float64)
    if x[-1] <= eigs[0]:  # early return if all values are just zero
        return ret
    if x[0] > eigs[-1]:  # early return if all x values above eigs
        n = len(eigs)
        for i in range(len(ret)):
            ret[i] = n
        return ret

    # find the first index of x where we hit values of x that are actually
    # within the range of eigs[0], eigs[-1]
    j = 0  # index into x
    while j < len(x) and x[j] < eigs[0]:
        j += 1
    # now j is the index of the first x value with a nonzero step function value

    i = 0  # index into eigs
    count = 0
    while j < len(x) and i < len(eigs):
        if x[j] >= eigs[i]:  # x could start in middle of eigs
            i += 1
            count += 1
            continue
        while j < len(x) and x[j] < eigs[i]:  # look ahead
            ret[j] = count
            j += 1

    while j < len(x):
        ret[j] = count
        j += 1

    return ret


@jit(nopython=True)
def find(arr: np.array, value: Any) -> int:
    for i, val in enumerate(arr):
        if val >= value:
            return i
    return -1
