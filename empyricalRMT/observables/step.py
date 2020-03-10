import numpy as np
from numpy import ndarray

from numba import jit, prange
from typing import Union


def step_values(eigs: ndarray, x: Union[float, ndarray]) -> Union[float, ndarray]:
    """For eigenvalues `eigs`, compute the values of the step function for values x.
    That is, compute the number of eigenvalues <= x_i, for each x_i in x.

    Parameters
    ----------
    eigs: ndarray
        The eigenvalues from which to compute the step function.

    x: float | ndarray
        Value or array of values for which to compute the step function.

    Returns
    -------
    vals: ndarray | float
        If x is a float, return a float. If x is a numpy array, return a numpy
        array of the same length as x.
    """
    if isinstance(x, float):
        return np.sum(eigs <= x)
    if not isinstance(x, ndarray):
        raise ValueError("`x` must be either a float or 1-dimensional numpy array.")
    return _step_function_fast(eigs, x)


@jit(nopython=True, fastmath=True, cache=True)
def _step_function_fast(eigs: ndarray, x: ndarray) -> ndarray:
    """optimized version that does not repeatedly call np.sum(eigs <= x), since
    this function needed to be called extensively in rigidity calculation."""
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

    while j < len(x):  # keep going for any remaining values of x
        ret[j] = count
        j += 1

    return ret


# guaranteed to be correct
@jit(nopython=True, fastmath=True, cache=True, parallel=True)
def _step_function_correct(eigs: ndarray, x: ndarray) -> ndarray:
    """Intended primarily for testing _step_function_fast correctness"""
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
def _step_function_g(eigs: ndarray, x: float) -> int:
    """[DEPRECATE] Old version. Slow when used repeatedly"""
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
def _step_function_slow(eigs: ndarray, x: ndarray) -> ndarray:
    """[DEPRECATE] Slow (like if n == len(x), O(n**2) or worse complexity)
    version. Remains for testing only."""
    ret = np.empty((len(x)), dtype=np.float64)
    for i in prange(len(x)):
        ret[i] = _step_function_g(eigs, x[i])
    return ret
