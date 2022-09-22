from typing import Union, cast

import numpy as np
from numba import njit, prange
from numpy import ndarray

from empyricalRMT._types import fArr, iArr


def step_values(eigs: fArr, x: Union[float, fArr]) -> Union[float, iArr]:
    """For eigenvalues `eigs`, compute the values of the step function for values x.
    That is, compute the number of eigenvalues <= x_i, for each x_i in x.

    Parameters
    ----------
    eigs: fArr
        The eigenvalues from which to compute the step function.

    x: float | fArr
        Value or array of values for which to compute the step function.

    Returns
    -------
    vals: fArr | float
        If x is a float, return a float. If x is a numpy array, return a numpy
        array of the same length as x.
    """
    if isinstance(x, float):
        return float(np.sum(eigs <= x))
    if not isinstance(x, ndarray):
        raise ValueError("`x` must be either a float or 1-dimensional numpy array.")
    return cast(iArr, _step_function_fast(eigs, x))


@njit(fastmath=True, cache=False)  # type: ignore[misc]
def _step_function_fast(eigs: fArr, x: fArr) -> iArr:
    """optimized version that does not repeatedly call np.sum(eigs <= x), since
    this function needed to be called extensively in rigidity calculation."""
    ret = np.zeros((len(x)), dtype=np.int64)
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


@njit(fastmath=True, cache=False, parallel=True)
def _step_function_correct(eigs: fArr, x: fArr) -> fArr:
    """Intended for testing _step_function_fast correctness, as this function
    is for sure correct, just slow.
    """
    ret = np.empty((len(x)), dtype=np.float64)
    for i in prange(len(x)):
        ret[i] = np.sum(eigs <= x[i])
    return ret
