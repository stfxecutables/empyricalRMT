import numpy as np
from numpy import ndarray

from numba import jit, prange

# this function is equivalent to `return len(eigs[eigs <= x])`
# in particular, since our eigenvalues are always sorted, we can simply
# return the index of the eigenvalue in eigs, e.g. if we have eigs
#   [0.1, 0.2, 0.3, 0.4]
# then stepFunctionG(eigs, 0.20) is
# this function could be improved with binary search and memoization if
# necessary or if it becomes a bottleneck
@jit(nopython=True, fastmath=True, cache=True)
def stepFunctionG(eigs: ndarray, x: float) -> float:
    cumulative = 0
    for eig in eigs:
        if x <= eig:
            break
        else:
            cumulative += 1
    return float(cumulative)


@jit(nopython=True, fastmath=True, cache=True)
def stepFunctionVectorized(eigs: ndarray, x: ndarray) -> ndarray:
    ret = np.empty((len(x)), dtype=np.float64)
    for i in prange(len(x)):
        ret[i] = stepFunctionG(eigs, x[i])
    return ret
