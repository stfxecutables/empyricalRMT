import numpy as np
from numba import njit

from empyricalRMT._types import fArr


@njit(fastmath=True)
def derivative(x: fArr, y: fArr) -> fArr:
    res = np.empty(x.shape, dtype=np.float64)
    # for i = 1 (i.e. y[1], we compute (y[0] - y[2]) / 2*spacing)
    # ...
    # for i = L - 2, we compute (y[L-3] - y[L-1]) / 2*spacing
    # i.e. (y[0:L-2] - y[2:]) / 2*spacing
    L = len(x)
    res[1:-1] = (y[2:] - y[0 : L - 2]) / (x[2:] - x[0 : L - 2])
    res[0] = (y[1] - y[0]) / (x[1] - x[0])
    res[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
    return res


@njit(fastmath=True)
def inverse_gompertz(x: fArr, a: float, b: float, c: float) -> fArr:
    """
    Parameters
    ----------
    x: ndarray
        Values over which to compute the inverse gompertz.

    a: float
        asymptote, should be close to max(eigs)

    b: float
        center, must be greater than zero


    Returns
    -------
    results: ndarray
        computed values
    """
    # return np.log(b / np.log(1 / t)) / c
    # return np.log((np.log(x/ a) - np.log(a)) / -b2) / np.log(b3)
    return np.log(np.log(b) - np.log(np.log(-x / a))) / c  # type: ignore


@njit(fastmath=True)
def gompertz(x: fArr, a: float, b: float, c: float) -> fArr:
    return a * np.exp(-b * np.exp(-c * x))  # type: ignore


@njit(fastmath=True)
def exponential(x: fArr, a: float, b: float, c: float, d: float) -> fArr:
    """b is intercept """
    return b - a * np.exp(-c * x ** (1 / d))  # type: ignore
