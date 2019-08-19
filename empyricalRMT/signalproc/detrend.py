import numpy as np

from numba import jit, prange
from PyEMD import EMD
from scipy.stats import linregress

from utils import slope, intercept


class Detrend:
    def __init__(self):
        pass

    def linear(series: np.array) -> np.array:
        """Remove the linear trend by fitting a linear model, and returning
        the residuals"""
        time = np.arange(0, len(series))
        m, b = linregress(time, series)  # m == slope, b == intercept
        fitted = m * time + b
        return series - fitted

    def emd(series: np.array) -> np.array:
        """Remove the lowest-frequency trend as determined by Empirical
        Mode Decomposition """
        trend = EMD().emd(series)[-1]
        return series - trend

    def difference(series: np.array) -> np.array:
        """Remove non-stationarity by differencing the data (once)"""
        differenced = np.empty([len(series - 1)])
        for i in range(len(series) - 1):
            differenced[i] = series[i + 1] - series[i]
        return differenced


@jit(nopython=True, parallel=True, fastmath=True)
def linear_detrend(signals: np.array, ret: np.array) -> np.array:
    """takes voxels with nonzero variance"""
    m, T = signals.shape
    x = np.arange(0, T)
    for i in prange(m):
        y = signals[i, :]
        a = slope(x, y)
        b = intercept(x, y, a)
        fitted = m * x + b
        detrended = y - fitted
        ret[i, :] = detrended
    return ret


@jit(nopython=True, parallel=True, fastmath=True)
def mean_detrend(signals: np.array, ret: np.array) -> np.array:
    """takes voxels with nonzero variance"""
    m, T = signals.shape
    for i in prange(m):
        ret[i, :] = signals[i, :] - np.mean(signals[i, :])
    return ret
