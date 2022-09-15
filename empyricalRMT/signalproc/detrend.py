from __future__ import annotations

from enum import Enum
from typing import Literal, cast

import numpy as np
from numba import njit, prange
from numpy import float64 as f64
from numpy.polynomial import Polynomial
from numpy.typing import NDArray
from PyEMD import EMD
from scipy.optimize import curve_fit
from scipy.stats import linregress

from empyricalRMT._types import fArr
from empyricalRMT.exponentials import gompertz, inverse_gompertz
from empyricalRMT.utils import intercept, slope


class DetrendMethod(Enum):
    EMD = "emd"
    Linear = "linear"
    Quadratic = "quadratic"
    # Exponential = "exp"
    # Gompertz = "gompertz"
    # Difference = "diff"

    @classmethod
    def validate(cls, s: str | DetrendMethod) -> DetrendMethod:
        try:
            if isinstance(s, str):
                return cls(s)
            return s
        except Exception as e:
            values = [e.value for e in cls]
            raise ValueError(f"{cls.__name__} must be one of {values}") from e


_DetrendMethod = Literal["emd", "linear", "quad", "exp", "diff"]


def detrend(eigs: fArr, method: DetrendMethod | _DetrendMethod) -> fArr:
    method = DetrendMethod.validate(method)
    time = np.arange(0, len(eigs))
    trend: fArr
    if method is DetrendMethod.EMD:
        trend = np.array(EMD().emd(eigs)[-1], dtype=f64)
    elif method is DetrendMethod.Linear:
        m, b = linregress(time, eigs)[:2]  # m == slope, b == intercept
        trend = m * time + b
    elif method is DetrendMethod.Quadratic:
        trend = Polynomial.fit(time, eigs, deg=2)(time)
    # elif method is DetrendMethod.Exponential:
    #     func = lambda t, a, b: a * np.exp(-b * t)  # type: ignore
    #     [a, b], cov = curve_fit(
    #         func,
    #         time,
    #         eigs,
    #         p0=(1.0, 0.5),
    #     )
    #     trend = func(eigs, a, b)  # type: ignore
    # elif method is DetrendMethod.Gompertz:
    #     [a, b, c], cov = curve_fit(inverse_gompertz, time, eigs, p0=(eigs[-1], 1, 1))
    #     trend = inverse_gompertz(eigs, a, b, c)
    # elif method is DetrendMethod.Difference:
    #     return np.diff(eigs)
    return eigs - trend


class Detrend:
    def __init__(self) -> None:
        return

    def linear(self, series: NDArray) -> fArr:
        """Remove the linear trend by fitting a linear model, and returning
        the residuals"""
        time = np.arange(0, len(series))
        m, b = linregress(time, series)  # m == slope, b == intercept
        fitted = m * time + b
        return cast(fArr, series - fitted)

    def emd(self, series: NDArray) -> fArr:
        """Remove the lowest-frequency trend as determined by Empirical
        Mode Decomposition"""
        trend = EMD().emd(series)[-1]
        return cast(fArr, series - trend)

    def difference(self, series: NDArray) -> fArr:
        """Remove non-stationarity by differencing the data (once)"""
        differenced = np.empty([len(series - 1)])
        for i in range(len(series) - 1):
            differenced[i] = series[i + 1] - series[i]
        return differenced


@njit(parallel=True, fastmath=True)
def linear_detrend(signals: NDArray, ret: fArr) -> None:
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


@njit(parallel=True, fastmath=True)
def mean_detrend(signals: NDArray, ret: fArr) -> None:
    """takes voxels with nonzero variance"""
    m, T = signals.shape
    for i in prange(m):
        ret[i, :] = signals[i, :] - np.mean(signals[i, :])
