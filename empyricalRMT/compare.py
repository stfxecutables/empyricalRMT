from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from numba import njit
from numpy import ndarray
from pandas import DataFrame
from typing_extensions import Literal

from empyricalRMT._types import fArr
from empyricalRMT._validate import make_1d_array

Metric = Literal["mad", "msqd", "corr"]


class Compare:
    """A helper class for implementing various curve comparison methods."""

    def __init__(
        self,
        curves: List[fArr],
        labels: List[str],
        base_curve: Optional[fArr] = None,
        base_label: Optional[str] = None,
    ):
        """Construct a Compare object for accessing various comparison methods.

        Parameters
        ----------
        curves: List[NDArray[floating]]
            A list of unidimensional numpy arrays of values to compare. For most
            comparison methods besides some piecewise / quantile comparison methods, the
            curves must have identical lengths.

        labels: List[str]
            A list of strings identifying each curve. Must be the same length as
            curves, and labels[i] must be the label for curves[i], for all valid
            values of i.

        base_curve: NDArray[floating]
            The base curve against which each curve of `curves` will be compared, if the
            desire is to compare multiple curves only to one single curve.

        base_label: str
            The label for identifying the base_curve.
        """
        self.curves: List[fArr] = [make_1d_array(curve) for curve in curves]
        self.labels = labels.copy()
        self.base_curve: Optional[fArr] = (
            make_1d_array(base_curve) if base_curve is not None else None
        )
        self.base_label = base_label  # don't need to copy strings in Python
        self.__validate_curve_lengths()
        self.dict = dict(zip(self.labels, self.curves))

    def correlate(self) -> DataFrame:
        """Return the grid of correlations across curves. """
        self.__validate_curve_lengths(
            message="Comparing via correlation requires all curves have identical lengths",
            check_all_equal=True,
        )
        if self.base_curve is not None:
            # index with [0, 1:], since [0, :] give first row of correlations, and since
            # [0, 0] is just the correlation of the base_curve with itself
            data = np.corrcoef(self.base_curve, self.curves)[0, 1:]
            return pd.DataFrame(data=data, index=self.labels, columns=[self.base_label])
        data = np.corrcoef(self.curves)
        return pd.DataFrame(data=data, index=self.labels, columns=self.labels)

    def mean_sq_difference(self) -> DataFrame:
        """Return the grid of mean square differences across curves."""
        self.__validate_curve_lengths(
            message=(
                "Comparing via mean squared differences requires "
                "all curves have identical lengths"
            ),
            check_all_equal=True,
        )
        curves = np.array(self.curves)
        if self.base_curve is not None:
            diffs = np.empty(curves.shape[0])
            for i in range(len(diffs)):
                diffs[i] = np.mean((self.base_curve - curves[i]) ** 2)
                return pd.DataFrame(data=diffs, index=self.labels, columns=[self.base_label])
        data = self.__fast_msqd(curves)
        return pd.DataFrame(data=data, index=self.labels, columns=self.labels)

    def mean_abs_difference(self) -> DataFrame:
        """Return the grid of mean absolute differences across curves."""
        self.__validate_curve_lengths(
            message=(
                "Comparing via mean squared differences requires "
                "all curves have identical lengths"
            ),
            check_all_equal=True,
        )
        curves = np.array(self.curves)
        if self.base_curve is not None:
            diffs = np.empty(curves.shape[0])
            for i in range(len(diffs)):
                diffs[i] = np.mean(np.abs(self.base_curve - curves[i]))
                return pd.DataFrame(data=diffs, index=self.labels, columns=[self.base_label])
        data = self.__fast_mad(curves)
        return pd.DataFrame(data=data, index=self.labels, columns=self.labels)

    def _test_validate(self, **kwargs: Any) -> None:
        self.__validate_curve_lengths(**kwargs)

    @staticmethod
    @njit(fastmath=True)
    def __fast_msqd(curves: ndarray) -> ndarray:
        n = curves.shape[0]
        data = np.empty((n, n), dtype=np.float64)
        for j in range(n):
            for i in range(n):
                data[i, j] = np.mean((curves[i] - curves[j]) ** 2)
        return data

    @staticmethod
    @njit(fastmath=True)
    def __fast_mad(curves: ndarray) -> ndarray:
        n = curves.shape[0]
        data = np.empty((n, n), dtype=np.float64)
        for j in range(n):
            for i in range(n):
                data[i, j] = np.mean(np.abs(curves[i] - curves[j]))
        return data

    @staticmethod
    def __histograms(
        curve1: ndarray, curve2: ndarray, n_bins: int = 10
    ) -> Tuple[ndarray, ndarray, ndarray]:
        """Compute a histogram over [min(curve1, curve2), max(curve1, curve2)].

        Returns
        -------
        counts1: ndarray
            The bin counts for curve1.

        counts2: ndarray
            The bin counts for curve2.

        endpoints: ndarray
            The (sorted) ndarray of bin endpoints.
        """
        vals1 = np.sort(curve1)
        vals2 = np.sort(curve2)
        endpoints = np.linspace(min(vals1[0], vals2[0]), max(vals1[-1], vals2[-1]), n_bins + 1)
        n, counts1, counts2 = 0, np.arange(n_bins), np.arange(n_bins)
        for val in vals1:
            if val < endpoints[n]:
                counts1[n] += 1
            else:
                n += 1
            if n >= len(counts1):
                raise RuntimeError("Problem with hist algorithm. Should be impossible.")
        n = 0
        for val in vals2:
            if val < endpoints[n]:
                counts2[n] += 1
            else:
                n += 1
            if n >= len(counts2):
                raise RuntimeError("Problem with hist algorithm. Should be impossible.")

        return endpoints, counts1, counts2

    def __validate_curve_lengths(
        self, message: Optional[str] = None, check_all_equal: bool = False
    ) -> None:
        """Ensure curve lengths are appropriate for desired comparison methods."""
        curves = self.curves
        labels = self.labels

        if len(curves) < 1:
            raise ValueError("There must be more than one curve to compare.")
        if len(curves) == 1 and self.base_curve is None:
            raise ValueError("There must be more than one curve to compare to the base curve.")
        if len(self.curves) != len(labels):
            raise ValueError("`labels` must have the same length as `curves`.")

        all_equal = np.all([len(curve) == len(curves[0]) for curve in curves])
        if check_all_equal:
            if self.base_curve is not None and self.base_label is not None:
                if len(curves[0]) != len(self.base_curve):
                    raise ValueError(message)
            if not all_equal:
                raise ValueError(message)
