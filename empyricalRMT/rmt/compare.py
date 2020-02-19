import numpy as np
import pandas as pd

from numpy import ndarray
from pandas import DataFrame
from typing import List

from empyricalRMT._validate import make_1d_array


class Compare:
    def __init__(self, curves: List[ndarray], labels: List[str]):
        """Construct a Compare object for accessing various comparison methods.

        Parameters
        ----------
        curves: List[ndarray]
            A list of unidimensional numpy arrays of values to compare. For most
            comparison methods besides some piecewise / quantile comparison methods, the
            curves must have identical lengths.
        labels: List[str]
            A list of strings identifying each curve. Must be the same length as curves,
            and labels[i] must be the label for curves[i], for all valid values of i.
        """
        self.curves = [make_1d_array(curve) for curve in curves]
        self.labels = labels.copy()
        self.__validate_curve_lengths()
        self.dict = dict(zip(self.labels, self.curves))

    def correlate(self) -> DataFrame:
        """Return the grid of correlations across curves. """
        self.__validate_curve_lengths(
            message="Comparing via correlation requires curves all have identical lengths",
            check_all_equal=True,
        )
        data = np.corrcoef(self.curves)
        return pd.DataFrame(data=data, index=self.labels, columns=self.labels)

    def __validate_curve_lengths(
        self, message: str = None, check_all_equal: bool = False
    ) -> None:
        curves = self.curves
        labels = self.labels

        if len(curves) <= 1:
            raise ValueError("There must be more than one curve to compare.")
        if len(self.curves) != len(labels):
            raise ValueError("`labels` must have the same length as `curves`.")

        all_equal = np.all([len(curve) == len(curves[0]) for curve in curves])
        if check_all_equal and not all_equal:
            raise ValueError(message)
