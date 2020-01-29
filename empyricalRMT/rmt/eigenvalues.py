import numpy as np
import pandas as pd

from numpy import ndarray
from pandas import DataFrame
from pyod.models.hbos import HBOS
from warnings import warn

from empyricalRMT.rmt._constants import (
    EXPECTED_GOE_MEAN,
    EXPECTED_GOE_VARIANCE,
    DEFAULT_POLY_DEGREE,
    DEFAULT_SPLINE_SMOOTH,
    DEFAULT_SPLINE_DEGREE,
    DEFAULT_POLY_DEGREES,
    DEFAULT_SPLINE_SMOOTHS,
    DEFAULT_SPLINE_DEGREES,
)
from empyricalRMT.rmt._eigvals import EigVals
from empyricalRMT.rmt.observables.step import stepFunctionVectorized
from empyricalRMT.rmt.smoother import Smoother
from empyricalRMT.utils import find_first, find_last, is_symmetric, mkdirp


_WARNED_SMALL = False


class Trimmed(EigVals):
    def __init__(self, eigenvalues: ndarray, trimmed: ndarray):
        super().__init__(eigenvalues)
        self._trim_indices = None
        self._trim_report = None
        self._vals = trimmed

    @property
    def values(self) -> ndarray:
        return self._vals

    @property
    def vals(self) -> ndarray:
        return self._vals

    @property
    def trim_indices(self) -> (int, int):
        raise NotImplementedError
        return self._trim_indices

    @property
    def trim_report(self) -> DataFrame:
        raise NotImplementedError
        return self._trim_report

    def plot_trimmed(self):
        raise NotImplementedError

    def unfold(self) -> Unfolded:
        raise NotImplementedError
        return


class Eigenvalues(EigVals):
    def __init__(self, eigenvalues):
        """Construct an Eigenvalues object.

        Parameters
        ----------
        eigs: array_like
            a list, numpy array, or other iterable of the computed eigenvalues
            of some matrix
        """
        global _WARNED_SMALL
        if eigenvalues is None:
            raise ValueError("`eigenvalues` must be an array_like.")
        try:
            length = len(eigenvalues)
            if length < 50 and not _WARNED_SMALL:
                warn(
                    "You have less than 50 eigenvalues, and the assumptions of Random "
                    "Matrix Theory are almost certainly not justified. Any results "
                    "obtained should be interpreted with caution",
                    category=UserWarning,
                )
                _WARNED_SMALL = True  # don't warn more than once per execution
        except TypeError:
            raise ValueError(
                "The `eigs` passed to unfolded must be an object with a defined length via `len()`."
            )

        super().__init__(eigenvalues)

    @property
    def values(self) -> ndarray:
        return self._vals

    @property
    def vals(self) -> ndarray:
        return self._vals

    def trim(self, outlier_tol=0.1, max_trim=0.5) -> Trimmed:
        """compute the optimal trim regions iteratively via histogram-based outlier detection

        Parameters
        ----------
        outlier_tol: float
            A float between 0 and 1. Determines the tolerance paramater for
            [HBOS](https://pyod.readthedocs.io/en/latest/pyod.models.html#module-pyod.models.hbos)
            histogram-based outlier detection
        max_trim: float
            A float between 0 and 1 of the maximum allowable proportion of eigenvalues
            that can be trimmed.
        """
        print("Trimming to central eigenvalues.")

        eigs = self.vals
        trimmed_steps = self.__collect_outliers(eigs, outlier_tol, max_trim)
        raise NotImplementedError
        return trimmed_steps

    def trim_manually(self, start, end) -> Trimmed:
        raise NotImplementedError

    def trim_interactively(self) -> Trimmed:
        raise NotImplementedError

    def trim_unfold(self) -> Unfolded:
        raise NotImplementedError

    def unfold(self) -> Unfolded:
        raise NotImplementedError

    def __collect_outliers(
        self, tolerance=0.1, max_trim=0.5, max_iters=5
    ) -> [DataFrame]:
        """Iteratively perform histogram-based outlier detection until reaching
        either max_trim or max_iters, saving outliers identified at each step.

        Paramaters
        ----------
        tolerance: float
            tolerance level for HBOS
        max_trim: float
            Value in (0,1) representing the maximum allowable proportion of eigenvalues
            trimmed.
        max_iters: int
            Maximum number of iterations (times) to perform HBOS outlier detection.

        Returns
        -------
        trim_iterations: [DataFrame]
            A list of pandas DataFrames with structure:
            ```
            {
                "eigs": np.array,
                "steps": np.array,
                "unfolded": np.array,
                "cluster": ["inlier" | "outlier"]
            }
            ```
            such that trim_iterations[0] is the original values without trimming, and
            trim_iterations[i] is a DataFrame of the eigenvalues, step function values,
            unfolded values, and inlier/outlier labels at iteration `i`.
        """
        # zeroth iteration is just the full set of values, none considered outliers
        eigs = self.vals
        steps = stepFunctionVectorized(eigs, eigs)
        iter_results = [
            pd.DataFrame(
                {
                    "eigs": eigs,
                    "steps": steps,
                    "unfolded": self.__fit(eigs, degree=DEFAULT_POLY_DEGREE)[0],
                    "cluster": ["inlier" for _ in eigs],
                }
            )
        ]
        # terminate if we have trimmed max_trim
        iters = 0
        while (len(iter_results[-1]) / len(eigs)) > max_trim and (iters < max_iters):
            iters += 1
            # because eigs are sorted, HBOS will always identify outliers at one of the
            # two ends of the eigenvalues, which is what we want
            df = iter_results[-1].copy(deep=True)
            df = df[df["cluster"] == "inlier"]
            hb = HBOS(tol=tolerance)
            is_outlier = np.array(
                hb.fit(df[["eigs", "steps"]]).labels_, dtype=bool
            )  # outliers get "1"

            # check we haven't removed middle values:
            if is_outlier[0]:
                start = find_first(is_outlier, False)
                for i in range(start, len(is_outlier)):
                    is_outlier[i] = False
            if is_outlier[-1]:
                stop = find_last(is_outlier, False)
                for i in range(stop):
                    is_outlier[i] = False
            if not is_outlier[0] and not is_outlier[-1]:  # force a break later
                is_outlier = np.zeros(is_outlier.shape, dtype=bool)

            df["cluster"] = ["outlier" if label else "inlier" for label in is_outlier]
            df["unfolded"], _ = self.__fit(
                np.array(df["eigs"]), degree=DEFAULT_POLY_DEGREE
            )
            iter_results.append(df)
            if np.alltrue(~is_outlier):
                break

        return iter_results
