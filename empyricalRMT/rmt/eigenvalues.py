import numpy as np
import pandas as pd

from numpy import ndarray
from pandas import DataFrame
from pyod.models.hbos import HBOS
from typing import Iterable, List, Sized
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
from empyricalRMT.rmt.smoother import Smoother, SmoothMethod
from empyricalRMT.rmt.trim import TrimReport
from empyricalRMT.rmt.unfold import Unfolded
from empyricalRMT.utils import find_first, find_last, is_symmetric, mkdirp


_WARNED_SMALL = False


class Eigenvalues(EigVals):
    def __init__(self, eigenvalues: Sized):
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

    def get_trimmed(
        self, max_trim: float = 0.5, max_iters: int = 7, outlier_tol: float = 0.1
    ) -> TrimReport:
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

        Returns
        -------
        trimmed: Trimmed
            An object of class Trimmed, which contains various information and functions
            for evaluating the different possible trim regions.
        """
        print("Trimming to central eigenvalues.")

        eigs = self.vals
        return TrimReport(eigs, max_trim, max_iters, outlier_tol)

    def get_best_trim(
        self,
        smoother: SmoothMethod = "poly",
        degree: int = DEFAULT_POLY_DEGREE,
        outlier_tol: float = 0.1,
        max_trim: float = 0.5,
    ) -> TrimReport:
        raise NotImplementedError

    def trim_marcenko_pastur(self, series_length: int, n_series: int) -> ndarray:
        """Trim to noise eigenvalues under assumption that eigenvalues come from
        correlation matrix.

        See https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6513117/#sec008title,

        A second and related improvement takes into account the effects of common
        (nonstationary) trends for a system with N cells, and in particular the largest
        eigenvalue eig_max. We realize that the effects of noise are inseparably coupled
        to those of the global trend [51], as the presence of the latter modifies and
        left-shifts the density of eigenvalues that we would otherwise observe in presence
        of noise only. So we do not simply superimpose the two effects as in [15]; on the
        contrary, we calculate the modification of the random bulk exactly, given the
        system’s empirical eig_max. In particular, we calculate the shifted value of an
        original Wishart matrix [15] to find

        eig_+- = (1 − eig_max/N)(1 +- 1/(sqrt(Q)))**2

        where Q = T/N is the ratio between the number of time steps in the data T and the
        number of cells N. Fig 2 shows both the modified and unmodified spectral
        densities. It also shows that taking the left-shift of the random bulk into
        account is very important, as it unveils informative empirical eigenvalues that
        would otherwise be classified as consistent with the random spectrum and hence
        discarded.
        """
        # https://en.wikipedia.org/wiki/Marchenko%E2%80%93Pastur_distribution
        if len(self.vals) != n_series:
            raise ValueError("There should be as many eigenvalues as series.")
        N, T = n_series, series_length
        eig_max = self.vals.max()
        trim_max = (1 - eig_max / N) * (1 + np.sqrt(N / T)) ** 2
        trim_min = (1 - eig_max / N) * (1 - np.sqrt(N / T)) ** 2
        return self.vals[(self.vals > trim_min) & (self.vals < trim_max)]

    def trim_manually(self, start: int, end: int) -> TrimReport:
        """trim sorted eigenvalues to [start:end), e.g. [eigs[start], ..., eigs[end-1]]"""
        trimmed_eigs = self.vals[start:end]
        raise NotImplementedError("Still need to implement `Trimmed` constructor")

    def trim_interactively(self) -> None:
        raise NotImplementedError

    def trim_unfold_best(
        self,
        poly_degrees: List[int] = DEFAULT_POLY_DEGREES,
        spline_smooths: List[float] = DEFAULT_SPLINE_SMOOTHS,
        spline_degrees: List[int] = DEFAULT_SPLINE_DEGREES,
    ) -> Unfolded:
        """Exhaustively trim and unfold for various smoothers, and select the "best" overall trim
        percent and smoother according to GOE score.

        Parameters
        ----------
        poly_degrees: List[int]
            the polynomial degrees for which to compute fits. Default [3, 4, 5, 6, 7, 8, 9, 10, 11]
        spline_smooths: List[float]
            the smoothing factors passed into scipy.interpolate.UnivariateSpline fits.
            Default np.linspace(1, 2, num=11)
        spline_degrees: List[int]
            A list of ints determining the degrees of scipy.interpolate.UnivariateSpline
            fits. Default [3]
        """
        raise NotImplementedError

    def unfold(self) -> Unfolded:
        raise NotImplementedError
