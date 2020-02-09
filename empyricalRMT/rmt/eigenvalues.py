import numpy as np
import pandas as pd

from numpy import ndarray
from pandas import DataFrame
from pyod.models.hbos import HBOS
from typing import Iterable, List, Sized, Tuple
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
from empyricalRMT.rmt.trim import Trimmed, TrimReport
from empyricalRMT.rmt.unfold import Unfolded
from empyricalRMT.utils import find_first, find_last, is_symmetric, mkdirp


_WARNED_SMALL = False


class Eigenvalues(EigVals):
    def __init__(self, eigenvalues: Sized):
        """Construct an Eigenvalues object.

        Parameters
        ----------
        eigs: Sized
            An object for which np.array(eigs) will return a sensible, one-dimensional
            array of floats which are the computed eigenvalues of some matrix.
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
        """Return the stored eigenvalues."""
        return self._vals

    @property
    def vals(self) -> ndarray:
        """Return the stored eigenvalues. Alternate for Eigenvalues.values"""
        return self._vals

    @property
    def eigenvalues(self) -> ndarray:
        """Return the stored eigenvalues. Alternate for Eigenvalues.values"""
        return self._vals

    @property
    def eigs(self) -> ndarray:
        """Return the stored eigenvalues. Alternate for Eigenvalues.values"""
        return self._vals

    def trim_report(
        self,
        max_trim: float = 0.5,
        max_iters: int = 7,
        poly_degrees: List[int] = DEFAULT_POLY_DEGREES,
        spline_smooths: List[float] = DEFAULT_SPLINE_SMOOTHS,
        spline_degrees: List[int] = DEFAULT_SPLINE_DEGREES,
        gompertz: bool = True,
        outlier_tol: float = 0.1,
    ) -> TrimReport:
        """Compute multiple trim regions iteratively via histogram-based outlier detection, perform
        unfolding for each trim region, and summarize the resultant spacings and trimmings.

        Parameters
        ----------
        max_trim: float
            Float in (0, 1). The maximum allowable portion of eigenvalues to be trimmed.
            E.g. `max_trim=0.8` means to allow up to 80% of the original eigenvalues to
            be trimmed away.
        max_iters: int
            The maximum allowable number of iterations of outlier detection to run.
            Setting `max_iters=0` will not allow any trimming / outlier detection, and so
            will simply evaluate unfolding for different smoothers on the original raw
            eigenvalues. Typically, you would want this to be >= 4, to allow for trimming
            both some of the most extreme positive and negative eigenvalues.
        poly_degrees: List[int]
            the polynomial degrees for which to compute fits. Default [3, 4, 5, 6, 7, 8, 9, 10, 11]
        spline_smooths: List[float]
            the smoothing factors passed into scipy.interpolate.UnivariateSpline fits.
            Default np.linspace(1, 2, num=11)
        spline_degrees: List[int]
            A list of ints determining the degrees of scipy.interpolate.UnivariateSpline
            fits. Default [3]
        gompertz: bool
            Whether or not to use a gompertz curve as one of the smoothers.
        outlier_tol: float
            A float between 0 and 1, and which is passed as the tolerance paramater for
            [HBOS](https://pyod.readthedocs.io/en/latest/pyod.models.html#module-pyod.models.hbos)
            histogram-based outlier detection

        Returns
        -------
        trimmed: TrimReport
            A TrimReport object, which contains various information and functions
            for evaluating the different possible trim regions.
        """
        print("Trimming to central eigenvalues.")

        eigs = self.vals
        return TrimReport(
            eigs,
            max_trim,
            max_iters,
            poly_degrees,
            spline_smooths,
            spline_degrees,
            gompertz,
            outlier_tol,
        )

    def get_best_trimmed(
        self,
        smoother: SmoothMethod = "poly",
        degree: int = DEFAULT_POLY_DEGREE,
        spline_smooth: float = DEFAULT_SPLINE_SMOOTH,
        max_iters: int = 7,
        max_trim: float = 0.5,
        outlier_tol: float = 0.1,
    ) -> Trimmed:
        """For the given smoother and smmothing and trim options, compute
        a up to `max_iters` different trim regions, and select the region
        which has an unfolding that is most GOE-like in terms of its local
        spacings.

        Parameters
        ----------
        max_trim: float
            Float in (0, 1). The maximum allowable portion of eigenvalues to be trimmed.
            E.g. `max_trim=0.8` means to allow up to 80% of the original eigenvalues to
            be trimmed away.
        max_iters: int
            The maximum allowable number of iterations of outlier detection to run.
            Setting `max_iters=0` will not allow any trimming / outlier detection, and so
            will simply evaluate unfolding for different smoothers on the original raw
            eigenvalues. Typically, you would want this to be >= 4, to allow for trimming
            both some of the most extreme positive and negative eigenvalues.
        smoother: "poly" | "spline" | "gompertz" | lambda
            the type of smoothing function used to fit the step function
        degree: int
            the degree of the polynomial or spline
        spline_smooth: float
            the smoothing factors passed into scipy.interpolate.UnivariateSpline
        outlier_tol: float
            A float between 0 and 1, and which is passed as the tolerance paramater for
            [HBOS](https://pyod.readthedocs.io/en/latest/pyod.models.html#module-pyod.models.hbos)
            histogram-based outlier detection

        Returns
        -------
        best_indices: Tuple[int, int]
            The indices (start, end) such that eigenvalues[start:end] is the trimmed
            region that is "most GOE" in terms of its nearest-neighbour level spacings.
        """
        report = None
        if smoother == "poly":
            report = TrimReport(
                self.vals, max_trim, max_iters, [degree], [], [], False, outlier_tol
            )
        elif smoother == "spline":
            report = TrimReport(
                self.vals,
                max_trim,
                max_iters,
                [],
                [spline_smooth],
                [degree],
                False,
                outlier_tol,
            )
        elif smoother == "gompertz":
            report = TrimReport(
                self.vals, max_trim, max_iters, [], [], [], True, outlier_tol
            )
        else:
            raise ValueError("Unknown smoother.")

        _, _, best_indices, _ = report.summarize_trim_unfoldings()
        start, end = best_indices[0][0], best_indices[0][1]
        return self.trim_manually(start, end)

    def trim_marcenko_pastur(self, series_length: int, n_series: int) -> Trimmed:
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
        return Trimmed(self.vals[(self.vals > trim_min) & (self.vals < trim_max)])

    def trim_manually(self, start: int, end: int) -> Trimmed:
        """trim sorted eigenvalues to [start:end), e.g. [eigs[start], ..., eigs[end-1]]"""
        trimmed_eigs = self.vals[start:end]
        return Trimmed(trimmed_eigs)

    def trim_interactively(self) -> None:
        raise NotImplementedError

    def trim_unfold_auto(
        self,
        max_trim: float = 0.5,
        max_iters: int = 7,
        poly_degrees: List[int] = DEFAULT_POLY_DEGREES,
        spline_smooths: List[float] = DEFAULT_SPLINE_SMOOTHS,
        spline_degrees: List[int] = DEFAULT_SPLINE_DEGREES,
        gompertz: bool = True,
        outlier_tol: float = 0.1,
    ) -> Unfolded:
        """Exhaustively compare mutliple trim regions and smoothers based on their "GOE score"
        and unfold the eigenvalues, using the trim region and smoothing parameters
        determined to be "most GOE" based on the exhaustive process.

        Summary of the automatic trim-unfold process:

        1. Compute multiple natural trim regions via histrogram-based outlier detection.
        2. For each trim region, fit all possible smoothers (i.e., smoothers + smoother
           parameters) and generate a set of unfolded eigenvalues.
        3. For each set of unfolded eigenvalues, compute the GOE score. The GOE score
           indexes how much the mean and variance of the spacings of the unfolded values
           differ from the expected spacing variance and mean for the unfolding of a GOE
           matrix.
        4. Assume that the choice of smoother should determine the optimal trim region,
           and not the converse. This is most consistent with the concern over smoothing
           choices in the literature, and acknowledges that trimming is done *so that*
           smoothers yield more accurate results.
        5. Assume that the best smoother is the one which results in the most GOE-like
           spacing distribution across all trims and all smoothers
        6.

        Exhaustively trim and unfold for various smoothers, and select the "best" overall trim
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
