import numpy as np
import pandas as pd

from numpy import ndarray
from pandas import DataFrame
from pathlib import Path
from statsmodels.nonparametric.kde import KDEUnivariate as KDE
from typing import Any, List, Optional, Tuple, Type
from typing_extensions import Literal

import empyricalRMT.rmt.plot as plot

from empyricalRMT.rmt._eigvals import EigVals
from empyricalRMT.rmt.compare import Metric, Compare
from empyricalRMT.rmt.ensemble import Ensemble
from empyricalRMT.rmt.observables.levelvariance import level_number_variance_stable
from empyricalRMT.rmt.observables.rigidity import spectral_rigidity
from empyricalRMT.rmt.plot import (
    _next_spacings,
    _spacings as _plot_spacings,
    _unfolded_fit,
    PlotMode,
    PlotResult,
)
from empyricalRMT._validate import make_1d_array

Observables = Literal["nnsd", "nnnsd", "rigidity", "levelvar"]


class Unfolded(EigVals):
    def __init__(self, originals: ndarray, unfolded: ndarray):
        super().__init__(originals)
        self._vals = np.array(unfolded)

    @property
    def values(self) -> ndarray:
        return self._vals

    @property
    def vals(self) -> ndarray:
        return self._vals

    def spectral_rigidity(
        self,
        min_L: float = 2,
        max_L: float = 50,
        L_grid_size: int = None,
        c_iters: int = 50000,
        integration: Literal["simps", "trapz"] = "simps",
        show_progress: bool = False,
    ) -> DataFrame:
        """Compute and the spectral rigidity.

        Parameters
        ----------
        min_L: int
            The lowest possible L value for which to compute the spectral
            rigidity. Default 2.
        max_L: int = 20
            The largest possible L value for which to compute the spectral
            rigidity.
        L_grid_size: int
            The number of values of L to generate betwen min_L and max_L. Default
            2 * (max_L - min_L).
        c_iters: int = 50
            How many times the location of the center, c, of the interval
            [c - L/2, c + L/2] should be chosen uniformly at random for
            each L in order to compute the estimate of the spectral
            rigidity. Not a particularly significant effect on performance.
        show_progress: bool
            Whether or not to display computation progress in stdout.

        Returns
        -------
        df: DataFrame
            DataFrame with columns "L" and "delta" where df["L"] contains The L values
            generated based on the values of L_grid_size, min_L, and max_L, and where
            df["delta"] contains the computed spectral rigidity values for each of L.
        """
        unfolded = self.values
        L, delta = spectral_rigidity(
            unfolded,
            c_iters=c_iters,
            L_grid_size=L_grid_size,
            min_L=min_L,
            max_L=max_L,
            integration=integration,
            show_progress=show_progress,
        )
        return pd.DataFrame({"L": L, "delta": delta})

    def level_variance(
        self,
        L: ndarray = np.arange(0.5, 20, 0.2),
        tol: float = 0.01,
        max_L_iters: int = 50000,
        min_L_iters: int = 1000,
        show_progress: bool = False,
    ) -> DataFrame:
        """Compute the level number variance of the current unfolded eigenvalues.

        Parameters
        ----------
        L: ndarray
            The grid of L values for which to compute the level variance.
        tol: float
            Stop iterating when the last `min_L_iters` computed values of the
            level variance have a range (i.e. max - min) < tol.
        max_L_iters: int
            Stop computing values for the level variance once max_L_iters values
            have been computed for each L value.
        min_L_iters: int
            Minimum number of iterations for each L value.
        show_progress: bool
            Whether or not to display computation progress in stdout.

        Returns
        -------
        df: DataFrame
            A dataframe with columns "L", the L values generated based on the
            input arguments, and "sigma", the computed level variances for each
            value of L.

        Notes
        -----
        Computes the level number variance by randomly selecting a point c in
        the interval [unfolded.min(), unfolded.max()], and counts the number
        of unfolded eigenvalues in (c - L/2, c + L/2) to determine a value for
        the level number variance, sigma(L). The process is repeated until the
        running averages stabilize, and the final running average is returned.
        """
        unfolded = self.values
        L, sigma = level_number_variance_stable(
            unfolded=unfolded,
            L=L,
            tol=tol,
            max_L_iters=max_L_iters,
            min_L_iters=min_L_iters,
            show_progress=show_progress,
        )
        return DataFrame({"L": L, "sigma": sigma})

    def ensemble_compare(
        self,
        ensemble: Type[Ensemble],
        observables: List[Observables] = ["nnsd", "nnnsd", "rigidity", "levelvar"],
        metrics: List[Metric] = ["msqd"],
        spacings: Tuple[float, float] = (0.5, 2.5),
        kde_gridsize: int = 5000,
        L_rigidity: ndarray = np.arange(2, 50, 0.2),
        L_levelvar: ndarray = np.arange(1, 20, 0.5),
        show_progress: bool = False,
    ) -> pd.DataFrame:
        """Compute various spectral observables for the unfolded eigenvalues
        and compare those computed observables to the those expected for a GOE
        ensemble to get indices of similarity.

        Parameters
        ----------
        ensemble: Ensemble
            The ensemble (Poisson / GDE, GUE, GOE, GSE) against which to compare
            the unfolded eigenvalues.
        observables: List[Observables]
            The observables to use for comparison.
        metrics: List["msqd" | "mad" | "corr"]
            The metrics (used here non-rigorously) used to compute the
            similiarities. Histograms will be compared via their respective
            kernel density estimates.
        spacings: (float, float)
            The range (min, max) spacing values to use for comparing kernel
            density estimates to the expected goe density. Default (0.5, 2.5).
        L_rigidity: ndarray
            The values of L for which to calculate and compare the spectral
            rigidity.
        L_levelvar: ndarray
            The values of L for which to calculate and compare the level number
            variance.

        Returns
        -------
        similarities: DataFrame
            The similarities for each value of `observables` for for each
            metric. The metrics are the index, and the columns are the
            observables, i.e. the DataFrame is metrics x observerables.


        Notes
        -----
        The kernel density estimate will use the full set of unfolded
        eigenvalues, but smoothing assumptions means KDE(s), the kernel density
        estimate of p(s) (the spacings density), will be quite inaccurate as
        s -> 0. So e.g. for a Poisson / Gaussian Diagonal Ensemble of size
        N <= 5000, KDE(s) for s <= 0.5 will quite often be closer to the
        expected density for GOE matrices than it will be to GDE matrices. This
        introduces noise and spuriously increases the apparent similarity to the
        GOE, and so a sensible minimum should be set. Likwise, for large s,
        there may be fluctuations due to noise / unusually large spacings,
        further reducing the utility of any similarity index.

        The default values of 0.5 and 2.5 for the spacings minimum and maximum,
        respectively, were chosen to be conservative: even for small N (~100),
        KDE(s) for 0.5 < s < 2.5 should generally not be deviating wildly from
        where it "should" be, regardless of whether the matrix is sampled from
        the GOE or Poisson / GDE. As N increases, the bounds can be increased
        in both directions.
        """

        def compare(
            expected: ndarray, curve: ndarray, name: str, metric: Metric
        ) -> np.float64:
            comp = Compare(
                curves=[curve], labels=[name], base_curve=expected, base_label="exp"
            )
            res = None
            if metric == "mad":
                res = comp.mean_abs_difference()
            elif metric == "msqd":
                res = comp.mean_sq_difference()
            elif metric == "corr":
                res = comp.correlate()
            else:
                raise ValueError(
                    "Invalid metric. Must be one of ['mad', 'msqd', 'corr']."
                )
            return np.float64(res["exp"][name])

        df = pd.DataFrame(index=metrics, columns=observables)
        if "nnsd" in observables:
            nnsd = self.__get_kde_values(
                spacings_range=spacings, kde_gridsize=kde_gridsize
            )
            nnsd_exp = ensemble.nnsd(spacings_range=spacings, n_points=kde_gridsize)
            for metric in metrics:
                df["nnsd"][metric] = compare(nnsd_exp, nnsd, "nnsd", metric)

        if "nnnsd" in observables:
            nnnsd = self.__get_kde_values(
                spacings_range=spacings, nnnsd=True, kde_gridsize=kde_gridsize
            )
            nnnsd_exp = ensemble.nnnsd(spacings_range=spacings, n_points=kde_gridsize)
            for metric in metrics:
                df["nnnsd"][metric] = compare(nnnsd_exp, nnnsd, "nnnsd", metric)

        if "rigidity" in observables:
            min_L, max_L, n_L = L_rigidity.min(), L_rigidity.max(), len(L_rigidity)
            rigidity = self.spectral_rigidity(
                min_L=min_L, max_L=max_L, L_grid_size=n_L, show_progress=show_progress
            )["delta"]
            rigidity_exp = ensemble.spectral_rigidity(
                min_L=min_L, max_L=max_L, L_grid_size=n_L
            )
            for metric in metrics:
                df["rigidity"][metric] = compare(
                    rigidity_exp, rigidity, "rigidity", metric
                )

        if "levelvar" in observables:
            min_L, max_L, n_L = L_levelvar.min(), L_levelvar.max(), len(L_levelvar)
            levelvar = self.level_variance(L=L_rigidity, show_progress=show_progress)[
                "sigma"
            ]
            levelvar_exp = ensemble.level_variance(L=L_rigidity)
            for metric in metrics:
                df["levelvar"][metric] = compare(
                    levelvar_exp, levelvar, "levelvar", metric
                )
        return df

    def plot_fit(
        self,
        title: str = "Unfolding Fit",
        mode: PlotMode = "block",
        outfile: Path = None,
    ) -> PlotResult:
        return _unfolded_fit(
            self.original_eigs, self.vals, title=title, mode=mode, outfile=outfile
        )

    def plot_nnsd(
        self,
        bins: int = 50,
        kde: bool = True,
        trim: float = 0.0,
        trim_kde: bool = False,
        title: str = "Unfolded Spacing Distribution",
        mode: PlotMode = "block",
        outfile: Path = None,
        ensembles: List[str] = ["poisson", "goe", "gue", "gse"],
    ) -> PlotResult:
        """Plots a histogram of the Nearest-Neighbors Spacing Distribution

        Parameters
        ----------
        unfolded: ndarray
            the unfolded eigenvalues
        bins: int
            the number of (equal-sized) bins to display and use for the histogram
        kde: boolean
            If False (default), do not display a kernel density estimate. If true, use
            [statsmodels.nonparametric.kde.KDEUnivariate](https://www.statsmodels.org/stable/generated/statsmodels.nonparametric.kde.KDEUnivariate.html#statsmodels.nonparametric.kde.KDEUnivariate)
            with arguments {kernel="gau", bw="scott", cut=0} to compute and display the kde
        trim: float
            If True, only use spacings <= `trim` for computing the KDE and plotting.
            Useful for when large spacings distort the histogram.
        trim_kde: bool
            If True, fit the KDE using only spacings <= `trim`. Otherwise, fit the
            KDE using all available spacings.
        title: string
            The plot title string
        mode: "block" | "noblock" | "save" | "return"
            If "block", call plot.plot() and display plot in a blocking fashion.
            If "noblock", attempt to generate plot in nonblocking fashion.
            If "save", save plot to pathlib Path specified in `outfile` argument
            If "return", return (fig, axes), the matplotlib figure and axes object for modification.
        outfile: Path
            If mode="save", save generated plot to Path specified in `outfile` argument.
            Intermediate directories will be created if needed.
        ensembles: ["poisson", "goe", "gue", "gse"]
            Which ensembles to display the expected spectral rigidity curves for comparison against.

        Returns
        -------
        (fig, axes): (Figure, Axes)
            The handles to the matplotlib objects, only if `mode` is "return".
        """
        return _plot_spacings(
            unfolded=self.vals,
            bins=bins,
            kde=kde,
            trim=trim,
            trim_kde=trim_kde,
            title=title,
            mode=mode,
            outfile=outfile,
            ensembles=ensembles,
        )

    def plot_next_nnsd(
        self,
        bins: int = 50,
        kde: bool = True,
        trim: float = 0.0,
        trim_kde: bool = False,
        title: str = "next Nearest-Neigbors Spacing Distribution",
        mode: PlotMode = "block",
        outfile: Path = None,
        ensembles: List[str] = ["goe", "poisson"],
    ) -> PlotResult:
        """Plots a histogram of the next Nearest-Neighbors Spacing Distribution

        Parameters
        ----------
        unfolded: ndarray
            the unfolded eigenvalues
        bins: int
            the number of (equal-sized) bins to display and use for the histogram
        kde: boolean
            If False (default), do not display a kernel density estimate. If true, use
            [statsmodels.nonparametric.kde.KDEUnivariate](https://www.statsmodels.org/stable/generated/statsmodels.nonparametric.kde.KDEUnivariate.html#statsmodels.nonparametric.kde.KDEUnivariate)
            with arguments {kernel="gau", bw="scott", cut=0} to compute and display the kde
        trim: float
            If True, only use spacings <= trim for computing the KDE and plotting.
            Useful for when large spacings distort the histogram.
        trim_kde: bool
            If True, fit the KDE using only spacings <= `trim`. Otherwise, fit the
            KDE using all available spacings.
        title: string
            The plot title string
        mode: "block" | "noblock" | "save" | "return"
            If "block", call plot.plot() and display plot in a blocking fashion.
            If "noblock", attempt to generate plot in nonblocking fashion.
            If "save", save plot to pathlib Path specified in `outfile` argument
            If "return", return (fig, axes), the matplotlib figure and axes object for modification.
        outfile: Path
            If mode="save", save generated plot to Path specified in `outfile` argument.
            Intermediate directories will be created if needed.
        ensembles: List["poisson", "goe"]
            Which ensembles to display the expected next-NNSD curves for.

        Returns
        -------
        (fig, axes): (Figure, Axes)
            The handles to the matplotlib objects, only if `mode` is "return".
        """
        return _next_spacings(
            unfolded=self.vals,
            bins=bins,
            kde=kde,
            trim=trim,
            trim_kde=trim_kde,
            title=title,
            mode=mode,
            outfile=outfile,
            ensembles=ensembles,
        )

    def plot_nnnsd(self, *args: Any, **kwargs: Any) -> PlotResult:
        """Alias for Unfolded.plot_next_nnsd(). """
        return self.plot_next_nnsd(*args, **kwargs)

    def plot_spectral_rigidity(
        self,
        min_L: float = 2,
        max_L: float = 50,
        L_grid_size: int = None,
        c_iters: int = 50000,
        integration: Literal["simps", "trapz"] = "simps",
        title: str = "Spectral Rigidity",
        mode: PlotMode = "block",
        outfile: Path = None,
        ensembles: List[str] = ["poisson", "goe", "gue", "gse"],
        show_progress: bool = True,
    ) -> Tuple[ndarray, ndarray, Optional[PlotResult]]:
        """Compute and plot the spectral rigidity.

        Parameters
        ----------
        min_L: int
            The lowest possible L value for which to compute the spectral
            rigidity. Default 2.
        max_L: int = 20
            The largest possible L value for which to compute the spectral
            rigidity.
        L_grid_size: int
            The number of values of L to generate betwen min_L and max_L. Default
            2 * (max_L - min_L).
        c_iters: int = 50
            How many times the location of the center, c, of the interval
            [c - L/2, c + L/2] should be chosen uniformly at random for
            each L in order to compute the estimate of the spectral
            rigidity. Not a particularly significant effect on performance.
        title: string
            The plot title string
        mode: "block" (default) | "noblock" | "save" | "return"
            If "block", call plot.plot() and display plot in a blocking fashion.
            If "noblock", attempt to generate plot in nonblocking fashion.
            If "save", save plot to pathlib Path specified in `outfile` argument
            If "return", return (fig, axes), the matplotlib figure and axes object for modification.
        outfile: Path
            If mode="save", save generated plot to Path specified in `outfile` argument.
            Intermediate directories will be created if needed.
        ensembles: ["poisson", "goe", "gue", "gse"]
            Which ensembles to display the expected spectral rigidity curves for comparison against.
        show_progress: bool
            Whether or not to display computation progress in stdout.


        Returns
        -------
        L : ndarray
            The L values generated based on the values of L_grid_size,
            min_L, and max_L.
        delta3 : ndarray
            The computed spectral rigidity values for each of L.
        figure, axes: Optional[PlotResult]
            If mode is "return", the matplotlib figure and axes object for modification.
            Otherwise, None.

        References
        ----------
        .. [1] Mehta, M. L. (2004). Random matrices (Vol. 142). Elsevier.
        """
        unfolded = self.values
        L, delta = spectral_rigidity(
            unfolded,
            c_iters=c_iters,
            L_grid_size=L_grid_size,
            min_L=min_L,
            max_L=max_L,
            integration=integration,
            show_progress=show_progress,
        )
        plot_result = plot._spectral_rigidity(
            unfolded,
            pd.DataFrame({"L": L, "delta": delta}),
            title,
            mode,
            outfile,
            ensembles,
        )
        return L, delta, plot_result

    def plot_level_variance(
        self,
        L: ndarray = np.arange(0.5, 20, 0.2),
        sigma: ndarray = None,
        tol: float = 0.01,
        max_L_iters: int = 50000,
        min_L_iters: int = 1000,
        title: str = "Level Number Variance",
        mode: PlotMode = "block",
        outfile: Path = None,
        ensembles: List[str] = ["poisson", "goe", "gue", "gse"],
        show_progress: bool = True,
    ) -> Tuple[ndarray, ndarray, Optional[PlotResult]]:
        """Compute and plot the level number variance of the current unfolded
        eigenvalues.

        Parameters
        ----------
        L: ndarray
            The grid of L values for which to compute the level variance.
        sigma: ndarray
            If the number level variance has already been computed, pass it in
            to `sigma` to save the values being re-computed.
        tol: float
            Stop iterating when the last `min_L_iters` computed values of the
            level variance have a range (i.e. max - min) < tol.
        max_L_iters: int
            Stop computing values for the level variance once max_L_iters values
            have been computed for each L value.
        min_L_iters: int
            Minimum number of iterations for each L value.
        title: string
            The plot title string
        mode: "block" (default) | "noblock" | "save" | "return"
            If "block", call plot.plot() and display plot in a blocking fashion.
            If "noblock", attempt to generate plot in nonblocking fashion.
            If "save", save plot to pathlib Path specified in `outfile` argument
            If "return", return (fig, axes), the matplotlib figure and axes object for modification.
        outfile: Path
            If mode="save", save generated plot to Path specified in `outfile` argument.
            Intermediate directories will be created if needed.
        ensembles: ["poisson", "goe", "gue", "gse"]
            Which ensembles to display the expected spectral rigidity curves for comparison against.
        show_progress: bool
            Show a pretty progress bar while computing.

        Returns
        -------
        L_vals: ndarray
            The L values generated based on the values of L_grid_size,
            min_L, and max_L.
        sigma_squared: ndarray
            The computed level number variance values for each L.
        figure, axes: Optional[PlotResult]
            If mode is "return", the matplotlib figure and axes object for modification.
            Otherwise, None.
        """
        unfolded = self.values
        if sigma is not None:
            sigma = make_1d_array(sigma)
            plot_result = plot._level_number_variance(
                unfolded=unfolded,
                data=pd.DataFrame({"L": L, "sigma": sigma}),
                title=title,
                mode=mode,
                outfile=outfile,
                ensembles=ensembles,
            )
            return L, sigma, plot_result

        L_vals, sigma = level_number_variance_stable(
            unfolded=unfolded,
            L=L,
            tol=tol,
            max_L_iters=max_L_iters,
            min_L_iters=min_L_iters,
            show_progress=show_progress,
        )
        plot_result = plot._level_number_variance(
            unfolded=unfolded,
            data=pd.DataFrame({"L": L, "sigma": sigma}),
            title=title,
            mode=mode,
            outfile=outfile,
            ensembles=ensembles,
        )
        return L_vals, sigma, plot_result

    def __get_kde_values(
        self,
        spacings_range: Tuple[float, float],
        nnnsd: bool = False,
        kde_gridsize: int = 1000,
    ) -> np.array:
        """Fit / derive the KDE using the entire set of unfolded values, but
        evaluating only over the given `spacings_range`. """
        spacings = np.sort(self.vals[2:] - self.vals[:-2]) if nnnsd else self.spacings
        kde = KDE(spacings)
        kde.fit(kernel="gau", bw="scott", cut=0, fft=False, gridsize=10000)
        s = np.linspace(spacings_range[0], spacings_range[1], kde_gridsize)
        # evaluated = np.empty_like(s)
        # for i, _ in enumerate(evaluated):
        #     evaluated[i] = kde.evaluate(s[i])
        evaluated = kde.evaluate(s)
        return evaluated
