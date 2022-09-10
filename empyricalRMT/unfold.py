from pathlib import Path
from typing import Any, List, Mapping, Optional, Tuple, Type, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from pandas import DataFrame
from statsmodels.nonparametric.kde import KDEUnivariate as KDE
from typing_extensions import Literal

import empyricalRMT.plot as plot
from empyricalRMT._constants import RIGIDITY_GRID
from empyricalRMT._eigvals import EigVals
from empyricalRMT._types import Smoother, fArr
from empyricalRMT._validate import make_1d_array
from empyricalRMT.brody import brody_fit_evaluate
from empyricalRMT.compare import Compare, Metric
from empyricalRMT.ensemble import Ensemble
from empyricalRMT.observables.levelvariance import level_number_variance
from empyricalRMT.observables.rigidity import spectral_rigidity as _spectral_rigidity
from empyricalRMT.plot import PlotMode, PlotResult, _brody_fit, _next_spacings, _observables
from empyricalRMT.plot import _spacings as _plot_spacings
from empyricalRMT.plot import _unfolded_fit

Observables = Literal["nnsd", "nnnsd", "rigidity", "levelvar"]


class Unfolded(EigVals):
    def __init__(
        self,
        originals: fArr,
        unfolded: fArr,
        smoother: Optional[Smoother] = None,
    ):
        super().__init__(originals)
        self._vals: fArr = np.array(unfolded)
        self._smoother = smoother

    @property
    def values(self) -> fArr:
        return self._vals

    @property
    def vals(self) -> fArr:
        return self._vals

    def evaluate_smoother(self, x: fArr) -> fArr:
        if self._smoother is None:
            raise NotImplementedError(
                "Your unfolded eigenvalues were probably constructed by auto-trimming."
                "Acquiring the smoother from this method is currently unsupported."
            )
        return self._smoother(x)

    def spectral_rigidity(
        self,
        L: fArr = np.arange(2, 50, 1, dtype=np.float64),
        max_iters: int = 0,
        gridsize: int = RIGIDITY_GRID,
        tol: float = 0.01,
        integration: Literal["simps", "trapz"] = "simps",
        show_progress: bool = True,
    ) -> DataFrame:
        """Compute and the spectral rigidity.

        Parameters
        ----------
        L: ndarray
            The values for which to compute the spectral rigidity.

        c_iters: int = 50
            How many times the location of the center, c, of the interval
            [c - L/2, c + L/2] should be chosen uniformly at random for
            each L in order to compute the estimate of the spectral
            rigidity. Not a particularly significant effect on performance.

        integration: "simps" | "trapz"
            Numerical integration method. "trapz" might be faster in some cases,
            at the cost of significant accuracy.

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
        L_vals, delta, converged, iters = _spectral_rigidity(
            unfolded=unfolded,
            L=L,
            gridsize=gridsize,
            max_iters=max_iters,
            tol=tol,
            integration=integration,
            show_progress=show_progress,
        )
        return pd.DataFrame(
            {
                "L": L_vals,
                "delta": delta,
                "converged": converged,
                "iters": iters,
            }
        )

    def level_variance(
        self,
        L: ndarray = np.arange(0.5, 20, 0.2),
        tol: float = 0.01,
        max_L_iters: int = int(1e6),
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
        L, sigma, converged, iters = level_number_variance(
            unfolded=unfolded,
            L=L,
            tol=tol,
            max_iters=max_L_iters,
            show_progress=show_progress,
        )
        return DataFrame({"L": L, "sigma": sigma, "converged": converged, "iters": iters})

    def fit_brody(self, method: str = "spacing") -> DataFrame:
        """Get an estimate for the beta parameter of the Brody distribution fit of the spacings.

        Parameters
        ----------
        method: "spacing" | "mse"
            Method for estimating parameter of the Brody distribution. If "spacing", use
            [maximum spacing estimation](https://en.wikipedia.org/wiki/Maximum_spacing_estimation).
            If "mle", use maximum likelihood. The default is "spacing",
            as this may be preferable for the J-shape of the Brody distribution.

        Returns
        -------
        info: dict
            A dict containing some useful fit information:
                "beta": float
                    The estimate for beta.
                "spacings": ndarray
                    The sorted spacings.
                "ecdf": ndarray
                    The empirical cumulative distribution function of the spacings.
                "brody_cdf": ndarray
                    The cumulative distribution function of the Brody distribution
                    fit to the spacings
                "mad": float
                    The mean absolute deviation between the empirical and Brody cdf
                "msqd": float
                    The mean squared deviation between the empirical and Brody cdf
        """
        s = self.spacings
        s = s[s > 0]
        return brody_fit_evaluate(s, method)

    def plot_brody_fit(
        self,
        method: str = "spacing",
        title: str = "Brody distribution fit",
        mode: PlotMode = PlotMode.Return,
        outfile: Optional[Path] = None,
        save_dpi: Optional[int] = None,
        ensembles: List[str] = ["goe", "poisson"],
        bins: int = 50,
        kde: bool = True,
        trim: float = 5.0,
        trim_kde: bool = False,
        kde_bw: Union[float, str] = "scott",
    ) -> PlotResult:
        unfolded = self.vals
        return _brody_fit(
            unfolded=unfolded,
            method=method,
            title=title,
            mode=mode,
            outfile=outfile,
            save_dpi=save_dpi,
            ensembles=ensembles,
            bins=bins,
            kde=kde,
            trim=trim,
            trim_kde=trim_kde,
            kde_bw=kde_bw,
        )

    def ensemble_compare(
        self,
        ensemble: Type[Ensemble],
        observables: List[Observables] = ["nnsd", "nnnsd", "rigidity", "levelvar"],
        metrics: List[Metric] = ["msqd"],
        spacings: Tuple[float, float] = (0.5, 2.5),
        kde_gridsize: int = 5000,
        L_rigidity: ndarray = np.arange(2, 50, 0.5),
        L_levelvar: ndarray = np.arange(0.2, 20, 0.2),
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

        def compare(expected: ndarray, curve: ndarray, name: str, metric: Metric) -> np.float64:
            comp = Compare(curves=[curve], labels=[name], base_curve=expected, base_label="exp")
            res = None
            if metric == "mad":
                res = comp.mean_abs_difference()
            elif metric == "msqd":
                res = comp.mean_sq_difference()
            elif metric == "corr":
                res = comp.correlate()
            else:
                raise ValueError("Invalid metric. Must be one of ['mad', 'msqd', 'corr'].")
            return np.float64(res["exp"][name])

        df = pd.DataFrame(index=metrics, columns=observables)
        if "nnsd" in observables:
            nnsd = self.__get_kde_values(spacings_range=spacings, kde_gridsize=kde_gridsize)
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
            rigidity = self.spectral_rigidity(L=L_rigidity, show_progress=show_progress)[
                "delta"
            ].to_numpy()
            rigidity_exp = ensemble.spectral_rigidity(L=L_rigidity)
            for metric in metrics:
                df["rigidity"][metric] = compare(rigidity_exp, rigidity, "rigidity", metric)

        if "levelvar" in observables:
            levelvar = self.level_variance(L=L_levelvar, show_progress=show_progress)[
                "sigma"
            ].to_numpy()
            levelvar_exp = ensemble.level_variance(L=L_levelvar)
            for metric in metrics:
                df["levelvar"][metric] = compare(levelvar_exp, levelvar, "levelvar", metric)
        return df

    def plot_fit(
        self,
        title: str = "Unfolding Fit",
        mode: PlotMode = PlotMode.Return,
        outfile: Optional[Path] = None,
        fig: Optional[Figure] = None,
        axes: Optional[Axes] = None,
    ) -> PlotResult:
        return _unfolded_fit(
            self.original_eigs,
            self.vals,
            title=title,
            mode=mode,
            outfile=outfile,
            fig=fig,
            axes=axes,
        )

    def plot_nnsd(
        self,
        bins: int = 50,
        kde: bool = True,
        trim: float = 10.0,
        trim_kde: bool = False,
        kde_bw: Union[float, str] = "scott",
        brody: bool = False,
        brody_fit: str = "spacing",
        title: str = "Unfolded Spacing Distribution",
        mode: PlotMode = PlotMode.Return,
        outfile: Optional[Path] = None,
        ensembles: List[str] = ["poisson", "goe", "gue", "gse"],
        fig: Optional[Figure] = None,
        axes: Optional[Axes] = None,
    ) -> PlotResult:
        """Plots a histogram of the Nearest-Neighbors Spacing Distribution

        Parameters
        ----------
        unfolded: ndarray
            the unfolded eigenvalues

        bins: int
            the number of (equal-sized) bins to display and use for the
            histogram

        kde: boolean
            If False (default), do not display a kernel density estimate. If true, use
            [statsmodels.nonparametric.kde.KDEUnivariate](https://www.statsmodels.org/stable/generated/statsmodels.nonparametric.kde.KDEUnivariate.html#statsmodels.nonparametric.kde.KDEUnivariate)
            with arguments {kernel="gau", bw="scott", cut=0} to compute and
            display the kde

        trim: float
            If True, only use spacings <= `trim` for computing the KDE and plotting.
            Useful for when large spacings distort the histogram.

        trim_kde: bool
            If True, fit the KDE using only spacings <= `trim`. Otherwise, fit the
            KDE using all available spacings.

        kde_bw: float
            The bandwidth to use for kernel density estimation.

        brody: bool
            If True, compute the best-fitting Brody distribution via MLE, and plot
            that distribution.

        brody_fit: "spacing" | "mle"
            Method for parametric distribution fitting of the Brody distribution to
            the data. If "spacing", use
            [maximum spacing estimation](https://en.wikipedia.org/wiki/Maximum_spacing_estimation).
            If "mle", use the maximum likelihood method. The default is "spacing",
            as this may be preferable for the J-shape of the Brody distribution.

        title: string
            The plot title string

        mode: "block" | "noblock" | "save" | "return"
            If "block", call plot.plot() and display plot in a blocking fashion.
            If "noblock", attempt to generate plot in nonblocking fashion.
            If "save", save plot to pathlib Path specified in `outfile` argument
            If "return", return (fig, axes), the matplotlib figure and axes
            object for modification.

        outfile: Path
            If mode="save", save generated plot to Path specified in `outfile` argument.
            Intermediate directories will be created if needed.

        ensembles: ["poisson", "goe", "gue", "gse"]
            Which ensembles to display the expected spectral rigidity curves for comparison against.

        fig: Figure
            If provided with `axes`, configure plotting with the provided `fig`
            object instead of creating a new figure. Useful for creating subplots.

        axes: Axes
            If provided with `fig`, plot to the provided `axes` object. Useful for
            creating subplots.


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
            kde_bw=kde_bw,
            brody=brody,
            brody_fit=brody_fit,
            title=title,
            mode=mode,
            outfile=outfile,
            ensembles=ensembles,
            fig=fig,
            axes=axes,
        )

    def plot_next_nnsd(
        self,
        bins: int = 50,
        kde: bool = True,
        trim: float = 10.0,
        trim_kde: bool = False,
        brody: bool = False,
        brody_fit: str = "spacing",
        title: str = "next Nearest-Neigbors Spacing Distribution",
        mode: PlotMode = PlotMode.Return,
        outfile: Optional[Path] = None,
        ensembles: List[str] = ["goe", "poisson"],
        fig: Optional[Figure] = None,
        axes: Optional[Axes] = None,
    ) -> PlotResult:
        """Plots a histogram of the next Nearest-Neighbors Spacing Distribution

        Parameters
        ----------
        unfolded: ndarray
            the unfolded eigenvalues

        bins: int
            the number of (equal-sized) bins to display and use for the
            histogram

        kde: boolean
            If False (default), do not display a kernel density estimate. If true, use
            [statsmodels.nonparametric.kde.KDEUnivariate](https://www.statsmodels.org/stable/generated/statsmodels.nonparametric.kde.KDEUnivariate.html#statsmodels.nonparametric.kde.KDEUnivariate)
            with arguments {kernel="gau", bw="scott", cut=0} to compute and
            display the kde

        trim: float
            If True, only use spacings <= trim for computing the KDE and plotting.
            Useful for when large spacings distort the histogram.

        trim_kde: bool
            If True, fit the KDE using only spacings <= `trim`. Otherwise, fit the
            KDE using all available spacings.

        brody: bool
            If True, compute the best-fitting Brody distribution via MLE, and plot
            that distribution.

        brody_fit: "spacing" | "mle"
            Method for parametric distribution fitting of the Brody distribution to
            the data. If "spacing", use
            [maximum spacing estimation](https://en.wikipedia.org/wiki/Maximum_spacing_estimation).
            If "mle", use the maximum likelihood method. The default is "spacing",
            as this may be preferable for the J-shape of the Brody distribution.

        title: string
            The plot title string

        mode: "block" | "noblock" | "save" | "return"
            If "block", call plot.plot() and display plot in a blocking fashion.
            If "noblock", attempt to generate plot in nonblocking fashion.
            If "save", save plot to pathlib Path specified in `outfile` argument
            If "return", return (fig, axes), the matplotlib figure and axes
            object for modification.

        outfile: Path
            If mode="save", save generated plot to Path specified in `outfile` argument.
            Intermediate directories will be created if needed.

        ensembles: List["poisson", "goe"]
            Which ensembles to display the expected next-NNSD curves for.

        fig: Figure
            If provided with `axes`, configure plotting with the provided `fig`
            object instead of creating a new figure. Useful for creating subplots.

        axes: Axes
            If provided with `fig`, plot to the provided `axes` object. Useful for
            creating subplots.


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
            brody=brody,
            brody_fit=brody_fit,
            title=title,
            mode=mode,
            outfile=outfile,
            ensembles=ensembles,
            fig=fig,
            axes=axes,
        )

    def plot_nnnsd(self, *args: Any, **kwargs: Any) -> PlotResult:
        """Alias for Unfolded.plot_next_nnsd()."""
        return self.plot_next_nnsd(*args, **kwargs)

    def plot_spectral_rigidity(
        self,
        data: Optional[DataFrame] = None,
        L: ndarray = np.arange(2, 50, 0.5),
        max_iters: int = int(1e6),
        gridsize: int = 1000,
        tol: float = 0.01,
        integration: Literal["simps", "trapz"] = "simps",
        title: str = "Spectral Rigidity",
        mode: PlotMode = PlotMode.Return,
        outfile: Optional[Path] = None,
        ensembles: List[str] = ["poisson", "goe"],
        show_iters: bool = False,
        show_progress: bool = True,
        fig: Optional[Figure] = None,
        axes: Optional[Axes] = None,
        **kwargs: Mapping,
    ) -> Tuple[ndarray, ndarray, Optional[PlotResult]]:
        """Compute and plot the spectral rigidity.

        Parameters
        ----------
        L: ndarray
            The values for which to compute the spectral rigidity.

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
            If "return", return (fig, axes), the matplotlib figure and axes
            object for modification.

        outfile: Path
            If mode="save", save generated plot to Path specified in `outfile` argument.
            Intermediate directories will be created if needed.

        ensembles: ["poisson", "goe", "gue", "gse"]
            Which ensembles to display the expected spectral rigidity curves for
            comparison against.

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
        if data is None:
            L_vals, delta, converged, iters = _spectral_rigidity(
                unfolded,
                L=L,
                gridsize=gridsize,
                max_iters=max_iters,
                tol=tol,
                integration=integration,
                show_progress=show_progress,
            )
            data = pd.DataFrame(
                {
                    "L": L_vals,
                    "delta": delta,
                    "converged": converged,
                    "iters": iters,
                }
            )
        plot_result = plot._spectral_rigidity(
            unfolded=unfolded,
            data=data,
            title=title,
            mode=mode,
            outfile=outfile,
            ensembles=ensembles,
            show_iters=show_iters,
            fig=fig,
            axes=axes,
            **kwargs,
        )
        return L, data["delta"], plot_result  # type: ignore

    def plot_level_variance(
        self,
        L: fArr = np.arange(0.5, 20, 0.2),
        sigma: Optional[fArr] = None,
        tol: float = 0.01,
        max_L_iters: int = 0,
        title: str = "Level Number Variance",
        mode: PlotMode = PlotMode.Return,
        outfile: Optional[Path] = None,
        ensembles: List[str] = ["goe"],
        show_iters: bool = False,
        show_progress: bool = True,
        fig: Optional[Figure] = None,
        axes: Optional[Axes] = None,
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
            If "return", return (fig, axes), the matplotlib figure and axes
            object for modification.

        outfile: Path
            If mode="save", save generated plot to Path specified in `outfile` argument.
            Intermediate directories will be created if needed.

        ensembles: ["poisson", "goe", "gue", "gse"]
            Which ensembles to display the expected spectral rigidity curves for
            comparison against.

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
            converged = np.ones_like(sigma, dtype=bool)
            plot_result = plot._level_number_variance(
                unfolded=unfolded,
                data=pd.DataFrame({"L": L, "sigma": sigma, "converged": converged}),
                title=title,
                mode=mode,
                outfile=outfile,
                ensembles=ensembles,
                show_iters=show_iters,
                fig=fig,
                axes=axes,
            )
            return L, sigma, plot_result

        L_vals, sigma, converged, iters = level_number_variance(
            unfolded=unfolded,
            L=L,
            tol=tol,
            max_iters=max_L_iters,
            show_progress=show_progress,
        )
        plot_result = plot._level_number_variance(
            unfolded=unfolded,
            data=pd.DataFrame(
                {
                    "L": L,
                    "sigma": sigma,
                    "converged": converged,
                    "iters": iters,
                }
            ),
            title=title,
            mode=mode,
            outfile=outfile,
            ensembles=ensembles,
            show_iters=show_iters,
            fig=fig,
            axes=axes,
        )
        return L_vals, sigma, plot_result

    def plot_observables(
        self,
        rigidity_L: ndarray = np.arange(2, 50, 0.5),
        levelvar_L: ndarray = np.arange(0.2, 20, 0.2),
        title: str = "Spectral Observables",
        mode: PlotMode = PlotMode.Return,
        outfile: Optional[Path] = None,
        ensembles: List[str] = ["goe", "poisson"],
        show_progress: bool = True,
        max_iters: int = 0,
        **levelvar_kwargs: Any,
    ) -> PlotResult:
        """Plot some popular spectral observables, as well as a plot of the unfolding
        fit. For public use, use `Unfolded.plot_observables()`.

        rigidity_L: ndarray
            The values of L for which to compute the spectral rigidity.

        levelvar_L: ndarray
            The values of L for which to compute the level variance.

        title: string
            The plot title string

        mode: "block" (default) | "noblock" | "save" | "return"
            If "block", call plot.plot() and display plot in a blocking fashion.
            If "noblock", attempt to generate plot in nonblocking fashion.
            If "save", save plot to pathlib Path specified in `outfile` argument
            If "return", return (fig, axes), the matplotlib figure and axes object
            for modification.

        outfile: Path
            If mode="save", save generated plot to Path specified in `outfile` argument.
            Intermediate directories will be created if needed.

        ensembles: ["poisson", "goe", "gue", "gse"]
            Which ensembles to display the expected number level variance curves
            for comparison against.

        show_progress: bool
            If True, print a progress bar during longer calculations.

        max_iters: int
            The number of iterations, per L, to use when calculating the
            spectral rigidity.

        levelvar_kwargs: dict
            The keyword args to pass on to Unfolded.level_variance().
        """
        rigidity = self.spectral_rigidity(
            L=rigidity_L, max_iters=max_iters, show_progress=show_progress
        )
        levelvar = self.level_variance(L=levelvar_L, show_progress=show_progress, **levelvar_kwargs)
        return _observables(
            eigs=self.original_eigs,
            unfolded=self.vals,
            rigidity_df=rigidity,
            levelvar_df=levelvar,
            ensembles=ensembles,
            suptitle=title,
            mode=mode,
            outfile=outfile,
        )

    def __get_kde_values(
        self,
        spacings_range: Tuple[float, float],
        nnnsd: bool = False,
        kde_gridsize: int = 1000,
    ) -> fArr:
        """Fit / derive the KDE using the entire set of unfolded values, but
        evaluating only over the given `spacings_range`."""
        spacings = np.sort(self.vals[2:] - self.vals[:-2]) if nnnsd else self.spacings
        kde = KDE(spacings)
        kde.fit(kernel="gau", bw="scott", cut=0, fft=False, gridsize=10000)
        s = np.linspace(spacings_range[0], spacings_range[1], kde_gridsize)
        # evaluated = np.empty_like(s)
        # for i, _ in enumerate(evaluated):
        #     evaluated[i] = kde.evaluate(s[i])
        evaluated: fArr = np.array(kde.evaluate(s))
        return evaluated
