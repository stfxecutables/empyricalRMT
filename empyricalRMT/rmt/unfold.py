import numpy as np
import pandas as pd

from numpy import ndarray
from pandas import DataFrame
from pathlib import Path
from typing import Any, List, Optional, Tuple
from typing_extensions import Literal

import empyricalRMT.rmt.plot as plot

from empyricalRMT.rmt._eigvals import EigVals
from empyricalRMT.rmt.observables.levelvariance import level_number_variance
from empyricalRMT.rmt.observables.rigidity import spectral_rigidity
from empyricalRMT.rmt.plot import _next_spacings, PlotMode, PlotResult


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
        min_L: float = 2,
        max_L: float = 50,
        c_iters: int = 50000,
        L_grid_size: int = None,
        show_progress: bool = False,
    ) -> DataFrame:
        """Compute the level number variance of the current unfolded eigenvalues.

        Parameters
        ----------
        min_L: int
            The lowest possible L value for which to compute the spectral
            rigidity.
        max_L: int
            The largest possible L value for which to compute the spectral
            rigidity.
        c_iters: int
            How many times the location of the center, c, of the interval
            [c - L/2, c + L/2] should be chosen uniformly at random for
            each L in order to compute the estimate of the number level
            variance.
        L_grid_size: int
            The number of values of L to generate betwen min_L and max_L.
        show_progress: bool
            Whether or not to display computation progress in stdout.

        Returns
        -------
        df: DataFrame
            A dataframe with columns "L", the L values generated based on the
            input arguments, and "sigma", the computed level variances for each
            value of L.
        """
        unfolded = self.values
        L, sigma = level_number_variance(
            unfolded=unfolded,
            c_iters=c_iters,
            L_grid_size=L_grid_size,
            min_L=min_L,
            max_L=max_L,
            show_progress=show_progress,
        )
        return DataFrame({"L": L, "sigma": sigma})

    def plot_nnsd(self, *args: Any, **kwargs: Any) -> PlotResult:
        return self.plot_spacings(*args, **kwargs)

    def plot_next_nnsd(
        self,
        bins: int = 50,
        kde: bool = True,
        title: str = "next Nearest-Neigbors Spacing Distribution",
        mode: PlotMode = "block",
        outfile: Path = None,
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

        Returns
        -------
        (fig, axes): (Figure, Axes)
            The handles to the matplotlib objects, only if `mode` is "return".
        """
        return _next_spacings(
            unfolded=self.vals,
            bins=bins,
            kde=kde,
            title=title,
            mode=mode,
            outfile=outfile,
        )

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
        min_L: float = 2,
        max_L: float = 50,
        c_iters: int = 50000,
        L_grid_size: int = None,
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
        min_L: int
            The lowest possible L value for which to compute the spectral
            rigidity.
        max_L: int
            The largest possible L value for which to compute the spectral
            rigidity.
        c_iters: int
            How many times the location of the center, c, of the interval
            [c - L/2, c + L/2] should be chosen uniformly at random for
            each L in order to compute the estimate of the number level
            variance.
        L_grid_size: int
            The number of values of L to generate betwen min_L and max_L.
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
        L: ndarray
            The L values generated based on the values of L_grid_size,
            min_L, and max_L.
        sigma_squared: ndarray
            The computed level number variance values for each L.
        figure, axes: Optional[PlotResult]
            If mode is "return", the matplotlib figure and axes object for modification.
            Otherwise, None.
        """
        unfolded = self.values
        L, sigma = level_number_variance(
            unfolded=unfolded,
            c_iters=c_iters,
            L_grid_size=L_grid_size,
            min_L=min_L,
            max_L=max_L,
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
        return L, sigma, plot_result
