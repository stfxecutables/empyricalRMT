import numpy as np
import pandas as pd

from numpy import ndarray
from pathlib import Path
from typing import Any, List
from warnings import warn

import empyricalRMT.rmt.plot as plot

from empyricalRMT.rmt._eigvals import EigVals
from empyricalRMT.rmt.observables.rigidity import spectralRigidity
from empyricalRMT.rmt.plot import PlotMode, PlotResult


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

    def plot_nnsd(self, *args: Any, **kwargs: Any) -> PlotResult:
        return self.plot_spacings(*args, **kwargs)

    def plot_spectral_rigidity(
        self,
        min_L: float = 2,
        max_L: float = 50,
        L_grid_size: int = None,
        c_iters: int = 50000,
        title: str = "Spectral Rigidity",
        mode: PlotMode = "block",
        outfile: Path = None,
        ensembles: List[str] = ["poisson", "goe", "gue", "gse"],
        show_progress: bool = True,
    ) -> PlotResult:
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


        Returns
        -------
        L : ndarray
            The L values generated based on the values of L_grid_size,
            min_L, and max_L.
        delta3 : ndarray
            The computed spectral rigidity values for each of L.

        References
        ----------
        .. [1] Mehta, M. L. (2004). Random matrices (Vol. 142). Elsevier.
        """
        unfolded = self.values
        L, delta = spectralRigidity(
            unfolded,
            c_iters=c_iters,
            L_grid_size=L_grid_size,
            min_L=min_L,
            max_L=max_L,
            show_progress=show_progress,
        )
        return plot.spectralRigidity(
            unfolded,
            pd.DataFrame({"L": L, "delta": delta}),
            title,
            mode,
            outfile,
            ensembles,
        )

    def plot_number_level_variance(self) -> PlotResult:
        raise NotImplementedError()
