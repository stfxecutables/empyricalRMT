import numpy as np
import pandas as pd

from numpy import ndarray
from pathlib import Path
from typing import Any

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
        title: str = "Spectral Rigidity",
        mode: PlotMode = "block",
        outfile: Path = None,
    ) -> PlotResult:
        eigs = self.original_eigenvalues
        unfolded = self.values
        L, delta = spectralRigidity(
            eigs, unfolded, c_iters=10000, L_grid_size=50, min_L=2.0, max_L=25.0
        )
        return plot.spectralRigidity(
            unfolded, pd.DataFrame({"L": L, "delta": delta}), title, mode, outfile
        )

    def plot_number_level_variance(self) -> PlotResult:
        raise NotImplementedError()
