from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

from empyricalRMT._types import fArr, iArr
from empyricalRMT._validate import make_1d_array
from empyricalRMT.observables.step import _step_function_fast
from empyricalRMT.plot import PlotResult, _raw_eig_dist, _raw_eig_sorted
from empyricalRMT.plot import _spacings as plotSpacings
from empyricalRMT.plot import _step_function


class EigVals:
    """Base class, not to be instantiated."""

    def __init__(self, values: ArrayLike):
        if values is None:
            raise ValueError("`eigenvalues` must be ArrayLike.")
        vals = np.array(values, dtype=np.float64)
        if vals.ndim > 2:
            raise ValueError(f"Cannot interpret input {vals} as 1D or 2D.")
        if vals.ndim == 2:
            vals = np.asmatrix(vals)
            if np.array_equal(vals.H, vals):
                eigs = np.linalg.eigvalsh(vals)
            else:
                eigs = np.linalg.eigvals(vals)
        else:
            eigs = make_1d_array(vals)
        eigs = np.sort(np.array(eigs, dtype=np.float64))

        self.__construct_vals: fArr = eigs
        self._steps: Optional[iArr] = None
        self._vals: fArr = np.sort(self.__construct_vals)  # to be overridden in actual classes

    @property
    def original_values(self) -> fArr:
        return self.__construct_vals

    @property
    def original_eigs(self) -> fArr:
        return self.__construct_vals

    @property
    def original_eigenvalues(self) -> fArr:
        return self.__construct_vals

    @property
    def values(self) -> fArr:
        raise NotImplementedError(".values() should be implemented in derived classes.")

    @property
    def vals(self) -> fArr:
        raise NotImplementedError(".vals() should be implemented in derived classes.")

    @property
    def steps(self) -> iArr:
        if self._steps is None:
            self._steps = _step_function_fast(self._vals, self._vals)
        return self._steps

    @property
    def spacings(self) -> fArr:
        return np.diff(np.sort(self.vals))  # type: ignore

    def step_function(self, x: fArr) -> fArr:
        return _step_function_fast(eigs=self.vals, x=x)  # type: ignore

    def plot_sorted(self, *args, **kwargs) -> PlotResult:  # type: ignore
        return _raw_eig_sorted(eigs=self.values, *args, **kwargs)  # type: ignore

    def plot_distribution(self, *args, **kwargs) -> PlotResult:  # type: ignore
        return _raw_eig_dist(eigs=self.values, *args, **kwargs)  # type: ignore

    def plot_steps(self, *args, **kwargs) -> PlotResult:  # type: ignore
        return _step_function(eigs=self.values, *args, **kwargs)  # type: ignore

    def plot_spacings(self, *args, **kwargs) -> PlotResult:  # type: ignore
        return plotSpacings(unfolded=self.values, *args, **kwargs)  # type: ignore
