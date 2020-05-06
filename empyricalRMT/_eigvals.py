import numpy as np

from numpy import ndarray
from typing import Sized

from empyricalRMT._validate import make_1d_array
from empyricalRMT.observables.step import _step_function_fast
from empyricalRMT.plot import _spacings as plotSpacings
from empyricalRMT.plot import _raw_eig_dist, _raw_eig_sorted, _step_function, PlotResult


class EigVals:
    """Base class, not to be instantiated. """

    def __init__(self, eigenvalues: Sized):
        self.__construct_vals: ndarray = make_1d_array(eigenvalues)
        self._steps = None
        self._vals = np.sort(eigenvalues)  # to be overridden in actual classes

    @property
    def original_values(self) -> ndarray:
        return self.__construct_vals

    @property
    def original_eigs(self) -> ndarray:
        return self.__construct_vals

    @property
    def original_eigenvalues(self) -> ndarray:
        return self.__construct_vals

    # NOTE: This *must* be overridden
    @property
    def values(self) -> ndarray:
        raise NotImplementedError(".values() should be implemented in derived classes.")

    # NOTE: This *must* be overridden
    @property
    def vals(self) -> ndarray:
        raise NotImplementedError(".vals() should be implemented in derived classes.")

    @property
    def steps(self) -> ndarray:
        if self._steps is None:
            self._steps = _step_function_fast(self._vals, self._vals)
        return self._steps

    @property
    def spacings(self) -> ndarray:
        return np.diff(np.sort(self.vals))

    def step_function(self, x: ndarray) -> ndarray:
        return _step_function_fast(eigs=self.vals, x=x)

    def plot_sorted(self, *args, **kwargs) -> PlotResult:  # type: ignore
        return _raw_eig_sorted(eigs=self.values, *args, **kwargs)  # type: ignore

    def plot_distribution(self, *args, **kwargs) -> PlotResult:  # type: ignore
        return _raw_eig_dist(eigs=self.values, *args, **kwargs)  # type: ignore

    def plot_steps(self, *args, **kwargs) -> PlotResult:  # type: ignore
        return _step_function(eigs=self.values, *args, **kwargs)  # type: ignore

    def plot_spacings(self, *args, **kwargs) -> PlotResult:  # type: ignore
        return plotSpacings(unfolded=self.values, *args, **kwargs)  # type: ignore
