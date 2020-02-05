import numpy as np

from numpy import ndarray
from typing import Iterable, Sized

from empyricalRMT.rmt.observables.step import stepFunctionVectorized
from empyricalRMT.rmt.plot import spacings as plotSpacings
from empyricalRMT.rmt.plot import rawEigDist, rawEigSorted, stepFunction, PlotResult


class EigVals:
    def __init__(self, eigenvalues: Sized):
        try:
            self.__construct_vals: ndarray = np.array(eigenvalues, dtype=np.float64)
        except ValueError as e:
            raise ValueError(
                "Must pass in eigenvalues that can be coerced to numpy.float type"
            ) from e
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
            self._steps = stepFunctionVectorized(self._vals, self._vals)
        return self._steps

    @property
    def spacings(self) -> ndarray:
        return self.vals[1:] - self.vals[:-1]

    def step_function(self, x: ndarray) -> ndarray:
        return stepFunctionVectorized(eigs=self.vals, x=x)

    def plot_sorted(self, *args, **kwargs) -> PlotResult:  # type: ignore
        return rawEigSorted(eigs=self.values, *args, **kwargs)  # type: ignore

    def plot_distribution(self, *args, **kwargs) -> PlotResult:  # type: ignore
        return rawEigDist(eigs=self.values, *args, **kwargs)  # type: ignore

    def plot_steps(self, *args, **kwargs) -> PlotResult:  # type: ignore
        return stepFunction(eigs=self.values, *args, **kwargs)  # type: ignore

    def plot_spacings(self, *args, **kwargs) -> PlotResult:  # type: ignore
        return plotSpacings(unfolded=self.values, *args, **kwargs)  # type: ignore

