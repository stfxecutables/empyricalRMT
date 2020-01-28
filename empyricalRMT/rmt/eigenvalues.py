from numpy import ndarray
from pandas import DataFrame

from empyricalRMT.rmt._eigvals import EigVals


class Unfolded(EigVals):
    def __init__(self, eigenvalues):
        super().__init__(eigenvalues)

    @property
    def values(self) -> ndarray:
        return self._vals

    @property
    def vals(self) -> ndarray:
        return self._vals


class Trimmed(EigVals):
    def __init__(self, eigenvalues: ndarray, trimmed: ndarray):
        super().__init__(eigenvalues)
        self._trim_indices = None
        self._trim_report = None
        self._vals = trimmed

    @property
    def values(self) -> ndarray:
        return self._vals

    @property
    def vals(self) -> ndarray:
        return self._vals

    @property
    def trim_indices(self) -> (int, int):
        raise NotImplementedError
        return self._trim_indices

    @property
    def trim_report(self) -> DataFrame:
        raise NotImplementedError
        return self._trim_report

    def plot_trimmed(self):
        raise NotImplementedError

    def unfold(self) -> Unfolded:
        raise NotImplementedError
        return


class Eigenvalues(EigVals):
    def __init__(self, eigenvalues):
        super().__init__(eigenvalues)

    @property
    def values(self) -> ndarray:
        return self._vals

    @property
    def vals(self) -> ndarray:
        return self._vals

    def trim(self) -> Trimmed:
        raise NotImplementedError

    def trim_manually(self, start, end) -> Trimmed:
        raise NotImplementedError

    def trim_interactively(self) -> Trimmed:
        raise NotImplementedError

    def trim_unfold(self) -> Unfolded:
        raise NotImplementedError

    def unfold(self) -> Unfolded:
        raise NotImplementedError

