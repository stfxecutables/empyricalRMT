from numpy import ndarray

from empyricalRMT.rmt._eigvals import EigVals


class Unfolded(EigVals):
    def __init__(self, eigenvalues: ndarray):
        super().__init__(eigenvalues)

    @property
    def values(self) -> ndarray:
        return self._vals

    @property
    def vals(self) -> ndarray:
        return self._vals
