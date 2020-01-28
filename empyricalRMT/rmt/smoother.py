import numpy as np

from numpy import ndarray
from pandas import DataFrame


class Smoother:
    def __init__(self, eigenvalues: ndarray):
        self._eigs = eigenvalues

    def fit(self, smoother="poly", emd_detrend=False) -> ndarray:
        raise NotImplementedError

    def fit_all(self, *args) -> DataFrame:
        raise NotImplementedError
