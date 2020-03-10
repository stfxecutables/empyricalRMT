import numpy as np
from numpy import ndarray


def correlate_fast(arr: ndarray, ddof: int = 0) -> ndarray:
    """Just a copy of the np.corrcoef source, with extras removed"""
    # TODO: use dask or memmaps to allow larger correlation matrices
    c = np.cov(arr, ddof=ddof)
    d = np.diag(c).reshape(-1, 1)

    sd = np.sqrt(d)
    c /= sd
    c /= sd.T
    np.clip(c, -1, 1, out=c)

    return c
