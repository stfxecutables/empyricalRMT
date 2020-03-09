import numpy as np
from numpy import ndarray


def correlate_fast(arr: ndarray) -> ndarray:
    """Just a copy of the np.corrcoef source, with extras removed"""
    c = np.cov(arr)
    d = np.diag(c).reshape(-1, 1)

    sd = np.sqrt(d)
    c /= sd
    c /= sd.T
    # np.clip(c, -1, 1, out=c)  # not needed for real matrices

    return c
