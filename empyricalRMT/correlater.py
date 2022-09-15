from typing import cast

import numpy as np

from empyricalRMT._types import fArr


def correlate_fast(arr: fArr, ddof: int = 0) -> fArr:
    """Just a copy of the np.corrcoef source, with extras removed"""
    # TODO: use dask or memmaps to allow larger correlation matrices
    c = np.cov(arr, ddof=ddof)
    d = np.diag(c).reshape(-1, 1)
    sd = np.sqrt(d)
    c /= sd
    c /= sd.T
    np.clip(c, -1, 1, out=c)
    return cast(fArr, c)
