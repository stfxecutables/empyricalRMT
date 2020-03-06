import numpy as np
from numpy import ndarray

from numba import jit, prange
from pathlib import Path

from tempfile import TemporaryDirectory
from ..utils import setup_progressbar, flatten_4D


OUT_DEFAULT = Path.home() / "Desktop" / "corrmat.npy"


def p_correlate(arr: ndarray) -> ndarray:
    N = arr.shape[0]
    corrs = np.zeros([N, N], dtype=np.float64)
    pbar = setup_progressbar(f"Computing correlations", N).start()
    for i in range(N):
        _compute_clean_row_corrs(corrs, arr, i)
        pbar.update(i)
    pbar.finish()
    return corrs


def compute_correlations(
    arr: ndarray,
    save: Path = OUT_DEFAULT,
    detrended: bool = False,
    k: int = None,
    M: int = None,
) -> Path:
    with TemporaryDirectory() as TEMPDIR:
        TEMPCORRS = Path(str(TEMPDIR)) / "corrstemp.dat"
        if len(arr.shape) == 4:
            arr = flatten_4D(arr)
        elif len(arr.shape) == 2:
            pass
        else:
            raise Exception("Can only compute correlations for 2D or 4D array")

        # use memory maps
        corrs = np.memmap(
            TEMPCORRS, dtype="float32", mode="w+", shape=(arr.shape[0], arr.shape[0])
        )

        info = f" for image {k+1} of {M}" if k is not None and M is not None else ""

        pbar = setup_progressbar(f"Computing correlations{info}", arr.shape[0]).start()
        for i in range(arr.shape[0]):
            if detrended:
                _compute_detrended_column_corrs(corrs, arr, i)
            else:
                _compute_clean_row_corrs(corrs, arr, i)
            if i % 10 == 0:
                pass
                pbar.update(i)
        pbar.finish()

        np.save(save, corrs)
    return save


# @jit(nopython=True, cache=False, parallel=True, fastmath=True)
@jit(nopython=True, cache=False, parallel=True)
def _compute_corrs(reshaped: ndarray, iters: int) -> ndarray:
    # compute correlations
    # (!!), inefficient, upper triangle is sufficient
    corrs = np.empty((iters, iters), dtype=np.float64)

    for i in prange(iters):
        print("Computing correlation matrix column ", i)
        for j in prange(iters):
            if i >= j:  # ignore lower triangle and diagonal
                corrs[i, j] = 0
                continue
            else:
                corr_mat = np.corrcoef(reshaped[i, :], reshaped[j, :])
                corrs[i, j] = corr_mat[1, 0]
                # R = np.correlate(reshaped[i, :], reshaped[j, :])[0]
                # corrs[i, j] = R

    corrs = corrs + corrs.T + np.identity(iters)
    return corrs


@jit(nopython=True, fastmath=True, parallel=True)
def _compute_column_corrs(corrs: ndarray, reshaped: ndarray, i: int) -> None:
    # compute correlations
    # (!!), inefficient, upper triangle is sufficient
    voxel_count = reshaped.shape[0]

    for j in prange(voxel_count):
        if i >= j:  # ignore lower triangle and diagonal
            corrs[i, j] = 0
        else:
            corr_mat = np.corrcoef(reshaped[i, :], reshaped[j, :])
            corrs[i, j] = corr_mat[1, 0]
            # corrs[i, j] = np.correlate(reshaped[i, :], reshaped[j, :])[0]


# @jit(nopython=True, cache=True, fastmath=True)
# def clean_correlate(x, y):
#     x_norm = x - x.mean()
#     y_norm = y - y.mean()
#     x_mag = np.linalg.norm(x_norm)
#     y_mag = np.linalg.norm(y_norm)
#     numerator = np.dot(x_norm, y_norm)
#     denom = np.prod(x_mag, y_mag)
#     if np.abs(denom) == 0:
#         return 0.0
#     else:
#         return numerator / denom


@jit(nopython=True, parallel=True, fastmath=True)
def _compute_clean_row_corrs(corrs: ndarray, array: ndarray, i: int) -> None:
    """Compute correlations of row i of array and save them in row i of corrs.


    Parameters
    ----------
    corrs: ndarray
        The array to store the correlation values. Must be size (N, N) where
        N == array.shape[0].

    array: ndarray
        The N x T matrix of time series.

    i: int
        The index of the current row of interest.
    """
    N = array.shape[0]
    for j in prange(N):
        if i < j:  # ignore upper triangle
            continue
        elif i == j:
            corrs[i, j] = 1
        else:
            corr = _clean_correlate(array[i, :], array[j, :])
            corrs[i, j] = corr


@jit(nopython=True, fastmath=True, cache=True)
def _fast_r_detrended(x: ndarray, y: ndarray) -> np.float64:
    num = np.sum(x * y)
    denom = np.sqrt(np.sum(x * x)) * np.sqrt(np.sum(y * y))
    if denom == 0:
        return 0
    return num / denom


@jit(nopython=True, parallel=True, fastmath=True)
def _compute_detrended_column_corrs(corrs: ndarray, reshaped: ndarray, i: int) -> None:
    # compute correlations
    # (!!), inefficient, upper triangle is sufficient
    voxel_count = reshaped.shape[0]
    for j in prange(voxel_count):
        if i > j:  # ignore lower triangle and diagonal
            continue
        elif i == j:
            corrs[i, j] = 1
        else:
            corr = _fast_r_detrended(reshaped[i, :], reshaped[j, :])
            corrs[i, j] = corr
            corrs[j, i] = corr


@jit(nopython=True, fastmath=True)
def _fast_r(x: ndarray, y: ndarray) -> float:
    """i.e. s^2"""
    x2, y2 = x ** 2, y ** 2
    return float((x * y) / np.sqrt(x2 * y2))


@jit(nopython=True, fastmath=True)
def _clean_correlate(x: ndarray, y: ndarray) -> float:
    x_norm = x - x.mean()
    y_norm = y - y.mean()
    x_mag = np.linalg.norm(x_norm)
    y_mag = np.linalg.norm(y_norm)
    denom = x_mag * y_mag
    if np.abs(denom) == 0:
        return 0.0
    numerator = np.dot(x_norm, y_norm)
    return float(numerator / denom)


# @jit(nopython=True, parallel=True, fastmath=True)
@jit(nopython=True, fastmath=True, cache=True, parallel=True)
def _compute_upper_correlations(corrs: ndarray, array: ndarray) -> None:
    """Compute the upper triangle of correlation coefficients of `array`,
    and save the values in `corrs`. """
    n = array.shape[0]
    for i in prange(n):
        for j in range(n):
            if i < j:
                corr = _clean_correlate(array[i, :], array[j, :])
                corrs[i, j] = corr

