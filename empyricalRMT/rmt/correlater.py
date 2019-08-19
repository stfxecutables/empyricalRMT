import numpy as np

from numba import jit, prange
from pathlib import Path

from tempfile import TemporaryDirectory
from utils import setup_progressbar

from fmri.flatten import flatten_4D

OUT_DEFAULT = Path.home() / "Desktop" / "corrmat.npy"


def compute_correlations(
    arr: np.array, save: Path = OUT_DEFAULT, detrended=False, k=None, M=None
) -> Path:
    with TemporaryDirectory() as TEMPDIR:
        TEMPCORRS = Path(TEMPDIR) / "corrstemp.dat"
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
                compute_detrended_column_corrs(corrs, arr, i)
            else:
                compute_clean_column_corrs(corrs, arr, i)
            if i % 10 == 0:
                pass
                pbar.update(i)
        pbar.finish()

        np.save(save, corrs)
    return save


# @jit(nopython=True, cache=False, parallel=True, fastmath=True)
@jit(nopython=True, cache=False, parallel=True)
def compute_corrs(reshaped, iters):
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


@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def compute_column_corrs(corrs, reshaped, i):
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
def compute_clean_column_corrs(corrs, reshaped, i):
    # compute correlations
    # (!!), inefficient, upper triangle is sufficient
    def clean_correlate(x, y):
        x_norm = x - x.mean()
        y_norm = y - y.mean()
        x_mag = np.linalg.norm(x_norm)
        y_mag = np.linalg.norm(y_norm)
        numerator = np.dot(x_norm, y_norm)
        denom = x_mag * y_mag
        if np.abs(denom) == 0:
            return 0.0
        else:
            return numerator / denom

    voxel_count = reshaped.shape[0]
    for j in prange(voxel_count):
        if i > j:  # ignore lower triangle and diagonal
            continue
        elif i == j:
            corrs[i, j] = 1
        else:
            corr = clean_correlate(reshaped[i, :], reshaped[j, :])
            corrs[i, j] = corr
            corrs[j, i] = corr


@jit(nopython=True, fastmath=True, cache=True)
def fast_r_detrended(x: np.array, y: np.array) -> np.float64:
    num = np.sum(x * y)
    denom = np.sqrt(np.sum(x * x)) * np.sqrt(np.sum(y * y))
    if denom == 0:
        return 0
    return num / denom


@jit(nopython=True, parallel=True, fastmath=True)
def compute_detrended_column_corrs(corrs, reshaped, i):
    # compute correlations
    # (!!), inefficient, upper triangle is sufficient
    voxel_count = reshaped.shape[0]
    for j in prange(voxel_count):
        if i > j:  # ignore lower triangle and diagonal
            continue
        elif i == j:
            corrs[i, j] = 1
        else:
            corr = fast_r_detrended(reshaped[i, :], reshaped[j, :])
            corrs[i, j] = corr
            corrs[j, i] = corr


@jit(nopython=True, fastmath=True)
def fast_r(x: np.array, y: np.array) -> float:
    """i.e. s^2"""
    x2, y2 = x ** 2, y ** 2
    return x * y / np.sqrt(x2 * y2)
