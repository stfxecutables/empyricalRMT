import os
import sys
from pathlib import Path
from sys import stderr
from typing import Any, Optional, Tuple

import numpy as np
from numba import jit
from numpy import float64 as f64
from numpy import ndarray


class ConvergenceError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


def res(path: Path) -> str:
    return str(path.absolute().resolve())


def eprint(*args: Any, **kwargs: Any) -> None:
    print(*args, file=stderr, **kwargs)


def log(label: str, var: Any) -> None:
    eprint(f"{label}: {var}")


# https://stackoverflow.com/a/42913743
def is_symmetric(a: ndarray, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    return bool(np.allclose(a, a.T, rtol=rtol, atol=atol))


# def make_cheaty_nii(orig: Nifti1Image, array: ndarray) -> Nifti1Image:
#     """clone the header and extraneous info from `orig` and data in `array`
#     into a new Nifti1Image object, for plotting
#     """
#     affine = orig.affine
#     header = orig.header
#     return nib.Nifti1Image(dataobj=array, affine=affine, header=header)


def mkdirp(path: Path) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        print(
            f"Error making directory {path}. Another program may have modified the file "
            "while this script was running.",
            file=sys.stderr,
        )
        print("Original error:", file=sys.stderr)
        raise e


def make_directory(path: Path) -> Path:
    if not os.path.exists(path):
        try:
            os.makedirs(path)
            return path
        except Exception as e:
            print(
                f"Error making directory {path}. Another program likely "
                "modified it while this script was running.",
                file=sys.stderr,
            )
            print("Original error:", file=sys.stderr)
            raise e
    else:
        return path


def make_parent_directories(path: Path) -> None:
    path = path.absolute()
    paths = []
    for folder in path.parents:
        if folder != Path.home():
            paths.append(folder)
        else:
            break
    paths.reverse()

    for path in paths:
        make_directory(path)


@jit(nopython=True)
def kahan_add(current_sum: f64, update: f64, carry_over: f64) -> Tuple[f64, f64]:
    """
    Returns
    -------
    updated_sum: float
        Updated sum.

    carry_over: float
        Carried-over value (often named "c") in pseudo-code.
    """
    remainder = update - carry_over
    lossy = current_sum + remainder
    c = (lossy - current_sum) - remainder
    updated_sum = lossy
    return updated_sum, c


@jit(nopython=True, fastmath=True)
def nd_find(arr: ndarray, value: Any) -> Optional[int]:
    for i, val in np.ndenumerate(arr):
        if val == value:
            return i  # type: ignore
    return None


@jit(nopython=True)
def find_first(arr: ndarray, value: Any) -> int:
    for i, val in enumerate(arr):
        if val == value:
            return i  # type: ignore
    return -1


@jit(nopython=True)
def find_last(arr: ndarray, value: Any) -> int:
    for i in range(len(arr)):
        j = len(arr) - i - 1
        if arr[j] == value:
            return j  # type: ignore
    return -1


def flatten_4D(img4D: ndarray) -> np.ndarray:
    if isinstance(img4D, ndarray):
        return img4D.reshape((np.prod(img4D.shape[0:-1]),) + (img4D.shape[-1],))


@jit(nopython=True, fastmath=True)
def slope(x: ndarray, y: ndarray) -> np.float64:
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_dev = x - x_mean
    y_dev = y - y_mean
    cov = np.sum(x_dev * y_dev)
    var = np.sum(x_dev * x_dev)
    if var == 0:
        return np.float64(0.0)
    return np.float64(cov / var)


@jit(nopython=True, fastmath=True)
def variance(arr: ndarray) -> np.float64:
    """i.e. s^2"""
    n = len(arr)
    scale = 1.0 / (n - 1.0)
    mean = np.mean(arr)
    diffs = arr - mean
    squares = diffs ** 2
    summed = np.sum(squares)
    return np.float64(scale * summed)


@jit(nopython=True, fastmath=True)
def intercept(x: ndarray, y: ndarray, slope: np.float64) -> np.float64:
    return np.float64(np.mean(y) - slope * np.mean(x))


@jit(nopython=True, fastmath=True)
def fast_r(x: ndarray, y: ndarray) -> np.float64:
    n = len(x)
    num = x * y - n * np.mean(x) * np.mean(y)
    denom = (n - 1) * np.sqrt(variance(x)) * np.sqrt(variance(y))
    if denom == 0:
        return np.float64(0.0)
    return np.float64(num / denom)


# termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
