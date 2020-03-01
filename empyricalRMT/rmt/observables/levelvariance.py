import ctypes
import numpy as np
import time
from numpy import ndarray

from colorama import Fore
from multiprocessing import Array, Process
from numba import jit, prange
from progressbar import AdaptiveETA, AnimatedMarker, Percentage, ProgressBar, Timer
from typing import Any, Tuple
from ...utils import eprint


def level_number_variance(
    unfolded: ndarray,
    min_L: float = 0.5,
    max_L: float = 20,
    c_iters: int = 50,
    L_grid_size: int = None,
    show_progress: bool = True,
) -> Tuple[ndarray, ndarray]:
    """Compute the level number variance for a particular unfolding.

    Computes the level number variance (sigma squared, [1]_) for a
    particular set of eigenvalues and their unfolding.

    Parameters
    ----------
    unfolded: ndarray
        The unfolded eigenvalues.
    L_grid_size: int
        The number of values of L to generate betwen min_L and max_L.
    min_L: int
        The lowest possible L value for which to compute the spectral
        rigidity.
    max_L: int
        The largest possible L value for which to compute the spectral
        rigidity.
    c_iters: int
        How many times the location of the center, c, of the interval
        [c - L/2, c + L/2] should be chosen uniformly at random for
        each L in order to compute the estimate of the number level
        variance.
    show_progress: bool
        Show a pretty progress bar while computing.

    Returns
    -------
    L : ndarray
        The L values generated based on the values of L_grid_size,
        min_L, and max_L.
    sigma_sq : ndarray
        The computed number level variance values for each of L.

    References
    ----------
    .. [1] Mehta, M. L. (2004). Random matrices (Vol. 142). Elsevier.
    """
    if L_grid_size is None:
        L_grid_size = int(2 * np.abs((np.floor(max_L) - np.floor(min_L))))
    L_grid = np.linspace(min_L, max_L, L_grid_size)
    sigma_sq = np.empty([L_grid_size])
    if show_progress:
        pbar_widgets = [
            f"{Fore.GREEN}Computing level variance: {Fore.RESET}",
            f"{Fore.BLUE}",
            Percentage(),
            f" {Fore.RESET}",
            " ",
            Timer(),
            f" | {Fore.YELLOW}",
            AdaptiveETA(),
            f"{Fore.RESET}",
        ]
        pbar = ProgressBar(widgets=pbar_widgets, maxval=L_grid.shape[0]).start()
    for i, L in enumerate(L_grid):
        sigma_sq[i] = _sigma_iter(unfolded, L, c_iters)
        if show_progress:
            pbar.update(i)
    if show_progress:
        pbar.finish()

    return L_grid, sigma_sq


@jit(nopython=True, cache=True, fastmath=True, parallel=True)
def _sigma_iter(unfolded: ndarray, L: float, c_iters: int = 100) -> ndarray:
    c_iters = np.min(np.array([int(L * 2000), int(25000)]))
    levels = np.empty((c_iters), dtype=np.float64)
    levels_sq = np.empty((c_iters), dtype=np.float64)  # levels squared
    for i in prange(c_iters):
        c_start = np.random.uniform(np.min(unfolded), np.max(unfolded))
        start, end = c_start - L / 2, c_start + L / 2
        # count number of eigenvalues within the current interval
        n_within = len(unfolded[(unfolded >= start) & (unfolded <= end)])
        n_within_sq = n_within * n_within
        levels[i] = n_within
        levels_sq[i] = n_within_sq

    ave = np.mean(levels)
    av_of_levels_sq = ave * ave
    av_of_sq_levels = np.mean(levels_sq)
    return av_of_sq_levels - av_of_levels_sq


def level_number_variance_stable(
    unfolded: ndarray,
    min_L: float = 0.5,
    max_L: float = 20,
    L_grid_size: int = None,
    tol: float = 0.01,
    max_L_iters: int = 10000,
    min_L_iters: int = 100,
    show_progress: bool = True,
) -> Tuple[ndarray, ndarray]:
    if L_grid_size is None:
        L_grid_size = 2 * np.floor(max_L - min_L)
    if show_progress:

        def update_progress(shared: Array, N: int) -> None:
            pbar_widgets = [
                f"{Fore.GREEN}Computing level variance: {Fore.RESET}",
                f"{Fore.BLUE}",
                Percentage(),
                f" {Fore.RESET}",
                " ",
                Timer(),
                f" {Fore.YELLOW}",
                AnimatedMarker(),
                f"{Fore.RESET}",
            ]
            pbar = ProgressBar(widgets=pbar_widgets, maxval=N).start(
                max_value=N, init=True
            )
            progress = np.frombuffer(shared.get_obj())
            done = int(progress[0])
            while done < N:  # type: ignore
                done = int(progress[0])
                pbar.update(done)
                # time.sleep(0.5)

        # see https://stackoverflow.com/a/9754423, https://stackoverflow.com/a/7908612
        # here we make the numpy arrays share the same memory
        shared = Array(ctypes.c_uint64, 1)
        progress = np.frombuffer(shared.get_obj())
        update = Process(target=update_progress, args=(shared, L_grid_size))
        update.start()
        L, sigma = _sigma_iter_converge(
            unfolded=unfolded,
            min_L=min_L,
            max_L=max_L,
            L_grid_size=L_grid_size,
            tol=tol,
            max_L_iters=max_L_iters,
            min_L_iters=min_L_iters,
            progress=progress,
        )
        progress[0] = L_grid_size  # ensure finish in case increments are non-atomic
        update.join()

    return L, sigma


@jit(nopython=True, cache=True, fastmath=True, parallel=True)
def _sigma_iter_converge(
    unfolded: ndarray,
    min_L: float = 0.5,
    max_L: float = 20,
    L_grid_size: int = None,
    tol: float = 0.1,
    max_L_iters: int = 10000,
    min_L_iters: int = 100,
    progress: ndarray = None,
) -> Tuple[ndarray, ndarray]:
    L = np.linspace(min_L, max_L, L_grid_size)
    sigma = np.empty((L_grid_size), np.float64)
    for i in prange(L_grid_size):
        tol_modified = tol + tol * (L[i] / 5.0)
        sigma[i] = _sigma_iter_converge_L(
            unfolded, L[i], tol_modified, max_L_iters, min_L_iters
        )
        if progress is not None:
            progress[0] += 1
            # print("Done", np.sum(progress) * 100 / L_grid_size)
    return L, sigma


@jit(nopython=True, cache=True, fastmath=True)
def _sigma_iter_converge_L(
    unfolded: ndarray,
    L: float,
    tol: float = 1e-6,
    max_iters: int = 100000,
    min_iters: int = 1000,
) -> Any:
    # if L < 50:
    #     tol = 0.0001
    level_mean = 0.0
    level_sq_mean = 0.0
    sigma = 0.0
    size = 50
    sigmas = np.zeros((size), dtype=np.float64)

    # initialize
    c = np.random.uniform(np.min(unfolded), np.max(unfolded))
    start, end = c - L / 2, c + L / 2
    n_within = len(unfolded[(unfolded >= start) & (unfolded <= end)])
    n_within_sq = n_within * n_within
    level_mean, level_sq_mean = n_within, n_within_sq
    sigma = level_sq_mean - level_mean * level_mean
    sigmas[0] = sigma

    # we'll use the fact that for x = [x_0, x_1, ... x_n-1], the
    # average a_k == (k*a_(k-1) + x_k) / (k+1) for k = 0, ..., n-1
    k = 0
    while True:
        k += 1
        c = np.random.uniform(np.min(unfolded), np.max(unfolded))
        start, end = c - L / 2, c + L / 2
        n_within = len(unfolded[(unfolded >= start) & (unfolded <= end)])
        n_within_sq = n_within * n_within
        level_mean = (k * level_mean + n_within) / (k + 1)
        level_sq_mean = (k * level_sq_mean + n_within_sq) / (k + 1)
        sigma = level_sq_mean - level_mean * level_mean
        sigmas[k % size] = sigma
        if np.abs(np.max(sigmas) - np.min(sigmas)) < tol and k > min_iters:
            break
        if k > max_iters:
            break

    return sigma


def _sigmaSquared_exhaustive(
    unfolded: ndarray, c_step: float = 0.05, L_grid_size: int = 100, max_L: float = 50
) -> ndarray:
    L_grid = np.linspace(0.001, max_L, L_grid_size)  # don't want L==0
    sigma_sq = np.empty([L_grid_size])
    for i, L in enumerate(L_grid):
        c_start = unfolded[0]
        c_end = unfolded[0] + L
        levels, levels_sq = [], []
        while c_end < unfolded[-1]:
            n_within = len(unfolded[(unfolded >= c_start) & (unfolded <= c_end)])
            n_within_sq = n_within * n_within
            levels.append(n_within)
            levels_sq.append(n_within_sq)
            c_start += c_step
            c_end += c_step
        sigma_sq[i] = np.var(levels) - np.var(levels) / L

    return np.array([L_grid, sigma_sq], dtype=float).T
