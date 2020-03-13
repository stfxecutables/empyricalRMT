import ctypes
import numpy as np
from numpy import ndarray

from colorama import Fore
from multiprocessing import Array, Process
from numba import jit, prange
from progressbar import AdaptiveETA, AnimatedMarker, Percentage, ProgressBar, Timer
from typing import Any, Tuple


def level_number_variance(
    unfolded: ndarray,
    min_L: float = 0.5,
    max_L: float = 20,
    c_iters: int = 50,
    L_grid_size: int = None,
    show_progress: bool = True,
) -> Tuple[ndarray, ndarray]:
    """[DEPRECATE]. This has been supersed by level_number_variance_stable

    Computes the level number variance (sigma squared, [1]_) for a
    particular set of eigenvalues and their unfolding.

    Parameters
    ----------
    unfolded: ndarray
        The unfolded eigenvalues.

    L_grid_size: int
        The number of values of L to generate betwen min_L and max_L.

    min_L: int
        The lowest possible L value for which to compute the spectral rigidit

    max_L: int
        The largest possible L value for which to compute the spectral rigidit

    c_iters: int
        How many times the location of the center, c, of the interval
        [c - L/2, c + L/2] should be chosen uniformly at random for
        each L in order to compute the estimate of the number level
        varianc

    show_progress: bool
        Show a pretty progress bar while computin

    Returns
    -------
    L : ndarray
        The L values generated based on the values of L_grid_size,
        min_L, and max_

    sigma_sq : ndarray
        The computed number level variance values for each of

    References
    ----------
    .. [1] Mehta, M. L. (2004). Random matrices (Vol. 142). Elsevier.

    Notes
    -----
    The level number variance Sigma^2(L) converges *much* more slowly to the
    expected limiting curves than either the NNSD, next NNSD, or spectral
    rigidity. Even for a (50 000 x 50 000) GOE matrix, unfolded via smooth /
    Wigner's unfolding, there are significant deviations (about 0.1 in absolute
    value) from the expected values for

    For smaller matrices (e.g. 5000 x 5000) there will generally be considerable
    deviance by L == 20. In general, one should probably refrain from computing
    the number level variance beyond L == 20.
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
    """[DEPRECATE] Use _sigma_iter_converge"""
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
    L: ndarray,
    tol: float,
    max_L_iters: int,
    min_L_iters: int,
    show_progress: bool,
) -> Tuple[ndarray, ndarray]:
    """Compute the level number variance of the current unfolded eigenvalues.

    Parameters
    ----------
    L: ndarray
        The grid of L values for which to compute the level variance.

    tol: float
        Stop iterating when the last `min_L_iters` computed values of the
        level variance have a range (i.e. max - min) < tol.

    max_L_iters: int
        Stop computing values for the level variance once max_L_iters values
        have been computed for each L value.

    min_L_iters: int
        Minimum number of iterations for each L value.

    show_progress: bool
        Whether or not to display computation progress in stdout.

    Returns
    -------
    L_vals: ndarray
        The L_values for which the number level variance was computed.

    sigma: ndarray
        The computed number level variance values.
    """
    if show_progress:
        # see https://stackoverflow.com/a/9754423, https://stackoverflow.com/a/7908612
        # here we make the numpy arrays share the same memory
        shared = Array(ctypes.c_uint64, 1)
        progress = np.frombuffer(shared.get_obj())
        update = Process(target=_update_progress, args=(shared, len(L)))
        update.start()

    L_vals, sigma = _sigma_iter_converge(
        unfolded=unfolded,
        L=L,
        tol=tol,
        max_L_iters=max_L_iters,
        min_L_iters=min_L_iters,
        progress=progress if show_progress else None,
    )
    if show_progress:
        progress[0] = len(L)  # ensure finish in case increments are non-atomic
        update.join()

    return L_vals, sigma


@jit(nopython=True, cache=True, fastmath=True, parallel=True)
def _sigma_iter_converge(
    unfolded: ndarray,
    L: ndarray,
    tol: float,
    max_L_iters: int,
    min_L_iters: int,
    progress: ndarray,
) -> Tuple[ndarray, ndarray]:
    """Compute the level number variance of the current unfolded eigenvalues.

    Parameters
    ----------
    L: ndarray
        The grid of L values for which to compute the level variance.

    tol: float
        Stop iterating when the last `min_L_iters` computed values of the
        level variance have a range (i.e. max - min) < tol.

    max_L_iters: int
        Stop computing values for the level variance once max_L_iters values
        have been computed for each L value.

    min_L_iters: int
        Minimum number of iterations for each L value.

    progress: bool
        Whether or not to display computation progress in stdout.

    Returns
    -------
    L_vals: ndarray
        The L_values for which the number level variance was computed.

    sigma: ndarray
        The computed number level variance values.
    """
    # the copy and different variable is needed here in the parallel context
    # https://github.com/numba/numba/issues/3652
    L_vals = np.copy(L)
    sigma = np.empty(L_vals.shape, dtype=np.float64)
    for i in prange(L_vals.shape[0]):
        # tol_modified = tol + tol * (L[i] / 5.0)
        tol_modified = tol
        sigma[i] = _sigma_iter_converge_L(
            unfolded, L_vals[i], tol_modified, max_L_iters, min_L_iters
        )
        if progress is not None:
            progress[0] += 1
    return L_vals, sigma


@jit(nopython=True, cache=True, fastmath=True)
def _sigma_iter_converge_L(
    unfolded: ndarray, L: float, tol: float, max_iters: int, min_iters: int
) -> Any:
    """Compute the level number variance of the current unfolded eigenvalues.

    Parameters
    ----------
    L: float
        The current L value to use for computation.

    tol: float
        Stop iterating when the last `min_iters` computed values of the
        level variance have a range (i.e. max - min) < tol

    max_iters: int
        Stop computing values for the level variance once max_iters values
        have been computed

    min_iters: int
        Minimum number of iterations


    Returns
    -------
    sigma: float
        The computed level variance for L.

    Notes
    -----
    Computes the level number variance by randomly selecting a point c in
    the interval [unfolded.min(), unfolded.max()], and counts the number
    of unfolded eigenvalues in (c - L/2, c + L/2) to determine a value for
    the level number variance, sigma(L). The process is repeated until the
    running averages stabilize, and the final running average is returned.
    """
    level_mean = 0.0
    level_sq_mean = 0.0
    sigma = 0.0
    size = min_iters
    sigmas = np.zeros((size), dtype=np.float64)  # hold the last `size` running averages

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


def _update_progress(shared: Array, N: int) -> None:
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
    pbar = ProgressBar(widgets=pbar_widgets, maxval=N).start()
    progress = np.frombuffer(shared.get_obj())
    done = int(progress[0])
    while done < N:  # type: ignore
        done = int(progress[0])
        pbar.update(done)
    pbar.finish()
