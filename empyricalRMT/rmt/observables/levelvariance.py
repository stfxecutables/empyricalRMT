import numpy as np
from numpy import ndarray

from colorama import Fore
from numba import jit, prange
from progressbar import AdaptiveETA, Percentage, ProgressBar, Timer
from typing import Tuple
from ...utils import eprint


def sigmaSquared(
    eigs: ndarray,
    unfolded: ndarray,
    c_iters: int = 50,
    L_grid_size: int = 100,
    min_L=0.5,
    max_L=20,
) -> Tuple[ndarray, ndarray]:
    """Compute the level number variance for a particular unfolding.

    Computes the level number variance (sigma squared, ∑² [1]_) for a
    particular set of eigenvalues and their unfolding.

    Parameters
    ----------
    eigs : ndarray
        The sorted (ascending) eigenvalues.
    unfolded : ndarray
        The sorted (ascending) eigenvalues computed from eigs.
    L_grid_size : int = 100
        The number of values of L to generate betwen min_L and max_L.
    min_L : int = 0.5
        The lowest possible L value for which to compute the spectral
        rigidity.
    max_L : int = 20
        The largest possible L value for which to compute the spectral
        rigidity.
    c_iters: int = 50
        How many times the location of the center, c, of the interval
        [c - L/2, c + L/2] should be chosen uniformly at random for
        each L in order to compute the estimate of the number level
        variance.

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
    L_grid = np.linspace(min_L, max_L, L_grid_size)
    sigma_sq = np.empty([L_grid_size])
    pbar_widgets = [
        f"{Fore.GREEN}Computing ∑²: {Fore.RESET}",
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
        sigma_sq[i] = sigma_iter(eigs, unfolded, L, c_iters)
        pbar.update(i)
    pbar.finish()

    return L_grid, sigma_sq


@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def sigma_iter(
    eigs: ndarray, unfolded: ndarray, L: float, c_iters: int = 100
) -> ndarray:
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


def sigmaSquared_exhaustive(
    unfolded: ndarray, c_step=0.05, L_grid_size: int = 100, max_L=50
) -> ndarray:
    if max_L is None:
        max_L = 0.2 * unfolded[-1] - unfolded[0]
    L_grid = np.linspace(0.001, max_L, L_grid_size)  # don't want L==0
    sigma_sq = np.empty([L_grid_size])
    for i, L in enumerate(L_grid):
        eprint(f"Beginning ∑² iterations for L value #{i} ({L})")
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
