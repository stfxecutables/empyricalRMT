import numpy as np

from colorama import Fore
from numba import jit, prange
from progressbar import AdaptiveETA, Percentage, ProgressBar, Timer
from ...utils import eprint


# number variance (sigma squared)
# the variance of the number of unfolded eigenvalues in intervals of
# length L around each of the eigenvalues. If the eigenvalues are
# uncorrelated, sigma^2 ~ L, while for the case a “rigid” rigenvalue
# spectrum, sigma^2 ~ 0. For the GOE case, we find the “intermediate”
# behavior Sigma^2 ~ ln(L), as predictedby RMT
@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def sigma_iter(eigs: np.array, unfolded: np.array, L: float, c_iters: int = 100):
    levels = np.empty((c_iters), dtype=np.float64)
    levels_sq = np.empty((c_iters), dtype=np.float64)  # levels squared
    for i in prange(c_iters):
        # c_start = np.random.uniform(unfolded[0], unfolded[-1] - L)
        # c_start = np.random.uniform(np.min(unfolded), np.max(unfolded))
        c_start = np.random.uniform(np.min(unfolded), np.max(unfolded))
        start, end = c_start - L / 2, c_start + L / 2
        n_within = len(unfolded[(unfolded >= start) & (unfolded <= end)])
        n_within_sq = n_within * n_within
        levels[i] = n_within
        levels_sq[i] = n_within_sq

    ave = np.mean(levels)
    av_of_levels_sq = ave * ave
    av_of_sq_levels = np.mean(levels_sq)
    return av_of_sq_levels - av_of_levels_sq


def sigmaSquared(
    eigs: np.array,
    unfolded: np.array,
    c_iters: int = 50,
    L_grid_size: int = 100,
    min_L=0.5,
    max_L=20,
) -> [np.array, np.array]:
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


def sigmaSquared_exhaustive(
    unfolded: np.array, c_step=0.05, L_grid_size: int = 100, max_L=50
):
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

        # av = np.average(levels)
        # av_of_levels_sq = av * av
        # av_of_sq_levels = np.average(levels_sq)
        # sigma_sq[i] = (av_of_sq_levels - av_of_levels_sq)
        # sigma_sq[i] = 1 - np.var(levels)/L

        # because the approximations plotted are + O(L^(-1)), to the
        # order 1/L
        sigma_sq[i] = np.var(levels) - np.var(levels) / L

    return np.array([L_grid, sigma_sq], dtype=float).T


def sigmaSquared_rewrite():
    pass


def count_levels(unfolded: np.array, L: float, c_start: float) -> int:
    pass
