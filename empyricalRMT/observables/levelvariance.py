from typing import Tuple

import numpy as np
from numba import jit, prange
from numpy import bool_
from numpy import float64 as f64
from numpy.typing import NDArray

from empyricalRMT._constants import LEVELVAR_PROG, PERCENT, PROG_FREQUENCY


def level_number_variance(
    unfolded: NDArray[f64],
    L: NDArray[f64],
    tol: float,
    max_L_iters: int,
    min_L_iters: int,
    show_progress: bool,
) -> Tuple[NDArray[f64], NDArray[f64], NDArray[bool_]]:
    """Compute the level number variance of the current unfolded eigenvalues.

    Parameters
    ----------
    L: NDArray[np.float64]
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
    L_vals: NDArray[np.float64]
        The L_values for which the number level variance was computed.

    sigma: NDArray[np.float64]
        The computed number level variance values.

    convergences: NDArray[np.bool_]
        An array of length of L_vals and sigma with True where the convergence
        criterion was met within `max_L_iters`.
    """

    return compute_sigmas(
        unfolded=unfolded,
        L=L,
        tol=tol,
        max_L_iters=max_L_iters,
        min_L_iters=min_L_iters,
        show_progress=show_progress,
    )


@jit(nopython=True, cache=False, fastmath=True, parallel=True)
def compute_sigmas(
    unfolded: NDArray[f64],
    L: NDArray[f64],
    tol: float,
    max_L_iters: int,
    min_L_iters: int,
    show_progress: bool = True,
) -> Tuple[NDArray[f64], NDArray[f64], NDArray[bool_]]:
    """Compute the level number variance of the current unfolded eigenvalues.

    Parameters
    ----------
    unfolded: NDArray[np.float64]
        Array of unfolded eigenvalues

    L: NDArray[np.float64]
        The grid of L values for which to compute the level variance.

    tol: float
        Stop iterating when the last `min_L_iters` computed values of the
        level variance have a range (i.e. max - min) < tol.

    max_L_iters: int
        Stop computing values for the level variance once max_L_iters values
        have been computed for each L value.

    min_L_iters: int
        Minimum number of iterations for each L value.

    progress: bool = True
        Whether or not to display computation progress in stdout.

    Returns
    -------
    L_vals: ndarray
        The L_values for which the number level variance was computed.

    sigma: ndarray
        The computed number level variance values.

    convergences: ndarray
    """
    # the copy and different variable is needed here in the parallel context
    # https://github.com/numba/numba/issues/3652
    L_vals = np.copy(L).ravel()
    all_sigmas = np.zeros_like(L_vals)
    converged: NDArray[bool_] = np.zeros_like(L_vals, dtype=bool_)

    prog_interval = len(L_vals) // PROG_FREQUENCY
    if prog_interval == 0:
        prog_interval = 1
    # We can save memory, at the cost of some time, by generating random numbers
    # in each L process / thread. It is OK that the RNG might be cloned here,
    # because the c-values sampled in each process / thread will still be
    # uniformly distributed, and allow for proper integration.
    if show_progress:
        print(LEVELVAR_PROG, 0, PERCENT)
    for i in prange(len(L_vals)):
        all_sigmas[i], converged[i] = _sigma_L(
            unfolded=unfolded,
            L=L_vals[i],
            max_iters=max_L_iters,
            tol=tol,
            min_iters=min_L_iters,
        )
        if show_progress and (i % prog_interval == 0):
            prog = int(100 * np.sum(all_sigmas > 0) / len(L_vals))
            print(LEVELVAR_PROG, prog, PERCENT)

    return L_vals, all_sigmas, converged


@jit(nopython=True, cache=False, fastmath=True)
def _sigma_L(
    unfolded: NDArray[f64],
    L: float,
    max_iters: int,
    tol: float,
    min_iters: int,
) -> Tuple[float, bool]:
    """Compute the level number variance of the current unfolded eigenvalues.

    Parameters
    ----------
    unfolded: ndarray
        Array of unfolded eigenvalues

    L: float
        The current L value to use for computation.

    max_iters: int
        Max number of iterations

    tol: float
        Stop iterating when the last 1000 `min_iters` computed values of the
        level variance have a range (i.e. max - min) < tol

    min_iters: int
        Minimum number of iterations


    Returns
    -------
    sigma: float
        The computed level variance for L.

    converged: bool
        True if convergence criterion was met.

    Notes
    -----
    Computes the level number variance by randomly selecting a point c in
    the interval [unfolded.min(), unfolded.max()], and counts the number
    of unfolded eigenvalues in (c - L/2, c + L/2) to determine a value for
    the level number variance, sigma(L). The process is repeated until the
    running averages stabilize, and the final running average is returned.

    TODO: See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    to improve numerica stability
    """
    level_mean = 0.0
    level_sq_mean = 0.0
    sigma: float = 0.0
    size = 1000
    sigmas = np.zeros((size), dtype=np.float64)  # hold the last `size` running averages
    mn, mx = np.min(unfolded), np.max(unfolded)

    c = np.random.uniform(mn, mx)
    start, end = c - L / 2, c + L / 2
    n_within = len(unfolded[(unfolded >= start) & (unfolded <= end)])
    n_within_sq = n_within * n_within
    level_mean, level_sq_mean = n_within, n_within_sq
    sigma = level_sq_mean - level_mean * level_mean
    sigmas[0] = sigma

    # we'll use the fact that for x = [x_0, x_1, ... x_n-1], the
    # average a_k == (k*a_(k-1) + x_k) / (k+1) for k = 0, ..., n-1
    k = np.uint64(0)
    while True:
        k += 1
        c = np.random.uniform(mn, mx)
        start, end = c - L / 2, c + L / 2
        n_within = len(unfolded[(unfolded >= start) & (unfolded <= end)])
        n_within_sq = n_within * n_within
        level_mean = (k * level_mean + n_within) / (k + 1)
        level_sq_mean = (k * level_sq_mean + n_within_sq) / (k + 1)
        sigma = level_sq_mean - level_mean * level_mean
        sigmas[int(k) % size] = sigma
        if k > min_iters and (k % 500 == 0) and (np.abs(np.max(sigmas) - np.min(sigmas)) < tol):
            break
        if k >= max_iters:
            break

    converged = np.abs(np.max(sigmas) - np.min(sigmas)) < tol
    return sigma, converged
