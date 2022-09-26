from typing import Tuple

import numpy as np
from numba import jit, prange
from numpy import bool_
from numpy import float64 as f64
from numpy import uint64 as u64

from empyricalRMT._constants import (
    AUTO_MAX_ITERS,
    CONVERG_PROG,
    CONVERG_PROG_INTERVAL,
    ITER_COUNT,
    LEVELVAR_PROG,
    MIN_ITERS,
    PERCENT,
    PROG_FREQUENCY,
)
from empyricalRMT._types import bArr, fArr, uArr
from empyricalRMT.utils import ConvergenceError, kahan_add


def level_number_variance(
    unfolded: fArr,
    L: fArr,
    tol: float = 0.01,
    max_iters: int = 0,
    show_progress: bool = True,
) -> Tuple[fArr, fArr, bArr, uArr]:
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

    show_progress: bool
        Whether or not to display computation progress in stdout.

    Returns
    -------
    L_vals: NDArray[np.float64]
        The L_values for which the number level variance was computed.

    sigma: NDArray[np.float64]
        The computed number level variance values.

    convergences: bArr
        An array of length of L_vals and sigma with True where the convergence
        criterion was met within `max_L_iters`.
    """
    if max_iters <= 0:
        success, max_iters = sigma_L(
            unfolded=unfolded,
            L=float(np.max(L)),
            tol=tol,
            max_iters=AUTO_MAX_ITERS,
            min_iters=MIN_ITERS * 10,
            show_progress=show_progress,
        )[
            1:
        ]  # type: ignore
        max_iters = int(max_iters * 2)
        if not success:
            raise ConvergenceError(
                f"For the largest L-value in your provided Ls, {np.max(L)}, the "
                f"level variance at L did not converge in {AUTO_MAX_ITERS} "
                "iterations. Either reduce the range of L values, reduce the "
                "`tol` tolerance value, or manually set `max_iters` to be some "
                "value other than the default of 0 to disable this check. Note "
                "the convergence criterion involves the range on the last 1000 "
                "values, which are themselves iteratively-computed means, so is "
                "a somewhat strict convergence criterion. However, setting "
                "`max_iters` too low then provides NO guarantee on the error for "
                "non-converging L values."
            )
    return sigma_parallel(  # type: ignore
        unfolded=unfolded,
        L=L,
        tol=tol,
        max_L_iters=max_iters,
        min_L_iters=MIN_ITERS,
        show_progress=show_progress,
    )


@jit(nopython=True, cache=False, fastmath=True, parallel=True)
def sigma_parallel(
    unfolded: fArr,
    L: fArr,
    tol: float,
    max_L_iters: int,
    min_L_iters: int,
    show_progress: bool = True,
) -> Tuple[fArr, fArr, bArr, uArr]:
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
    iters = np.empty_like(L_vals, dtype=np.uint64)
    converged: bArr = np.zeros_like(L_vals, dtype=bool_)

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
        all_sigmas[i], converged[i], iters[i] = sigma_L(
            unfolded=unfolded,
            L=L_vals[i],
            max_iters=max_L_iters,
            tol=tol,
            min_iters=min_L_iters,
        )
        if show_progress and (i % prog_interval == 0):
            prog = int(100 * np.sum(all_sigmas > 0) / len(L_vals))
            print(LEVELVAR_PROG, prog, PERCENT)
    if show_progress:
        print("")

    return L_vals, all_sigmas, converged, iters


@jit(nopython=True, cache=False, fastmath=True)
def sigma_L(
    unfolded: fArr,
    L: float,
    max_iters: int,
    tol: float,
    min_iters: int,
    show_progress: bool = False,
) -> Tuple[f64, bool, u64]:
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

    iters: int
        Count of iterations

    Notes
    -----
    Computes the level number variance by randomly selecting a point c in
    the interval [unfolded.min(), unfolded.max()], and counts the number
    of unfolded eigenvalues in (c - L/2, c + L/2) to determine a value for
    the level number variance, sigma(L). The process is repeated until the
    running averages stabilize, and the final running average is returned.
    """
    prog_interval = CONVERG_PROG_INTERVAL
    if L > 100:
        prog_interval *= 10
    level_mean = np.float64(0.0)
    level_sq_mean = np.float64(0.0)
    sigma = np.float64(0.0)
    size = 1000
    sigmas = np.zeros((size), dtype=np.float64)  # hold the last `size` running averages
    mn, mx = np.min(unfolded), np.max(unfolded)

    c = np.random.uniform(mn, mx)
    start, end = c - L / 2, c + L / 2
    n_within = len(unfolded[(unfolded >= start) & (unfolded <= end)])
    n_within_sq = n_within * n_within
    level_mean, level_sq_mean = np.float64(n_within), np.float64(n_within_sq)
    sigma = level_sq_mean - level_mean * level_mean
    sigmas[0] = sigma
    # for Kahan summation
    c_mean = np.float64(0.0)
    c_sq = np.float64(0.0)

    if show_progress:
        print(CONVERG_PROG, 0, ITER_COUNT)

    k = np.uint64(0)
    while True:
        k += 1
        c = np.random.uniform(mn, mx)
        start, end = c - L / 2, c + L / 2
        n_within = len(unfolded[(unfolded >= start) & (unfolded <= end)])
        n_within_sq = n_within * n_within
        level_mean_update = (n_within - level_mean) / k
        level_sq_mean_update = (n_within_sq - level_sq_mean) / k
        level_mean, c_mean = kahan_add(
            current_sum=level_mean, update=level_mean_update, carry_over=c_mean
        )
        level_sq_mean, c_sq = kahan_add(
            current_sum=level_sq_mean, update=level_sq_mean_update, carry_over=c_sq
        )

        # level_mean = (k * level_mean + n_within) / (k + 1)
        # level_sq_mean = (k * level_sq_mean + n_within_sq) / (k + 1)
        sigma = level_sq_mean - level_mean * level_mean
        sigmas[int(k) % size] = sigma
        if show_progress and k % prog_interval == 0:
            print(CONVERG_PROG, int(k * 2), ITER_COUNT)  # x2 for safety factor
        if k > min_iters and (k % 500 == 0) and (np.abs(np.max(sigmas) - np.min(sigmas)) < tol):
            break
        if k >= max_iters:
            break

    if show_progress:
        print(CONVERG_PROG, int(k), ITER_COUNT)
        print("")
    converged = np.abs(np.max(sigmas) - np.min(sigmas)) < tol
    return sigma, converged, k
