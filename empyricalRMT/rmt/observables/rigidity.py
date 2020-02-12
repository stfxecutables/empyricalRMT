import numpy as np
from numpy import ndarray

from colorama import Fore
from numba import jit, prange
from progressbar import AdaptiveETA, Percentage, ProgressBar, Timer
from typing import Tuple

from empyricalRMT.rmt.observables.step import stepFunctionG, stepFunctionVectorized


# spectral rigidity ∆3
# the least square devation of the unfolded cumulative eigenvalue
# density from a fit to a straight line in an interval of length L
# For uncorrelated eigenvalues, ∆3 ~ L, whereas for the rigid case,
# ∆3 = const. For the GOE case, ∆3 ~ ln(L)

# adapted from page 8 of:
# Feher, Kristen; Whelan, James; and Müller, Samuel (2011) "Assessing
# Modularity Using a Random Matrix Theory Approach," Statistical
# Applications in Genetics and Molecular Biology: Vol. 10: Iss. 1,
# Article 44.
# DOI: 10.2202/1544-6115.1667

# Algorithm 2. (∆3 statistic)
# 1. Eigenvalues of A are unfolded using Algorithm 1 to give
#    {l_1 , #    ... , l_r }.
# 2. Starting point c is chosen at random from the uniform distribution
#    U(l 1 , l r − 15).
# 3. L is chosen from U(2, 15)
#    (experience showed that probing at L < 2 generally yielded too
#    few eigenvalues).
# 4. If there were ≥ 3 eigenvalues l_i with c ≤ l_i ≤ c + L, a least
#    squares straight line was fit to G(l_i) in that interval to yield
#    κ and ω.
# 5. ∆3(L) is thus defined as: the definite integral of the
#    piecewise-linear function y = (G(l) − κl − ω) on the interval c ≤
#    l_i ≤ c + L, as per Equation (4). This process can be repeated an
#    appropriate number of times to generate a dataset consisting of
#    datapoints (L, ∆3(L)).
def spectralRigidity(
    eigs: ndarray,
    unfolded: ndarray,
    c_iters: int = 1000,
    L_grid_size: int = 50,
    min_L: float = 2,
    max_L: float = 25,
) -> Tuple[ndarray, ndarray]:
    """Compute the spectral rigidity for a particular unfolding.

    Computes the spectral rigidity (delta_3, ∆₃ [1]_) for a
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
        each L in order to compute the estimate of the spectral
        rigidity.

    Returns
    -------
    L : ndarray
        The L values generated based on the values of L_grid_size,
        min_L, and max_L.
    delta3 : ndarray
        The computed spectral rigidity values for each of L.

    References
    ----------
    .. [1] Mehta, M. L. (2004). Random matrices (Vol. 142). Elsevier.
    """
    L_vals = np.linspace(min_L, max_L, L_grid_size)
    delta3 = np.zeros(L_vals.shape)
    pbar_widgets = [
        f"{Fore.GREEN}Computing spectral rigidity: {Fore.RESET}",
        f"{Fore.BLUE}",
        Percentage(),
        f" {Fore.RESET}",
        " ",
        Timer(),
        f"|{Fore.YELLOW}",
        AdaptiveETA(),
        f"{Fore.RESET}",
    ]
    pbar = ProgressBar(widgets=pbar_widgets, maxval=L_vals.shape[0]).start()
    for i, L in enumerate(L_vals):
        delta3_L_vals = np.empty((c_iters))
        spectralIter(eigs, unfolded, delta3_L_vals, L, c_iters, 100)
        if len(delta3_L_vals) != c_iters:
            raise Exception("We aren't computing enough L values")
        delta3[i] = np.mean(delta3_L_vals)
        pbar.update(i)
    pbar.finish()
    return L_vals, delta3


@jit(nopython=True, fastmath=True, cache=True)
def spectralIter(
    eigs: ndarray,
    unfolded: ndarray,
    delta3_L_vals: ndarray,
    L: float,
    c_iters: int = 100,
    interval_gridsize: int = 100,
) -> ndarray:
    c_starts = np.random.uniform(eigs[0], eigs[-1], c_iters)
    for i in prange(len(c_starts)):
        # c_start is in space of eigs, not unfolded
        grid = np.linspace(c_starts[i] - L / 2, c_starts[i] + L / 2, interval_gridsize)
        step_vals = stepFunctionVectorized(eigs, grid)
        K = slope(grid, step_vals)
        w = intercept(grid, step_vals, K)
        y_vals = sq_lin_deviation_all(eigs, K, w, grid)
        delta3 = integrateFast(grid, y_vals)
        delta3_L_vals[i] = delta3 / L
    return delta3_L_vals


@jit(nopython=True, fastmath=True, cache=True)
def slope(x: ndarray, y: ndarray) -> np.float64:
    """Perform linear regression to compute the slope."""
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_dev = x - x_mean
    y_dev = y - y_mean
    cov = np.sum(x_dev * y_dev)
    var = np.sum(x_dev * x_dev)
    if var == 0:
        return 0
    return cov / var


@jit(nopython=True, fastmath=True, cache=True)
def intercept(x: ndarray, y: ndarray, slope: np.float64) -> np.float64:
    return np.mean(y) - slope * np.mean(x)


@jit(nopython=True, fastmath=True, cache=True)
def integrateFast(grid: ndarray, values: ndarray) -> np.float64:
    """scipy.integrate.trapz is excruciatingly slow and unusable for our purposes.
    This tiny rewrite seems to result in a near 20x speedup."""
    integral = 0
    for i in range(len(grid) - 1):
        w = grid[i + 1] - grid[i]
        h = values[i] + values[i + 1]
        integral += w * h / 2
    return integral


@jit(nopython=True, fastmath=True, cache=True)
def sq_lin_deviation(eigs: ndarray, K: float, w: float, l: float) -> np.float64:
    n = stepFunctionG(eigs, l)
    deviation = n - K * l - w
    return deviation * deviation


@jit(nopython=True, fastmath=True, cache=True)
def sq_lin_deviation_all(eigs: ndarray, K: float, w: float, x: ndarray) -> ndarray:
    ret = np.empty(len(x))
    for i in prange(len(x)):
        ret[i] = sq_lin_deviation(eigs, K, w, x[i])
    return ret
