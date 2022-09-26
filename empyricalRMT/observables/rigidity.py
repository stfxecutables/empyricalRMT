from typing import Tuple

import numpy as np
from numba import jit, prange
from numpy import float64 as f64
from numpy import ndarray
from typing_extensions import Literal

from empyricalRMT._constants import (
    AUTO_MAX_ITERS,
    CONVERG_PROG,
    CONVERG_PROG_INTERVAL,
    ITER_COUNT,
    MIN_ITERS,
    PERCENT,
    RIGIDITY_GRID,
    RIGIDITY_PROG,
)
from empyricalRMT._types import bArr, fArr, iArr
from empyricalRMT.observables.step import _step_function_fast
from empyricalRMT.utils import ConvergenceError, kahan_add

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
def spectral_rigidity(
    unfolded: fArr,
    L: fArr = np.arange(2, 20, 0.5),
    tol: float = 0.01,
    max_iters: int = 0,
    gridsize: int = RIGIDITY_GRID,
    integration: Literal["simps", "trapz"] = "simps",
    show_progress: bool = True,
) -> Tuple[fArr, fArr, bArr, iArr]:
    """Compute the spectral rigidity for a particular unfolding.

    Computes the spectral rigidity (delta_3, ∆₃ [1]) for a
    particular set of eigenvalues and their unfoldings via random samling.
    The internal integral of the staircase deviation from a linear fit is
    computed via Simpson's method, and samples of c are drawn iteratively
    for each L value, until a convergence criterion is met.

    Parameters
    ----------
    unfolded: ndarray
        The unfolded eigenvalues.

    L: ndarray
        The values of L at which to compute the rigidity.

    max_iters: int = -1
        How many times the location of the center, c, of the interval
        [c - L/2, c + L/2] should be chosen uniformly at random for
        each L in order to compute the estimate of the spectral
        rigidity.

        For an NxN GOE matrix where N in [5000, 10 000, 20 000], unfolded with a
        high-degree polynomial or analytically unfolded, the default values of
        `tol=0.01` and `max_iters=int(1e4)`, the algorithm should converge for L
        values up to about 70-90. Larger L-values than this will require
        increasing `max_iters` to about `1e5` for consistent convergence.
        Smaller matrices (e.g. N=2000) struggle to converge past L values of
        60 for the defaults.

    tol: float = 0.01
        Convergence criterion. Convergence is reached when the range of
        the last 1000 computed values is less than `tol`.


    gridsize: int = 100
        Each internal integral is computed over a grid
        of `gridsize` points on [c - L/2, c + L/2]. Smaller values here
        increase the variance of sampled values of delta_3, and increase the
        likelihood that `max_iters` is reached. Probably best to leave at the
        default value.

    integration: "simps" | "trapz"
        Whether to use the trapezoidal or simpson's rule for integration. Default
        `simps`. Method "trapz" might be faster in some cases, at the cost of some
        accuracy.


    Returns
    -------
    L : ndarray
        The L values generated based on the values of L_grid_size,
        min_L, and max_L.

    delta3 : ndarray
        The computed spectral rigidity values for each of L.

    Notes
    -----
    This algorithm is fairly heavily optimized (for Python), and executes fast
    for even high grid densities and a large number of iterations per grid
    value. Efficiency seems to be roughly O(L_grid_size). Increasing
    len(unfolded) and increasing c_iters also increases the execution time, but
    not nearly as much as increasing the grid density. Even for a large number
    of iterations per L value (e.g. 100000) the algorithm is still quite quick,
    so large values are generally recommended here. This is especially the case
    if looking at large L values (e.g. probably for L > 20, definitely for L >
    50).

    In general, the extent to which observed spectral rigidity values will match
    those predicted by theory will depend heavily on the choice of unfolding
    function and the matrix size. In addition, convergence to the expected
    curves can be quite slow, especially for large L values. E.g. the
    eigenvalues of an n == 1000 GOE matrix, with polynomial unfolding, will
    generally only match expected values up to about L == 20. For an n == 5000
    GOE matrix, rigidity values will start to deviate from theory by about L ==
    40 or L == 50. For n == 10000, you will probably start seeing noticable
    deviation by about L == 60 or L == 70.

    References
    ----------
    .. [1] Mehta, M. L. (2004). Random matrices (Vol. 142). Elsevier
    """
    # delta3 = _spectral_iter_grid(
    #     unfolded=unfolded,
    #     L_vals=L.copy().ravel(),
    #     gridsize=gridsize,
    #     use_simpson=True,
    # )
    if max_iters <= 0:
        success, max_iters = delta_L(
            unfolded=unfolded,
            L=float(np.max(L)),
            gridsize=RIGIDITY_GRID,
            max_iters=AUTO_MAX_ITERS,
            min_iters=10 * MIN_ITERS,  # also safety
            tol=tol,
            use_simpson=integration != "trapz",
            show_progress=show_progress,
        )[1:]
        max_iters = int(max_iters * 2)  # precaution
        if not success:
            raise ConvergenceError(
                f"For the largest L-value in your provided Ls, {np.max(L)}, the "
                f"spectral rigidity at L did not converge in {AUTO_MAX_ITERS} "
                "iterations. Either reduce the range of L values, reduce the "
                "`tol` tolerance value, or manually set `max_iters` to be some "
                "value other than the default of 0 to disable this check. Note "
                "the convergence criterion involves the range on the last 1000 "
                "values, which are themselves iteratively-computed means, so is "
                "a somewhat strict convergence criterion. However, setting "
                "`max_iters` too low then provides NO guarantee on the error for "
                "non-converging L values."
            )

    delta3, converged, iters = delta_parallel(
        unfolded=unfolded,
        L_vals=L.copy().ravel(),
        gridsize=gridsize,
        max_iters=max_iters,
        min_iters=MIN_ITERS,
        tol=tol,
        use_simpson=integration != "trapz",
        show_progress=show_progress,
    )
    return L, delta3, converged, iters


@jit(nopython=True, cache=False, parallel=True, fastmath=True)
def delta_parallel(
    unfolded: fArr,
    L_vals: fArr,
    tol: float = 0.01,
    max_iters: int = int(1e6),
    gridsize: int = 1000,
    min_iters: int = MIN_ITERS,
    use_simpson: bool = True,
    show_progress: bool = True,
) -> Tuple[fArr, bArr, iArr]:
    prog_interval = len(L_vals) // 50
    if prog_interval == 0:
        prog_interval = 1
    delta3 = np.zeros(L_vals.shape)
    iters = np.zeros(L_vals.shape, dtype=np.int64)
    converged = np.zeros_like(L_vals, dtype=np.bool_)
    if show_progress:
        print(RIGIDITY_PROG, 0, PERCENT)
    for i in prange(len(L_vals)):
        L = L_vals[i]
        delta3[i], converged[i], iters[i] = delta_L(
            unfolded=unfolded,
            L=L,
            gridsize=gridsize,
            max_iters=max_iters,
            min_iters=min_iters,
            tol=tol,
            use_simpson=use_simpson,
        )

        if show_progress and i % prog_interval == 0:
            prog = int(100 * np.sum(delta3 != 0) / len(delta3))
            print(RIGIDITY_PROG, prog, PERCENT)
    return delta3, converged, iters


# See https://stackoverflow.com/a/54078906 for why Kahan summation *might* be worth
# it here. However, keep in mind our convergence criterion sort-of implies this is
# not an issue.
@jit(nopython=True, cache=False, fastmath=True)
def delta_L(
    unfolded: ndarray,
    L: float,
    gridsize: int = 100,
    max_iters: int = int(1e6),
    min_iters: int = 100,
    tol: float = 0.01,
    use_simpson: bool = True,
    show_progress: bool = False,
) -> Tuple[float, bool, int]:
    buf = 1000
    prog_interval = CONVERG_PROG_INTERVAL
    if L > 100:
        prog_interval *= 10

    delta_running = np.zeros((buf,))
    k = np.uint64(0)
    c = np.float64(0.0)  # compensation (carry-over) term for Kahan summation
    d3_mean: f64 = np.float64(0.0)
    if show_progress:
        print(CONVERG_PROG, 0, ITER_COUNT)
    while True:
        if k != 0:  # awkward, want uint64 k
            k += 1
        start = np.random.uniform(unfolded[0], unfolded[-1])
        grid = np.linspace(start - L / 2, start + L / 2, gridsize)
        steps = _step_function_fast(unfolded, grid)  # performance bottleneck
        K = _slope(grid, steps)
        w = _intercept(grid, steps, K)
        y_vals = _sq_lin_deviation(unfolded, steps, K, w, grid)
        if use_simpson:
            delta3_c = _int_simps_nonunif(grid, y_vals)  # O(len(grid))
        else:
            delta3_c = _integrate_fast(grid, y_vals)  # O(len(grid))
        d3 = delta3_c / L
        if k == 0:  # initial value
            d3_mean = d3
            delta_running[0] = d3_mean
            k += 1
            continue
        else:
            # Regular sum
            # d3_mean = (k * d3_mean + d3) / (k + 1)
            # d3_mean += (d3 - d3_mean) / k

            # Kahan sum - but can we be sure Numba isn't optimizing away?
            update = (d3 - d3_mean) / k  # mean + update is new mean
            d3_mean, c = kahan_add(current_sum=d3_mean, update=update, carry_over=c)
            # remainder = update - c
            # lossy = d3_mean + remainder
            # c = (lossy - d3_mean) - remainder
            # d3_mean = lossy
            delta_running[int(k) % buf] = d3_mean

        if show_progress and k % prog_interval == 0:
            print(CONVERG_PROG, int(k * 2), ITER_COUNT)  # x2 for safety factor
        if k >= max_iters:
            break
        if (
            (k > min_iters)
            and (k % buf == 0)  # all buffer values must have changed
            and (np.abs(np.max(delta_running) - np.min(delta_running)) < tol)
        ):
            break

    if show_progress:
        print(CONVERG_PROG, int(k), ITER_COUNT)
    converged = np.abs(np.max(delta_running) - np.min(delta_running)) < tol
    # I don't think it matters much if we use median, mean, max, or min for the
    # final returned single value, given our convergence criterion is so strict
    # return d3_mean, converged, k
    return np.mean(delta_running), converged, k  # type: ignore


@jit(nopython=True, cache=False, fastmath=True)
def _delta_grid(
    unfolded: ndarray, starts: ndarray, L: float, gridsize: int, use_simpson: bool
) -> f64:
    delta3s = np.empty_like(starts)
    for i, start in enumerate(starts):
        grid = np.linspace(start - L / 2, start + L / 2, gridsize)
        steps = _step_function_fast(unfolded, grid)  # performance bottleneck
        K = _slope(grid, steps)
        w = _intercept(grid, steps, K)
        y_vals = _sq_lin_deviation(unfolded, steps, K, w, grid)
        if use_simpson:
            delta3 = _int_simps_nonunif(grid, y_vals)  # O(len(grid))
        else:
            delta3 = _integrate_fast(grid, y_vals)  # O(len(grid))
        delta3s[i] = delta3 / L
    return np.mean(delta3s)  # type: ignore


@jit(nopython=True, cache=False, fastmath=True)
def _slope(x: ndarray, y: ndarray) -> f64:
    """Perform linear regression to compute the slope."""
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_dev = x - x_mean
    y_dev = y - y_mean
    cov = np.sum(x_dev * y_dev)
    var = np.sum(x_dev * x_dev)
    if var == 0.0:
        return 0.0  # type: ignore
    return cov / var  # type: ignore


@jit(nopython=True, cache=False, fastmath=True)
def _intercept(x: ndarray, y: ndarray, slope: f64) -> f64:
    return np.mean(y) - slope * np.mean(x)  # type: ignore


@jit(nopython=True, cache=False, fastmath=True)
def _integrate_fast(grid: ndarray, values: ndarray) -> f64:
    """scipy.integrate.trapz is excruciatingly slow and unusable for our purposes.
    This tiny rewrite seems to result in a near 20x speedup. However, being trapezoidal
    integration, it is quite inaccurate."""
    integral = 0
    for i in range(len(grid) - 1):
        w = grid[i + 1] - grid[i]
        h = values[i] + values[i + 1]
        integral += w * h / 2
    return integral  # type: ignore


# NOTE: !!!! Very important *NOT* to use parallel=True here, since we parallelize
# the outer loops. Adding it inside *dramatically* slows performance.
@jit(nopython=True, cache=False, fastmath=True)
def _sq_lin_deviation(eigs: ndarray, steps: ndarray, K: f64, w: f64, grid: fArr) -> fArr:
    """Compute the sqaured deviation of the staircase function of the best fitting
    line, over the region in `grid`.

    Parameters
    ----------
    eigs: ndarray
        The raw, sorted eigenvalues

    steps: ndarray
        The step function values which were computed on `grid`.

    K: float
        The calculated slope of the line of best fit.

    w: float
        The calculated intercept.

    grid: ndarray
        The grid of values for which the step function was evaluated.

    Returns
    -------
    sq_deviations: ndarray
        The squared deviations.
    """
    ret = np.empty(len(grid))
    for i in range(len(grid)):
        n = steps[i]
        deviation = n - K * grid[i] - w
        ret[i] = deviation * deviation
    return ret


# fmt: off
@jit(nopython=True, cache=False, fastmath=True)
def _int_simps_nonunif(grid: fArr, vals: fArr) -> f64:
    """
    Simpson rule for irregularly spaced data. Copied shamelessly from
    https://en.wikipedia.org/w/index.php?title=Simpson%27s_rule&oldid=938527913#Composite_Simpson's_rule_for_irregularly_spaced_data
    for compilation here with Numba, and to overcome the extremely slow performance
    problems with scipy.integrate.simps.

    Parameters
    ----------
    grid: list or np.array of floats
            Sampling points for the function values

    vals: list or np.array of floats
            Function values at the sampling points

    Returns
    -------
    float: approximation for the integral
    """
    N = len(grid) - 1
    h = np.diff(grid)

    result = f64(0.0)
    for i in range(1, N, 2):
        hph = h[i] + h[i - 1]
        result += vals[i] * (h[i]**3 + h[i-1]**3 + 3.0*h[i]*h[i-1]*hph) / (6*h[i]*h[i-1])
        result += vals[i-1] * (2.0*h[i-1]**3 - h[i]**3 + 3.0*h[i]*h[i-1]**2) / (6*h[i-1] * hph)
        result += vals[i+1] * (2.0*h[i]**3 - h[i-1]**3 + 3.0*h[i-1]*h[i]**2) / (6*h[i] * hph)

    if (N + 1) % 2 == 0:
        result += vals[N] * (2*h[N-1]**2 + 3.0*h[N-2]*h[N-1]) / (6*(h[N-2] + h[N-1]))
        result += vals[N-1] * (h[N-1]**2 + 3*h[N-1]*h[N-2]) / (6*h[N-2])
        result -= vals[N-2] * h[N-1]**3 / (6*h[N-2]*(h[N-2] + h[N-1]))
    return result
# fmt: on
