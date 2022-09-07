from typing import Tuple

import numpy as np
from numba import jit, prange
from numpy import float64 as f64
from numpy import int64 as i64
from numpy import ndarray
from numpy.typing import NDArray
from typing_extensions import Literal

from empyricalRMT._constants import PERCENT, RIGIDITY_PROG
from empyricalRMT.observables.step import _step_function_fast

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
    unfolded: NDArray[f64],
    L: NDArray[f64] = np.arange(2, 50, 10000),
    max_iters: int = int(1e4),
    tol: float = 0.01,
    gridsize: int = 100,
    min_iters: int = 100,
    integration: Literal["simps", "trapz"] = "simps",
    show_progress: bool = True,
) -> Tuple[NDArray[f64], NDArray[f64], NDArray[np.bool_], NDArray[i64]]:
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

    max_iters: int = int(1e5)
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
    delta3, converged, iters = _spectral_iter_converge(
        unfolded=unfolded,
        L_vals=L.copy().ravel(),
        gridsize=gridsize,
        max_iters=max_iters,
        min_iters=min_iters,
        tol=tol,
        use_simpson=integration != "trapz",
        show_progress=show_progress,
    )
    return L, delta3, converged, iters


@jit(nopython=True, cache=False, parallel=True, fastmath=True)
def _spectral_iter_converge(
    unfolded: ndarray,
    L_vals: ndarray,
    gridsize: int = 1000,
    max_iters: int = int(1e6),
    min_iters: int = 1000,
    tol: float = 0.01,
    use_simpson: bool = True,
    show_progress: bool = True,
) -> Tuple[NDArray[f64], NDArray[np.bool_], NDArray[i64]]:
    # buf = 10000
    prog_interval = len(L_vals) // 50
    if prog_interval == 0:
        prog_interval = 1
    delta3 = np.zeros(L_vals.shape, dtype=unfolded.dtype)
    iters = np.zeros(L_vals.shape, dtype=np.uint64)
    # delta_running = np.zeros((len(L_vals), buf))
    # ks = np.zeros_like(L_vals, dtype=np.int64) - 1
    converged = np.zeros_like(L_vals, dtype=np.bool_)
    # starts = np.random.uniform(unfolded[0], unfolded[-1], (len(L_vals), max_iters))
    if show_progress:
        print(RIGIDITY_PROG, 0, PERCENT)
    for i in prange(len(L_vals)):
        # delta3_cs = np.empty_like(starts[i])
        # d3_mean = 0.0
        L = L_vals[i]
        delta3[i], converged[i], iters[i] = _spectral_converge_L(
            unfolded=unfolded,
            L=L,
            gridsize=gridsize,
            max_iters=max_iters,
            min_iters=min_iters,
            tol=tol,
            use_simpson=use_simpson,
        )
        # # delta_running = np.zeros((buf,))
        # while True:
        #     ks[i] += 1
        #     start = np.random.uniform(unfolded[0], unfolded[-1])
        #     grid = np.linspace(start - L / 2, start + L / 2, gridsize)
        #     steps = _step_function_fast(unfolded, grid)  # performance bottleneck
        #     K = _slope(grid, steps)
        #     w = _intercept(grid, steps, K)
        #     y_vals = _sq_lin_deviation(unfolded, steps, K, w, grid)
        #     if use_simpson:
        #         delta3_c = _int_simps_nonunif(grid, y_vals)  # O(len(grid))
        #     else:
        #         delta3_c = _integrate_fast(grid, y_vals)  # O(len(grid))
        #     d3 = delta3_c / L
        #     # delta3_cs[k] = d3
        #     # we use the fact that for x = [x_0, x_1, ... x_n-1], the
        #     # average a_k == (k*a_(k-1) + x_k) / (k+1) for k = 0, ..., n-1
        #     if ks[i] == 0:  # initial value
        #         d3_mean = d3
        #         delta_running[i][ks[i] % buf] = d3_mean
        #         break
        #     else:
        #         d3_mean = (ks[i] * d3_mean + d3) / (ks[i] + 1)
        #         delta_running[i][ks[i] % buf] = d3_mean

        #     if (
        #         (ks[i] > min_iters)
        #         and (ks[i] % 500 == 0)
        #         and (np.abs(np.max(delta_running) - np.min(delta_running)) < tol)
        #     ):
        #         break
        #     if ks[i] >= max_iters:
        #         break

        # delta3[i] = np.mean(delta3_cs)
        # delta3[i] = d3_mean
        # delta3[i] = np.mean(delta_running[i])
        # converged[i] = np.abs(np.max(delta_running[i]) - np.min(delta_running[i])) < tol

        if show_progress and i % prog_interval == 0:
            prog = int(100 * np.sum(delta3 != 0) / len(delta3))
            print(RIGIDITY_PROG, prog, PERCENT)
    return delta3, converged, iters


"""
https://stackoverflow.com/a/54078906

I've use Kahan summation for Monte-Carlo integration. You have a scalar valued
function f which you believe is rather expensive to evaluate; a reasonable
estimate is 65ns/dimension. Then you accumulate those values into an
average-updating an average takes about 4ns. So if you update the average using
Kahan summation (4x as many flops, ~16ns) then you're really not adding that
much compute to the total. Now, often it is said that the error of Monte-Carlo
integration is sigma/sqrt(N), but this is incorrect. The real error bound (in finite
precision arithmetic) is

sigma/sqrt(N) + cond(I_n) * eps * N

Where cond(In) is the condition number of summation and ε is twice the unit
roundoff. So the algorithm diverges faster than it converges. For 32 bit
arithmetic, getting eps N ~ 1 is simple: 10^7 evaluations can be done exceedingly
quickly, and after this your Monte-Carlo integration goes on a random walk. The
situation is even worse when the condition number is large.

If you use Kahan summation, the expression for the error changes to

sigma/sqrt(N) + cond(I_n) * eps^2 * N,

Which, admittedly still diverges faster than it converges, but eps^2 * N cannot
be made large on a reasonable timescale on modern hardware.

"""


@jit(nopython=True, cache=False, fastmath=True)
def _spectral_converge_L(
    unfolded: ndarray,
    L: float,
    gridsize: int = 1000,
    max_iters: int = int(1e6),
    min_iters: int = 1000,
    tol: float = 0.01,
    use_simpson: bool = True,
) -> Tuple[float, bool, int]:
    buf = 1000

    delta_running = np.zeros((buf,))
    k = np.uint64(0)
    d3_mean = 0.0
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
        # we use the fact that for x = [x_0, x_1, ... x_n-1], the
        # average a_k == (k*a_(k-1) + x_k) / (k+1) for k = 0, ..., n-1
        if k == 0:  # initial value
            d3_mean = d3
            delta_running[0] = d3_mean
            k += 1
            continue
        else:
            # d3_mean = (k * d3_mean + d3) / (k + 1)
            d3_mean += (d3 - d3_mean) / k
            delta_running[int(k) % buf] = d3_mean

        if k >= max_iters:
            break
        if (
            (k > min_iters)
            and (k % buf == 0)  # all buffer values must have changed
            and (np.abs(np.max(delta_running) - np.min(delta_running)) < tol)
        ):
            break

    converged = np.abs(np.max(delta_running) - np.min(delta_running)) < tol
    # return d3_mean, converged, k
    return np.mean(delta_running), converged, k


@jit(nopython=True, cache=True, fastmath=True)
def compute_delta(
    unfolded: ndarray, starts: ndarray, L: float, gridsize: int, use_simpson: bool
) -> float:
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
    return np.mean(delta3s)


@jit(nopython=True, cache=True, fastmath=True)
def _slope(x: ndarray, y: ndarray) -> np.float64:
    """Perform linear regression to compute the slope."""
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_dev = x - x_mean
    y_dev = y - y_mean
    cov = np.sum(x_dev * y_dev)
    var = np.sum(x_dev * x_dev)
    if var == 0.0:
        return 0.0
    return cov / var


@jit(nopython=True, cache=True, fastmath=True)
def _intercept(x: ndarray, y: ndarray, slope: np.float64) -> np.float64:
    return np.mean(y) - slope * np.mean(x)


@jit(nopython=True, cache=True, fastmath=True)
def _integrate_fast(grid: ndarray, values: ndarray) -> np.float64:
    """scipy.integrate.trapz is excruciatingly slow and unusable for our purposes.
    This tiny rewrite seems to result in a near 20x speedup. However, being trapezoidal
    integration, it is quite inaccurate."""
    integral = 0
    for i in range(len(grid) - 1):
        w = grid[i + 1] - grid[i]
        h = values[i] + values[i + 1]
        integral += w * h / 2
    return integral


# NOTE: !!!! Very important *NOT* to use parallel=True here, since we parallelize
# the outer loops. Adding it inside *dramatically* slows performance.
@jit(nopython=True, cache=True, fastmath=True)
def _sq_lin_deviation(eigs: ndarray, steps: ndarray, K: float, w: float, grid: ndarray) -> ndarray:
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
@jit(nopython=True, cache=True, fastmath=True)
def _int_simps_nonunif(grid: np.array, vals: np.array) -> float:
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

    result = 0.0
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
