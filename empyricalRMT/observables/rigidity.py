import numpy as np
from numpy import ndarray

from colorama import Fore
from numba import jit, prange
from progressbar import AdaptiveETA, Percentage, ProgressBar, Timer
from typing import Tuple
from typing_extensions import Literal

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
    unfolded: ndarray,
    L: ndarray = np.arange(2, 50, 10000),
    c_iters: int = 10000,
    integration: Literal["simps", "trapz"] = "simps",
    show_progress: bool = True,
) -> Tuple[ndarray, ndarray]:
    """Compute the spectral rigidity for a particular unfolding.

    Computes the spectral rigidity (delta_3, ∆₃ [1]_) for a
    particular set of eigenvalues and their unfolding.

    Parameters
    ----------
    unfolded: ndarray
        The unfolded eigenvalues.

    L: ndarray
        The values of L to compute the rigidity.

    c_iters: int
        How many times the location of the center, c, of the interval
        [c - L/2, c + L/2] should be chosen uniformly at random for
        each L in order to compute the estimate of the spectral
        rigidity. Default 10000.

    integration: "simps" | "trapz"
        Whether to use the trapezoidal or simpson's rule for integration. Default
        `simps`. Method "trapz" might be faster in some cases, but final
        calculations will be considerably more inaccurate.


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
    L_vals = np.copy(L)
    L_grid_size = len(L_vals)
    delta3 = np.zeros(L_vals.shape)
    if show_progress:
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
        if integration == "trapz":
            _spectral_iter(
                unfolded, delta3_L_vals, L, c_iters, L_grid_size, use_simpsons=False
            )
        else:
            _spectral_iter(unfolded, delta3_L_vals, L, c_iters, L_grid_size)
        if len(delta3_L_vals) != c_iters:
            raise Exception("We aren't computing enough L values")
        delta3[i] = np.mean(delta3_L_vals)
        if show_progress:
            pbar.update(i)
    if show_progress:
        pbar.finish()
    return L_vals, delta3


@jit(nopython=True, fastmath=True, cache=True, parallel=True)
def _spectral_iter(
    unfolded: ndarray,
    delta3_L_vals: ndarray,
    L: float,
    c_iters: int = 10000,
    interval_gridsize: int = 10000,  # does not tend to effect performance significantly
    use_simpsons: bool = True,
) -> ndarray:
    """Compute c_iters values of L and save them in delta3_L_vals.

    Parameters
    ----------
    unfolded: ndarray
        The sorted unfolded eigenvalues.

    delta3_L_vals: ndarray
        The array to store all the c_iters computed L values.

    L: float
        The current L value for which the spectral rigidity is being calculated.

    c_iters: int
        The number of centre-points (c) to choose (i.e. the number of L values to
        compute).

    interval_gridsize: int
        The number of points for which to evaluate the deviation from a straight
        line on [c - L/2, c + L/2].
    """
    # make each iteration centred at a random value on the unfolded spectrum
    starts = np.random.uniform(unfolded[0], unfolded[-1], c_iters)
    for i in prange(len(starts)):
        # c_start is in space of unfolded, not unfolded
        grid = np.linspace(starts[i] - L / 2, starts[i] + L / 2, interval_gridsize)
        steps = _step_function_fast(unfolded, grid)  # performance bottleneck
        K = _slope(grid, steps)
        w = _intercept(grid, steps, K)
        y_vals = _sq_lin_deviation(unfolded, steps, K, w, grid)
        if use_simpsons:
            delta3 = _int_simps_nonunif(grid, y_vals)  # O(len(grid))
        else:
            delta3 = _integrate_fast(grid, y_vals)  # O(len(grid))
        delta3_L_vals[i] = delta3 / L


@jit(nopython=True, fastmath=True, cache=True)
def _slope(x: ndarray, y: ndarray) -> np.float64:
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
def _intercept(x: ndarray, y: ndarray, slope: np.float64) -> np.float64:
    return np.mean(y) - slope * np.mean(x)


@jit(nopython=True, fastmath=True, cache=True)
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
@jit(nopython=True, fastmath=True, cache=True)
def _sq_lin_deviation(
    eigs: ndarray, steps: ndarray, K: float, w: float, grid: ndarray
) -> ndarray:
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
    ret = np.empty((len(grid)), dtype=np.float64)
    for i in prange(len(grid)):
        n = steps[i]
        deviation = n - K * grid[i] - w
        ret[i] = deviation * deviation
    return ret


# fmt: off
@jit(nopython=True, fastmath=True, cache=True)
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
