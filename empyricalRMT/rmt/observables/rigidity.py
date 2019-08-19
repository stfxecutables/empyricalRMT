import numpy as np

from colorama import Fore
from numba import jit, prange
from progressbar import AdaptiveETA, Percentage, ProgressBar, Timer

from ...rmt.eigenvalues import stepFunctionG, stepFunctionVectorized


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
def spectralRigidityRewrite(
    eigs: np.array,
    unfolded: np.array,
    c_iters: int = 1000,
    L_grid_size: int = 50,
    min_L=2,
    max_L=25,
) -> [np.array, np.array]:
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
        spectralIterRewrite(eigs, unfolded, delta3_L_vals, L, c_iters, 100)
        if len(delta3_L_vals) != c_iters:
            raise Exception("We aren't computing enough L values")
        delta3[i] = np.mean(delta3_L_vals)
        pbar.update(i)
    pbar.finish()
    return L_vals, delta3


@jit(nopython=True, fastmath=True, cache=True)
def slope(x: np.array, y: np.array) -> np.float64:
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
def intercept(x: np.array, y: np.array, slope: np.float64) -> np.float64:
    return np.mean(y) - slope * np.mean(x)


@jit(nopython=True, fastmath=True, cache=True)
def integrateFast(grid: np.array, values: np.array) -> np.float64:
    """https://en.wikipedia.org/wiki/Trapezoidal_rule#Uniform_grid"""
    delta_x = (grid[-1] - grid[0]) / len(grid)
    scale = delta_x / 2
    integral = scale * (values[0] + values[-1] + 2 * np.sum(values[1:-1]))
    return integral


@jit(nopython=True, fastmath=True, cache=True)
def spectralIterRewrite(
    eigs: np.array,
    unfolded: np.array,
    delta3_L_vals: np.array,
    L: float,
    c_iters: int = 100,
    interval_gridsize: int = 100,
) -> float:
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
def sq_lin_deviation(eigs, K, w, l) -> np.float64:
    n = stepFunctionG(eigs, l)
    deviation = n - K * l - w
    return deviation * deviation


@jit(nopython=True, fastmath=True, cache=True)
def sq_lin_deviation_all(eigs, K, w, x: np.array) -> np.array:
    ret = np.empty(len(x))
    for i in prange(len(x)):
        ret[i] = sq_lin_deviation(eigs, K, w, x[i])
    return ret
