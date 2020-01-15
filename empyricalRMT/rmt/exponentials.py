import matplotlib.pyplot as plt
import numpy as np
import seaborn as sbn

from numba import jit
from scipy.optimize import curve_fit

from empyricalRMT.rmt.eigenvalues import stepFunctionVectorized


@jit(nopython=True, fastmath=True)
def slope(arr: np.array) -> np.float64:
    x = np.arange(0, len(arr))
    return fullSlope(x, arr)


@jit(nopython=True, fastmath=True)
def derivative(x: np.array, y=np.array) -> np.array:
    res = np.empty(x.shape, dtype=np.float64)
    # for i = 1 (i.e. y[1], we compute (y[0] - y[2]) / 2*spacing)
    # ...
    # for i = L - 2, we compute (y[L-3] - y[L-1]) / 2*spacing
    # i.e. (y[0:L-2] - y[2:]) / 2*spacing
    L = len(x)
    res[1:-1] = (y[2:] - y[0 : L - 2]) / (x[2:] - x[0 : L - 2])
    res[0] = (y[1] - y[0]) / (x[1] - x[0])
    res[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
    return res


@jit(nopython=True, fastmath=True)
def inverse_gompertz(x, a, b, c):
    """a - asymptote, should be close to max(eigs)
    b - center, must be greater than zero
    """
    # return np.log(b / np.log(1 / t)) / c
    # return np.log((np.log(x/ a) - np.log(a)) / -b2) / np.log(b3)
    return np.log(np.log(b) - np.log(np.log(-x / a))) / c


@jit(nopython=True, fastmath=True)
def gompertz(x, a, b, c):
    return a * np.exp(-b * np.exp(-c * x))


@jit(nopython=True, fastmath=True)
def exponential(x, a, b, c, d):
    """b is intercept,
    """
    return b - a * np.exp(-c * x ** (1 / d))


A = np.random.standard_normal(1000 * 1000).reshape([1000, 1000])
A = (1 / np.sqrt(2)) * (A + A.T)
eigs = np.linalg.eigvalsh(A)
steps = stepFunctionVectorized(eigs, eigs)


sbn.set_style("darkgrid")
plt.plot(eigs, steps, color="black", label="step")
[a, b, c], cov = curve_fit(gompertz, eigs, steps, (np.max(eigs), 1, 1))
plt.plot(eigs, gompertz(eigs, a, b, c), color="orange", label="gompertz")
[a, b, c, d], cov2 = curve_fit(
    exponential, eigs, steps, (np.min(steps), np.max(steps), 1000, 10)
)
plt.plot(eigs, exponential(eigs, a, b, c, d), color="red", label="exponential")
plt.legend()
plt.show()

# x = np.linspace(-10, 100, 100000)
# print(inverse_gompertz(x, 1, 1, 1))
# plt.plot(x, inverse_gompertz(x, 1, 2, 2))
# plt.show()

