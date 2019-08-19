import numpy as np

from numba import jit, prange

from ..rmt.observables.spacings import computeSpacings
from ..utils import eprint, nd_find


def getEigs(matrix):
    eprint("Calculating eigenvalues...")
    eigs = np.linalg.eigvalsh(matrix)
    eprint("Calculated eigenvalues")
    return eigs


# this function is equivalent to `return len(eigs[eigs <= x])`
# in particular, since our eigenvalues are always sorted, we can simply
# return the index of the eigenvalue in eigs, e.g. if we have eigs
#   [0.1, 0.2, 0.3, 0.4]
# then stepFunctionG(eigs, 0.20) is
# this function could be improved with binary search and memoization if
# necessary or if it becomes a bottleneck
@jit(nopython=True, fastmath=True, cache=True)
def stepFunctionG(eigs: np.array, x: float):
    cumulative = 0
    for eig in eigs:
        if x <= eig:
            break
        else:
            cumulative += 1
    return float(cumulative)


@jit(nopython=True, fastmath=True, cache=True)
def stepFunctionVectorized(eigs: np.array, x: np.array):
    ret = np.empty(len(x))
    for i in prange(len(x)):
        ret[i] = stepFunctionG(eigs, x[i])
    return ret


class StepFunction:
    def __init__(self, eigs: np.array, grid_density: int = 100, precompute=False):
        """eigs: trimmed, sorted eigenvalues
        density: number of values between each eigenvalue to precompute for the step function
        """
        self.eigs = eigs
        self.__computed = []
        self.memo = {}
        if precompute:
            print("Computing memoized step function values")
            for i in range(len(eigs) - 1):
                grid_i = np.linspace(eigs[i], eigs[i + 1], grid_density)
                for val in grid_i:
                    self.memo[val] = stepFunctionG(eigs, val)
                    self.computed.append(val)
            print("Done memoization.")
        self.__computed = np.array(self.__computed)

    def memoized(self):
        """Usage:
        >>> stepFunction = StepFunction(eigs).fast()
        >>> stepFunction(20)  # 1.2e-6
        """

        def fastStep(x: float):
            if x not in self.memo:
                self.memo[x] = stepFunctionG(self.eigs, x)
            return self.memo[x]

        return fastStep

    def raw(self, x):
        return stepFunctionG(self.eigs, x)

    @property
    def computed(self):
        return self.__computed


def trim_largest(eigs: np.array, percentile: int = 98) -> np.array:
    abs_eigs = np.absolute(eigs)
    largest = np.percentile(abs_eigs, percentile)
    trimmed = eigs[abs_eigs < largest]
    trimmed.sort()
    return trimmed


def trim_iteratively(eigs: np.array) -> np.array:
    abs_eigs = np.absolute(eigs)

    percentile = 98
    trimmed = eigs
    while percentile > 94.5:
        largest = np.percentile(abs_eigs, percentile)
        trimmed = eigs[abs_eigs < largest]
        median = np.median(trimmed)
        mean = np.mean(trimmed)
        if np.absolute(mean - median) > 10 * median:
            percentile -= 0.5
        else:
            break
    trimmed.sort()

    lower = nd_find(trimmed, np.percentile(trimmed, 1))
    trimmed = trimmed[lower:]

    # median = np.median(trimmed)
    # min_index = 0
    # for i in range(len(trimmed)):
    #     if trimmed[i] > np.percentile(eigs, 5):
    #         min_index = i
    #         break
    #     if (trimmed[i] < median / 100):
    #         min_index = i
    #         break
    # trimmed = trimmed[min_index:]

    print(f"Trimmed to {np.round(100 * len(trimmed) / len(eigs), 1)} percent of data.")
    return trimmed


def trim_middle(eigs: np.array, percent: float = 97.5):
    bottom = (100 - percent) / 2
    top = percent + bottom
    largest = np.percentile(eigs, top)
    smallest = np.percentile(eigs, bottom)
    trimmed = eigs[eigs < largest]
    trimmed = trimmed[trimmed > smallest]
    trimmed.sort()
    return trimmed


def trim_unfolded_via_spacings(unfolded: np.array) -> np.array:
    """assumes eigs are sorted"""
    spacings = computeSpacings(unfolded, trim=True)
    return unfolded[: len(spacings)]
