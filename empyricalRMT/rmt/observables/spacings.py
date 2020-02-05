from numpy import ndarray

from numba import jit

# this is essentially the nearest neighbours spacing distribution
@jit(nopython=True, fastmath=True)
def computeSpacings(unfolded: ndarray, trim: bool = True) -> ndarray:
    spacings = unfolded[1:] - unfolded[:-1]
    if trim:
        spacings = spacings[spacings < 10]
    return spacings
