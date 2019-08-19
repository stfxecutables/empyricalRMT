import numpy as np

from utils import eprint


def almostIdentity(size: int = 100):
    E = np.random.standard_normal(size * size).reshape(size, size)
    E = (E + E.T) / np.sqrt(2)
    M = np.ma.identity(size)
    return M + E


def random_1vector(size: int = 100):
    vals = np.random.standard_normal([size, 1])
    return vals * vals.T


def generateGOEMatrix(size: int = 100, mean=0, sd=1):
    if mean != 0 or sd != 1:
        M = np.random.normal(mean, sd, [size, size])
    else:
        M = np.random.standard_normal([size, size])
    return (M + M.T) / np.sqrt(2)


def generatePoisson(size: int = 100):
    return np.diag(np.random.standard_normal(size))


def generateRandomMatrix(size: int = 100):
    norm_means = np.abs(np.random.normal(10.0, 2.0, size=size * size))
    norm_sds = np.abs(np.random.normal(3.0, 0.5, size=size * size))
    # exp_rates = np.abs(np.random.normal(size=size*size))
    M = np.empty([size, size])
    eprint("Initialized matrix distribution data")

    # TODO: generate NaNs and Infs

    it = np.nditer(M, flags=["f_index"], op_flags=["readwrite"])
    while not it.finished:
        i = it.index
        # original formulation
        # it[0] = np.random.normal(norm_means[i], norm_sds[i], 1) +\
        #     np.random.exponential(exp_rates[i], 1)

        # independent random normals
        it[0] = np.random.normal(norm_means[i], norm_sds[i], 1)

        # one random normal
        it[0] = np.random.normal(0, 1, 1)
        it.iternext()
    it.close()
    eprint("Filled matrix")
    return M
