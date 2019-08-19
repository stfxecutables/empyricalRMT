import numpy as np

from ..rmt.construct import generateGOEMatrix, generatePoisson
from ..rmt.eigenvalues import getEigs
from ..test.levelvar_test import test_levelvariance
from ..test.spectral_test import test_spectral_rigidity
from ..test.test_nnsd import test_nnsd, test_semicircle


def generateUniform(matsize=1000, lower=0, upper=1):
    pass


def newEigs(matsize, mean=0, sd=1, kind="goe"):
    if kind == "poisson":
        M = generatePoisson(matsize)
    elif kind == "uniform":
        raise Exception("UNIMPLEMENTED!")
        M = generateUniform(matsize)
    elif kind == "gue":
        size = [matsize, matsize]
        A = np.random.standard_normal(size) + 1j * np.random.standard_normal(size)
        M = (A + A.conjugate().T) / 2
    elif kind == "goe":
        M = generateGOEMatrix(matsize, mean, sd)
    else:
        kinds = ["goe", "gue", "poisson", "uniform"]
        raise ValueError(f"`kind` must be one of {kinds}")

    eigs = getEigs(M)
    return eigs


def test_observables(
    matsize=1000,
    iters=2,
    semicircle=True,
    nnsd=True,
    rigidity=True,
    levelvar=True,
    kind="goe",
):
    if not semicircle and not nnsd and not rigidity and not levelvar:
        raise ValueError("Must test at least one observable")
    for _ in range(2):
        eigs = newEigs(matsize, kind=kind)
        if semicircle:
            test_semicircle(1000, eigs=eigs)
        if nnsd:
            test_nnsd(1000, eigs=eigs, kind=kind)
        if rigidity:
            test_spectral_rigidity(1000, eigs=eigs, kind=kind)
        if levelvar:
            test_levelvariance(1000, eigs=eigs, kind=kind)


def test_poisson_observables(
    matsize=1000, iters=2, semicircle=True, nnsd=True, rigidity=True, levelvar=True
):
    if not semicircle and not nnsd and not rigidity and not levelvar:
        raise ValueError("Must test at least one observable")
    for _ in range(2):
        eigs = newEigs(matsize, kind="poisson")
        if semicircle:
            test_semicircle(1000, eigs=eigs)
        if nnsd:
            test_nnsd(1000, eigs=eigs)
        if rigidity:
            test_spectral_rigidity(1000, eigs=eigs)
        if levelvar:
            test_levelvariance(1000, eigs=eigs)


def test_uniform_observables(
    matsize=1000, iters=2, semicircle=True, nnsd=True, rigidity=True, levelvar=True
):
    if not semicircle and not nnsd and not rigidity and not levelvar:
        raise ValueError("Must test at least one observable")
    for _ in range(2):
        eigs = newEigs(matsize, poisson=True)
        if semicircle:
            test_semicircle(1000, eigs=eigs)
        if nnsd:
            test_nnsd(1000, eigs=eigs)
        if rigidity:
            test_spectral_rigidity(1000, eigs=eigs)
        if levelvar:
            test_levelvariance(1000, eigs=eigs)
