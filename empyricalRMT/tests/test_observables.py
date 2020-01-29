import numpy as np
import pytest

from empyricalRMT.rmt.construct import generateGOEMatrix, generatePoisson
from empyricalRMT.tests.test_levelvar import test_levelvariance
from empyricalRMT.tests.test_spectral import test_spectral_rigidity
from empyricalRMT.tests.test_nnsd import test_semicircle


def generateUniform(matsize=1000, lower=0, upper=1):
    pass


def generate_eigs(matsize, mean=0, sd=1, kind="goe"):
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

    eigs = np.linalg.eigvalsh(M)
    return eigs


@pytest.mark.old
def test_observables(
    matsize=1000, iters=1, semicircle=True, rigidity=True, levelvar=True, kind="goe"
):
    if not semicircle and not rigidity and not levelvar:
        raise ValueError("Must test at least one observable")
    for _ in range(iters):
        eigs = generate_eigs(matsize, kind=kind)
        if semicircle:
            test_semicircle(1000, eigs=eigs)
        if rigidity:
            test_spectral_rigidity(1000, eigs=eigs, kind=kind)
        if levelvar:
            test_levelvariance(1000, eigs=eigs, kind=kind)


@pytest.mark.old
def test_poisson_observables(
    matsize=1000, iters=1, semicircle=True, rigidity=True, levelvar=True
):
    if not semicircle and not rigidity and not levelvar:
        raise ValueError("Must test at least one observable")
    for _ in range(iters):
        eigs = generate_eigs(matsize, kind="poisson")
        if semicircle:
            test_semicircle(1000, eigs=eigs)
        if rigidity:
            test_spectral_rigidity(1000, eigs=eigs)
        if levelvar:
            test_levelvariance(1000, eigs=eigs)


@pytest.mark.old
def test_uniform_observables(
    matsize=1000, iters=1, semicircle=True, rigidity=True, levelvar=True
):
    if not semicircle and not rigidity and not levelvar:
        raise ValueError("Must test at least one observable")
    for _ in range(iters):
        eigs = generate_eigs(matsize, poisson=True)
        if semicircle:
            test_semicircle(1000, eigs=eigs)
        if rigidity:
            test_spectral_rigidity(1000, eigs=eigs)
        if levelvar:
            test_levelvariance(1000, eigs=eigs)
