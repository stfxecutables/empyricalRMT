import numpy as np
from numpy import ndarray
import pytest

from empyricalRMT.rmt.construct import generateGOEMatrix, generatePoisson
from empyricalRMT.tests.test_levelvar import test_levelvariance
from empyricalRMT.tests.test_spectral import test_spectral_rigidity
from empyricalRMT.tests.test_nnsd import test_semicircle


def generateUniform(matsize: int = 1000, lower: float = 0, upper: float = 1) -> ndarray:
    raise NotImplementedError


def generate_eigs(
    matsize: int, mean: float = 0, sd: float = 1, kind: str = "goe"
) -> ndarray:
    if kind == "poisson":
        M = generatePoisson(matsize)
    elif kind == "uniform":
        raise Exception("UNIMPLEMENTED!")
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
    matsize: int = 1000,
    iters: int = 1,
    semicircle: bool = True,
    rigidity: bool = True,
    levelvar: bool = True,
    kind: str = "goe",
) -> None:
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
    matsize: int = 1000,
    iters: int = 1,
    semicircle: bool = True,
    rigidity: bool = True,
    levelvar: bool = True,
) -> None:
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
    matsize: int = 1000,
    iters: int = 1,
    semicircle: bool = True,
    rigidity: bool = True,
    levelvar: bool = True,
) -> None:
    if not semicircle and not rigidity and not levelvar:
        raise ValueError("Must test at least one observable")
    for _ in range(iters):
        eigs = generate_eigs(matsize)
        if semicircle:
            test_semicircle(1000, eigs=eigs)
        if rigidity:
            test_spectral_rigidity(1000, eigs=eigs)
        if levelvar:
            test_levelvariance(1000, eigs=eigs)
