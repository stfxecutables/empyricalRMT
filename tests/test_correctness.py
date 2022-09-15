import numpy as np
import pytest
from pytest import CaptureFixture

from empyricalRMT._types import MatrixKind
from empyricalRMT.eigenvalues import Eigenvalues
from empyricalRMT.plot import PlotMode
from empyricalRMT.smoother import SmoothMethod

ENSEMBLES = ["goe", "poisson", "gue"]
L = np.arange(2, 20, 1, dtype=np.float64)


@pytest.mark.plot
def test_goe(capsys: CaptureFixture) -> None:
    with capsys.disabled():
        eigs = Eigenvalues.generate(5000, kind=MatrixKind.GOE, log_time=True)
        unfolded = eigs.unfold(smoother=SmoothMethod.GOE)
        unfolded.plot_observables(
            rigidity_L=L,
            levelvar_L=L,
            mode=PlotMode.Block,
            ensembles=ENSEMBLES,
            title="Spectral Observables - GOE",
        )


@pytest.mark.plot
def test_gue(capsys: CaptureFixture) -> None:
    with capsys.disabled():
        eigs = Eigenvalues.generate(5000, kind=MatrixKind.GUE, log_time=True)
        unfolded = eigs.unfold(smoother=SmoothMethod.Polynomial, degree=5)
        unfolded.plot_observables(
            rigidity_L=L,
            levelvar_L=L,
            mode=PlotMode.Block,
            ensembles=ENSEMBLES,
            title="Spectral Observables - GUE",
        )


@pytest.mark.plot
def test_poisson(capsys: CaptureFixture) -> None:
    with capsys.disabled():
        eigs = Eigenvalues.generate(10000, kind=MatrixKind.Poisson)
        # unfolded = eigs.unfold(smoother="poly", degree=5)
        unfolded = eigs.unfold(smoother=SmoothMethod.Poisson)
        # unfolded = eigs.unfold(smoother="gompertz")
        # unfolded = eigs.unfold(smoother="exp")
        unfolded.plot_observables(
            rigidity_L=L,
            levelvar_L=L,
            mode=PlotMode.Block,
            ensembles=ENSEMBLES,
            title="Spectral Observables - Poisson",
        )  #
