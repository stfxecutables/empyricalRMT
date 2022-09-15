import numpy as np
from pytest import CaptureFixture

from empyricalRMT._types import MatrixKind
from empyricalRMT.eigenvalues import Eigenvalues
from empyricalRMT.plot import PlotMode
from empyricalRMT.smoother import SmoothMethod


def test_plot_rigidity(capsys: CaptureFixture) -> None:
    with capsys.disabled():
        # eigs = Eigenvalues.generate(2000, kind="goe", log=True)
        eigs = Eigenvalues.generate(10000, kind=MatrixKind.Poisson, log_time=True)
        # eigs = Eigenvalues.generate(10000, kind="goe", log=True)
        # eigs = Eigenvalues.generate(2000, kind="goe", log=True)
        # eigs = Eigenvalues.generate(5000, kind="gue", log=True)
        unfolded = eigs.unfold(smoother=SmoothMethod.Polynomial, degree=19)
        # unfolded = eigs.unfold(smoother="goe")
        df = unfolded.spectral_rigidity(
            # L=np.arange(2, 100, 1, dtype=np.float64),
            L=np.arange(2, 20, 1, dtype=np.float64),
            integration="simps",
            tol=0.01,
            show_progress=True,
        )
        print(df.tail())
        unfolded.plot_spectral_rigidity(
            data=df,
            ensembles=["goe", "gue", "poisson"],
            mode=PlotMode.Block,
            show_iters=True,
        )
