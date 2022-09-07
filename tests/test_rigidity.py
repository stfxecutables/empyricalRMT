import matplotlib.pyplot as plt
import numpy as np
import pytest
from pytest import CaptureFixture

from empyricalRMT.construct import generate_eigs
from empyricalRMT.eigenvalues import Eigenvalues
from empyricalRMT.plot import PlotMode


@pytest.mark.plot
def test_plot_call() -> None:
    # good fit for max_L=50 when using generate_eigs(10000)
    # good fit for max_L=55 when using generate_eigs(20000)
    # not likely to be good fit for max_L beyond 20 for generate_eigs(1000)
    # L good | len(eigs) |     percent
    # -----------------------------------
    # 30-40  |    2000   |
    # 30-50  |    8000   |  0.375 - 0.625
    # 50-70  |   10000   |    0.5 - 0.7
    #   50   |   20000   |      0.25
    eigs = Eigenvalues(generate_eigs(2000, log=True))
    unfolded = eigs.unfold(smoother="poly", degree=19)

    unfolded.plot_nnsd(mode="test")
    # unfolded.plot_next_nnsd(mode="test")
    unfolded.plot_level_variance(
        L=np.arange(0.5, 100, 0.2), mode="test", ensembles=["goe", "poisson"]
    )
    unfolded.plot_spectral_rigidity(L=np.arange(1, 200, 0.5), c_iters=10000, mode="test")


@pytest.mark.plot
def test_plot_rigidity(capsys: CaptureFixture) -> None:
    # good fit for max_L=50 when using generate_eigs(10000)
    # good fit for max_L=55 when using generate_eigs(20000)
    # not likely to be good fit for max_L beyond 20 for generate_eigs(1000)
    # L good | len(eigs) |     percent
    # -----------------------------------
    # 30-40  |    2000   |
    # 30-50  |    8000   |  0.375 - 0.625
    # 50-70  |   10000   |    0.5 - 0.7
    #   50   |   20000   |      0.25
    with capsys.disabled():
        # eigs = Eigenvalues(generate_eigs(2000, kind="goe", log=True))
        eigs = Eigenvalues(generate_eigs(10000, kind="goe", log=True))
        # eigs = Eigenvalues(generate_eigs(5000, kind="gue", log=True))
        # unfolded = eigs.unfold(smoother="poly", degree=19)
        unfolded = eigs.unfold(smoother="goe")
        df = unfolded.spectral_rigidity(
            L=np.arange(2, 100, 1),
            integration="simps",
            gridsize=int(1e2),
            max_iters=int(1e6),
            min_iters=int(1e3),
            tol=0.01,
            show_progress=True,
        )
        print(df.tail())
        unfolded.plot_spectral_rigidity(
            data=df,
            ensembles=["goe", "gue"],
            mode=PlotMode.NoBlock,
            show_iters=True,
        )
        plt.show()
