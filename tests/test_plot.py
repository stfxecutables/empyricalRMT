import numpy as np

from empyricalRMT.eigenvalues import Eigenvalues
from empyricalRMT.plot import PlotMode
from empyricalRMT.smoother import SmoothMethod


def test_plot_calls() -> None:
    eigs = Eigenvalues.generate(2000, log_time=True)
    unfolded = eigs.unfold(smoother=SmoothMethod.Polynomial, degree=19)

    unfolded.plot_nnsd(mode=PlotMode.Test)
    # unfolded.plot_next_nnsd(mode=PlotMode.Test)
    L = np.arange(2, 20, 1, dtype=np.float64)
    unfolded.plot_level_variance(L=L, mode=PlotMode.Test, ensembles=["goe", "poisson"])
    unfolded.plot_spectral_rigidity(L=L, mode=PlotMode.Test)
    unfolded.plot_observables(
        rigidity_L=L,
        levelvar_L=L,
        mode=PlotMode.Test,
        ensembles=["goe", "poisson"],
    )
