import numpy as np
import pytest

from empyricalRMT.construct import generate_eigs
from empyricalRMT.eigenvalues import Eigenvalues


@pytest.mark.plot
def test_plot_rigidity() -> None:
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
    unfolded.plot_spectral_rigidity(
        L=np.arange(1, 200, 0.5), c_iters=10000, mode="test"
    )
