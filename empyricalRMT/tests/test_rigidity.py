import pytest

from empyricalRMT.rmt.construct import generate_eigs
from empyricalRMT.rmt.eigenvalues import Eigenvalues


@pytest.mark.plot
def test_plot_rigidity() -> None:
    # good fit for max_L=50 when using generate_eigs(10000)
    # good fit for max_L=55 when using generate_eigs(20000)
    # not likely to be good fit for max_L beyond 20 for generate_eigs(1000)
    eigs = Eigenvalues(generate_eigs(2000))
    unfolded = eigs.unfold(smoother="poly", degree=9)
    unfolded.plot_spectral_rigidity(max_L=40, c_iters=5000)
