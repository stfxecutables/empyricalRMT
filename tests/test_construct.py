import pytest

from empyricalRMT.eigenvalues import Eigenvalues
from empyricalRMT.plot import PlotMode


@pytest.mark.construct
@pytest.mark.fast
def test_poisson() -> None:
    for i in range(1):
        eigs = Eigenvalues.generate(5000, kind="poisson")
        unfolded = eigs.unfold()
        unfolded.plot_nnsd(
            title="Poisson Spacing Test",
            bins=10,
            kde=True,
            mode=PlotMode.Test,
            ensembles=["poisson"],
        )
