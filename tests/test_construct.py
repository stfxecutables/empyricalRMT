import pytest

from empyricalRMT.construct import generate_eigs
from empyricalRMT.eigenvalues import Eigenvalues


@pytest.mark.construct
@pytest.mark.fast
def test_poisson() -> None:
    for i in range(1):
        vals = generate_eigs(5000, kind="poisson")
        unfolded = Eigenvalues(vals).unfold()
        unfolded.plot_nnsd(
            title="Poisson Spacing Test",
            bins=10,
            kde=True,
            mode="test",
            ensembles=["poisson"],
        )
