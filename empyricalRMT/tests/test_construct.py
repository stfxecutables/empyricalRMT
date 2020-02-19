import pytest

from empyricalRMT.rmt.construct import fast_poisson_eigs
from empyricalRMT.rmt.eigenvalues import Eigenvalues
from empyricalRMT.rmt.plot import _spacings as plotSpacings


@pytest.mark.construct
@pytest.mark.fast
def test_fast_poisson() -> None:
    for i in range(1):
        vals = fast_poisson_eigs(5000)
        unfolded = Eigenvalues(vals).unfold()
        plotSpacings(
            unfolded, title="Fast Poisson Spacing Test", bins=20, kde=True, mode="block"
        )
