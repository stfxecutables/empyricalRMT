import pytest

from empyricalRMT.rmt.construct import fast_poisson_eigs
from empyricalRMT.rmt.plot import _spacings as plotSpacings
from empyricalRMT.rmt.unfolder import Unfolder


@pytest.mark.construct
@pytest.mark.fast
def test_fast_poisson() -> None:
    for i in range(1):
        eigs = fast_poisson_eigs(5000)
        unfolded = Unfolder(eigs).unfold(trim=False)
        plotSpacings(
            unfolded, title="Fast Poisson Spacing Test", bins=20, kde=True, mode="block"
        )
