import pytest

from empyricalRMT.rmt.construct import fast_poisson_eigs
from empyricalRMT.rmt.plot import spacings as plotSpacings
from empyricalRMT.rmt.unfold import Unfolder


@pytest.mark.construct
def test_fast_poisson():
    for i in range(10):
        eigs = fast_poisson_eigs(5000)
        unfolded = Unfolder(eigs).unfold(trim=False)
        plotSpacings(unfolded, bins=20, kde=True, mode="block")

