import numpy as np
import pytest

from empyricalRMT.rmt.construct import goe_unfolded


@pytest.mark.fast
def test_levelvariance() -> None:
    unfolded = goe_unfolded(50000)
    unfolded.plot_level_variance(
        L=np.arange(2, 100, 0.2), show_progress=True, mode="block"
    )
