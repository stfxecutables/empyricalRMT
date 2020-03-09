import numpy as np
import pytest

from empyricalRMT.construct import goe_unfolded


@pytest.mark.fast
def test_levelvariance() -> None:
    unfolded = goe_unfolded(2000)
    unfolded.plot_level_variance(
        L=np.arange(2, 50, 0.2), max_L_iters=10000, show_progress=True, mode="test"
    )
