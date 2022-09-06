import numpy as np
import pytest

from empyricalRMT.construct import goe_unfolded
from empyricalRMT.plot import PlotMode


@pytest.mark.fast
def test_plotting() -> None:
    unfolded = goe_unfolded(2000)
    unfolded.plot_level_variance(
        L=np.arange(2, 50, 0.5),
        max_L_iters=int(1e6),
        show_progress=True,
        mode=PlotMode.Test,
    )


@pytest.mark.fast
def test_levelvar() -> None:
    unfolded = goe_unfolded(5000)
    unfolded.level_variance(
        L=np.arange(0.5, 50, 0.5),
        tol=0.01,
        max_L_iters=int(1e6),
        min_L_iters=1000,
        show_progress=True,
    )
