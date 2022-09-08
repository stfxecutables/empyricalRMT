import numpy as np
import pytest
from pytest import CaptureFixture

from empyricalRMT.construct import goe_unfolded
from empyricalRMT.plot import PlotMode


@pytest.mark.fast
def test_plotting(capsys: CaptureFixture) -> None:
    unfolded = goe_unfolded(15000)
    with capsys.disabled():
        unfolded.plot_level_variance(
            L=np.arange(2, 30, 0.5),
            tol=0.01,
            show_progress=True,
            mode=PlotMode.Block,
            show_iters=True,
        )


@pytest.mark.fast
def test_levelvar(capsys: CaptureFixture) -> None:
    unfolded = goe_unfolded(5000)
    with capsys.disabled():
        df = unfolded.level_variance(
            L=np.arange(0.5, 100, 0.5),
            # L=np.arange(50, 100, 0.5),
            tol=0.01,
            max_L_iters=int(1e6),
            min_L_iters=int(1e2),
            show_progress=True,
        )
        print(df)
