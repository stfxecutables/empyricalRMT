import numpy as np
import pytest
from pytest import CaptureFixture

from empyricalRMT.construct import goe_unfolded
from empyricalRMT.plot import PlotMode


@pytest.mark.fast
def test_plotting(capsys: CaptureFixture) -> None:
    unfolded = goe_unfolded(10000)
    with capsys.disabled():
        unfolded.plot_level_variance(
            L=np.arange(2, 50, 0.5),
            tol=0.01,
            max_L_iters=int(1e6),
            min_L_iters=1000,
            show_progress=True,
            mode=PlotMode.Test,
        )


def test_convergence(capsys: CaptureFixture) -> None:
    unfolded = goe_unfolded(10000)
    with capsys.disabled():
        unfolded.plot_level_variance(
            L=np.arange(20, 30, 0.5),
            tol=0.01,
            max_L_iters=int(1e6),
            min_L_iters=int(1e5),
            show_progress=True,
            mode=PlotMode.NoBlock,
        )
        unfolded.plot_level_variance(
            L=np.arange(20, 30, 0.5),
            tol=0.01,
            max_L_iters=int(1e8),
            min_L_iters=int(1e5),
            show_progress=True,
            mode=PlotMode.Block,
        )


@pytest.mark.fast
def test_levelvar(capsys: CaptureFixture) -> None:
    unfolded = goe_unfolded(5000)
    with capsys.disabled():
        unfolded.level_variance(
            L=np.arange(0.5, 50, 0.5),
            tol=0.01,
            max_L_iters=int(1e6),
            min_L_iters=1000,
            show_progress=True,
        )
