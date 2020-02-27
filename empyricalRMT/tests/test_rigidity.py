import numpy as np
import pandas as pd
import pytest

from empyricalRMT.rmt.construct import generate_eigs, goe_unfolded
from empyricalRMT.rmt.eigenvalues import Eigenvalues
from empyricalRMT.rmt.plot import _spectral_rigidity
from empyricalRMT.rmt.unfold import Unfolded


@pytest.mark.plot
def test_plot_rigidity() -> None:
    # good fit for max_L=50 when using generate_eigs(10000)
    # good fit for max_L=55 when using generate_eigs(20000)
    # not likely to be good fit for max_L beyond 20 for generate_eigs(1000)
    # L good | len(eigs) |     percent
    # -----------------------------------
    # 30-40  |    2000   |
    # 30-50  |    8000   |  0.375 - 0.625
    # 50-70  |   10000   |    0.5 - 0.7
    #   50   |   20000   |      0.25
    eigs = Eigenvalues(generate_eigs(2000, log=True))
    unfolded = eigs.unfold(smoother="poly", degree=19)

    unfolded.plot_nnsd(mode="noblock")
    # unfolded.plot_next_nnsd(mode="block")
    unfolded.plot_level_variance(
        min_L=1,
        max_L=100,
        L_grid_size=100,
        mode="noblock",
        ensembles=["goe", "poisson"],
    )
    unfolded.plot_spectral_rigidity(max_L=200, c_iters=10000)


@pytest.mark.plot
def test_average_rigidity() -> None:
    n = 5
    size = 5000
    L, delta3 = None, []
    for i in range(n):
        unfolded = goe_unfolded(size, log=True)
        df = unfolded.spectral_rigidity(
            min_L=5, max_L=100, c_iters=10000, show_progress=True
        )
        if i == 0:
            L = df["L"]
        delta3.append(df["delta"].to_numpy())

    delta3 = np.mean(np.array(delta3), axis=0)  # vertically stacked rows of delta3
    df = pd.DataFrame({"L": L, "delta": delta3})
    _spectral_rigidity(
        unfolded=None,
        data=df,
        title=f"Rigidity: average of {n} N={size} GOE",
        ensembles=["goe", "poisson"],
    )

