import matplotlib.pyplot as plt
import numpy as np
import pytest

from empyricalRMT.eigenvalues import Eigenvalues
from empyricalRMT.construct import generate_eigs


@pytest.mark.plot
def test_brody_fit() -> None:
    # test GOE eigs
    for _ in range(4):
        eigs = generate_eigs(1000)
        Eigenvalues(eigs).unfold(degree=7).plot_nnsd(brody=True, mode="noblock")

    # test time series
    for _ in range(4):
        eigs = np.linalg.eigvalsh(np.corrcoef(np.random.standard_normal([2000, 500])))
        # eigs = eigs[eigs > 100 * np.abs(eigs.min())]
        Eigenvalues(eigs).unfold(degree=7).plot_nnsd(
            brody=True, mode="noblock", title="Time Series"
        )
    plt.show()
