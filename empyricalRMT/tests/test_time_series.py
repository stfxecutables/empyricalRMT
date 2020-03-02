import matplotlib.pyplot as plt
import numpy as np
import pytest
import time

from numpy import ndarray

from empyricalRMT.rmt.construct import generate_eigs
from empyricalRMT.rmt.correlater import p_correlate
from empyricalRMT.rmt.eigenvalues import Eigenvalues
from empyricalRMT.rmt.plot import _observables


def get_eigs(arr: ndarray) -> ndarray:
    print(f"\n{time.strftime('%H:%M:%S (%b%d)')} -- computing eigenvalues...")
    eigs = np.linalg.eigvalsh(arr)
    print(f"\n{time.strftime('%H:%M:%S (%b%d)')} -- computed eigenvalues...")
    return eigs


def unfold_and_plot(eigs: ndarray, suptitle: str) -> None:
    unfolded = Eigenvalues(eigs).trim_unfold_auto(
        max_trim=0.5, max_iters=9, poly_degrees=[13], gompertz=False
    )
    trimmed = np.round(100 * len(unfolded.vals) / len(eigs), 1)

    _observables(
        unfolded=unfolded.vals,
        rigidity_df=unfolded.spectral_rigidity(c_iters=10000, show_progress=True),
        levelvar_df=unfolded.level_variance(show_progress=True),
        suptitle=suptitle + f" ({100 - trimmed}% removed)",
        mode="noblock",
    )
    # unfolded.plot_steps(mode="noblock")
    # unfolded.plot_nnsd(ensembles=["goe", "poisson"], mode="noblock")
    # unfolded.plot_spectral_rigidity(
    #     ensembles=["goe", "poisson"], c_iters=20000, mode="noblock"
    # )
    # unfolded.plot_level_variance(
    #     ensembles=["goe", "poisson"], max_L_iters=10000, mode="block"
    # )


@pytest.mark.plot
def test_gaussian_noise() -> None:
    A = np.random.standard_normal([1000, 250])
    M = p_correlate(A)
    eigs = get_eigs(M)
    unfold_and_plot(eigs, "Gaussian Noise")


@pytest.mark.plot
def test_correlated_gaussian_noise() -> None:
    var = 0.5
    for percent in [25, 50, 75, 95]:
        A = np.random.standard_normal([1000, 250])
        correlated = np.random.permutation(A.shape[0] - 1) + 1  # don't select first row
        last = int(np.floor((percent / 100) * A.shape[0]))
        corr_indices = correlated[:last]
        # introduce correlation in A
        for i in corr_indices:
            A[i, :] = np.random.uniform(0, 1) * A[0, :] + np.random.normal(
                0, var, size=A.shape[1]
            )
        M = p_correlate(A)
        eigs = get_eigs(M)
        print(f"\nPercent correlated noise: {percent}%")
        unfold_and_plot(eigs, f"\nCorrelated noise: {percent}%")
    plt.show()


@pytest.mark.plot
def test_uniform_noise() -> None:
    A = np.random.uniform(0, 1, size=[1000, 250])
    M = p_correlate(A)
    eigs = get_eigs(M)
    unfold_and_plot(eigs, "Uniform Noise")
