import numpy as np
import pytest
import time

from numpy import ndarray


from empyricalRMT.eigenvalues import Eigenvalues
from empyricalRMT.construct import goe_unfolded
from empyricalRMT.correlater import correlate_fast


def get_eigs(arr: ndarray) -> ndarray:
    print(f"\n{time.strftime('%H:%M:%S (%b%d)')} -- computing eigenvalues...")
    eigs = np.linalg.eigvalsh(arr)
    print(f"\n{time.strftime('%H:%M:%S (%b%d)')} -- computed eigenvalues...")
    return eigs


@pytest.mark.plot
def test_axes_configuring() -> None:
    var = 0.1
    percent = 25
    A = np.random.standard_normal([1000, 500])
    correlated = np.random.permutation(A.shape[0] - 1) + 1  # don't select first row
    last = int(np.floor((percent / 100) * A.shape[0]))
    corr_indices = correlated[:last]
    # introduce correlation in A
    for i in corr_indices:
        A[i, :] = np.random.uniform(1, 2) * A[0, :] + np.random.normal(
            0, var, size=A.shape[1]
        )
    M = correlate_fast(A)
    eigs = get_eigs(M)
    print(f"\nPercent correlated noise: {percent}%")
    unfolded = Eigenvalues(eigs).unfold(degree=13)
    unfolded.plot_fit(mode="noblock")

    goe_unfolded(1000, log=True).plot_fit(mode="block")
