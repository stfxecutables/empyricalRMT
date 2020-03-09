import numpy as np
import pytest

from empyricalRMT.eigenvalues import Eigenvalues
from empyricalRMT.construct import generate_eigs


@pytest.mark.fast
def test_unfold_methods() -> None:
    eigs = Eigenvalues(generate_eigs(500, seed=2))
    trimmed = eigs.get_best_trimmed()
    print("Trim starts and ends:")
    print(trimmed.vals[0])
    print(trimmed.vals[-1])
    assert np.allclose(trimmed.vals[0], -35.84918623729985)
    assert np.allclose(trimmed.vals[-1], 34.709818777689364)

    unfolded = eigs.trim_unfold_auto()
    print("Trim starts and ends:")
    print(unfolded.vals[0])
    print(unfolded.vals[-1])
    assert np.allclose(unfolded.vals[0], -2.473290621491799)
    assert np.allclose(unfolded.vals[-1], 504.2764217889801)
