import numpy as np
import pandas as pd
import pytest

from empyricalRMT.smoother import Smoother
from empyricalRMT.construct import generate_eigs


@pytest.mark.fast
def test_smoother_class() -> None:
    # sanity
    eigs = np.sort(generate_eigs(200))
    smoother = Smoother(eigs)
    assert np.allclose(smoother._eigs, eigs)

    # fail on bad inputs
    with pytest.raises(ValueError):
        eigs = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        smoother = Smoother(eigs)

    with pytest.raises(ValueError):
        eigs = np.array([[1, 2, 3], [4, 5, 6]])
        smoother = Smoother(eigs)
