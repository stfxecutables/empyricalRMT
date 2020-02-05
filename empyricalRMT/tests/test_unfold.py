import numpy as np
import os
import pytest

from pathlib import Path

from empyricalRMT.rmt.eigenvalues import Eigenvalues
from empyricalRMT.rmt.construct import generateGOEMatrix
from empyricalRMT.rmt.trim import TrimReport
from empyricalRMT.rmt.unfolder import Unfolder


@pytest.mark.fast
@pytest.mark.unfolder
def test_unfold_init() -> None:
    M = generateGOEMatrix(2000)
    eigs = np.linalg.eigvalsh(M)
    unfolder = Unfolder(eigs)
    assert np.alltrue(unfolder.eigs == eigs)
    assert np.alltrue(unfolder.eigenvalues == eigs)

