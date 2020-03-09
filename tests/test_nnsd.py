import numpy as np
from numpy import ndarray
import pytest

from pathlib import Path

from empyricalRMT.construct import _generate_GOE_matrix
from empyricalRMT.plot import _raw_eig_dist
from empyricalRMT.utils import eprint

CUR_DIR = Path(__file__).parent


def res(path: Path) -> str:
    return str(path.absolute().resolve())


def load_eigs(matsize: int = 10000) -> ndarray:
    eigs = None
    eigs_out = CUR_DIR / f"test_eigs{matsize}.npy"
    try:
        eigs = np.load(res(eigs_out))
    except IOError as e:
        M = _generate_GOE_matrix(matsize)
        eprint(e)
        eigs = np.linalg.eigvalsh(M)
        np.save("test_eigs.npy", res(eigs_out))

    return eigs


def generate_eigs(matsize: int) -> ndarray:
    M = _generate_GOE_matrix(matsize)
    eigs = np.linalg.eigvalsh(M)
    return eigs


@pytest.mark.fast
def test_semicircle(
    matsize: int = 1000, new_eigs: bool = False, eigs: ndarray = None
) -> None:
    if eigs is not None:
        pass  # use passed in eigenvalues
    else:
        eigs = generate_eigs(matsize) if new_eigs else load_eigs(matsize)
    _raw_eig_dist(
        eigs, bins=100, title="Wigner Semicircle PLotting Test", kde=False, mode="test"
    )
