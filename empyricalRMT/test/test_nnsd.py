import numpy as np

from pathlib import Path

from ..rmt.construct import generateGOEMatrix
from ..rmt.eigenvalues import getEigs, trim_iteratively
from ..rmt.plot import rawEigDist
from ..rmt.plot import spacings as plotSpacings
from ..utils import eprint

CUR_DIR = Path(__file__).parent


def res(path) -> str:
    return str(path.absolute().resolve())


def load_eigs(matsize=10000):
    eigs = None
    eigs_out = CUR_DIR / f"test_eigs{matsize}.npy"
    try:
        eigs = np.load(res(eigs_out))
    except IOError as e:
        M = generateGOEMatrix(matsize)
        eprint(e)
        eigs = getEigs(M)
        np.save("test_eigs.npy", res(eigs_out))

    return eigs


def new_eigs(matsize):
    M = generateGOEMatrix(matsize)
    eigs = getEigs(M)
    return eigs


def test_semicircle(matsize=1000, neweigs=False, eigs=None, use_brain=False):
    if eigs is not None:
        pass  # use passed in eigenvalues
    elif use_brain:
        eigs = np.load(
            "/home/derek/Desktop/fMRI_Data/Rest+Various/ds000224-download/derivatives/"
            "rmt/sub-MSC01/eigs_corrmat_sub-MSC01_ses-func03_task-rest_bold.npy"
        )
        eigs = trim_iteratively(eigs)
    else:
        eigs = new_eigs(matsize) if neweigs else load_eigs(matsize)
    rawEigDist(eigs, bins=100, title="Wigner Semicircle", block=True)


def test_nnsd(matsize=10000, neweigs=False, eigs=None, kind="goe"):
    if eigs is not None:
        pass  # use passed in eigenvalues
    else:
        eigs = new_eigs(matsize) if neweigs else load_eigs(matsize)
    spacings = eigs[1:] - eigs[:-1]
    D = np.mean(spacings)
    s = spacings / D

    plotSpacings(s, kde=True, title=f"NNSD of {kind.upper()} N={matsize} Matrix")
