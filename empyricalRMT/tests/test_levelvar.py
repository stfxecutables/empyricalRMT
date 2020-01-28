import numpy as np
import pandas as pd

from pathlib import Path

import empyricalRMT.rmt.unfold as unfold

from empyricalRMT.rmt.construct import generateGOEMatrix
from empyricalRMT.rmt._eigvals import trim_iteratively
from empyricalRMT.rmt.observables.levelvariance import sigmaSquared
from empyricalRMT.rmt.plot import levelNumberVariance
from empyricalRMT.utils import eprint

CUR_DIR = Path(__file__).parent


def res(path) -> str:
    return str(path.absolute().resolve())


def load_eigs(matsize=10000):
    eigs = None
    filename = f"test_eigs{matsize}.npy"
    eigs_out = CUR_DIR / filename
    try:
        eigs = np.load(res(eigs_out))
    except IOError as e:
        M = generateGOEMatrix(matsize)
        eprint(e)
        eigs = np.linalg.eigvalsh(M)
        np.save(filename, res(eigs_out))

    return eigs


def newEigs(matsize):
    M = generateGOEMatrix(matsize)
    eigs = np.linalg.eigvalsh(M)
    return eigs


def test_levelvariance(matsize=1000, neweigs=True, eigs=None, kind="goe"):
    unfolded = None
    if eigs is not None:
        unfolded = unfold.polynomial(
            eigs, 11, 10000
        )  # eigs are GOE, so we don't need to unfold
    else:
        eigs = newEigs(matsize) if neweigs else load_eigs(matsize)
        unfolded = unfold.polynomial(
            eigs, 11, 10000
        )  # eigs are GOE, so we don't need to unfold

    L_grid, sigma_sq = sigmaSquared(
        eigs, unfolded, c_iters=1000, L_grid_size=100, min_L=0.5, max_L=20
    )
    df = pd.DataFrame({"L": L_grid, "∑²(L)": sigma_sq})
    levelNumberVariance(unfolded, df, title=f"{kind.upper()} Matrix", mode="block")
