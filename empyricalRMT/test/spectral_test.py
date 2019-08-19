import numpy as np
import pandas as pd

from pathlib import Path

import rmt.unfold as unfold
import rmt.plot

from rmt.construct import generateGOEMatrix
from rmt.eigenvalues import getEigs, trim_iteratively
from rmt.observables.rigidity import spectralRigidityRewrite
from rmt.plot import spectralRigidity as plotSpectral
from utils import eprint

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
        eigs = getEigs(M)
        np.save(filename, res(eigs_out))

    return eigs


def new_eigs(matsize):
    M = generateGOEMatrix(matsize)
    eigs = getEigs(M)
    return eigs


def test_spectral_rigidity(
    matsize=1000,
    neweigs=True,
    eigs=None,
    plot_step=False,
    unfold_degree=None,
    use_brain=False,
    kind="goe",
):
    unfolded = None
    if eigs is not None:
        pass
        unfolded = unfold.polynomial(eigs, 11)
    elif use_brain:
        eigs = np.load(
            "/home/derek/Desktop/fMRI_Data/Rest+Various/ds000224-download/derivatives/"
            "rmt/sub-MSC01/eigs_corrmat_sub-MSC01_ses-func03_task-rest_bold.npy"
        )
        eigs = trim_iteratively(eigs)
        unfolded = unfold.polynomial(eigs, unfold_degree, 10000, None, None)
    else:
        eigs = new_eigs(matsize) if neweigs else load_eigs(matsize)
        unfolded = unfold.polynomial(eigs, 11)

    if plot_step:
        rmt.plot.stepFunction(eigs, trim=False, block=True)

    L_vals, delta3 = spectralRigidityRewrite(
        unfolded, eigs, c_iters=2000, L_grid_size=100, min_L=0.5, max_L=25
    )
    df = pd.DataFrame({"L": L_vals, "∆3(L)": delta3})
    plotSpectral(unfolded, df, f"{kind.upper()} Matrix", mode="block")
