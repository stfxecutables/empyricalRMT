# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort:skip
sys.path.append(str(Path(__file__).resolve().parent.parent))
# fmt: on

import numpy as np

from empyricalRMT._types import MatrixKind
from empyricalRMT.eigenvalues import Eigenvalues
from empyricalRMT.plot import PlotMode
from empyricalRMT.smoother import SmoothMethod

OUTDIR = Path(__file__).resolve().parent


def make_observables_plot() -> None:
    # generate eigenvalues from a 5000x5000 sample from the Gaussian Orthogonal Ensemble
    eigs = Eigenvalues.generate(matsize=5000, kind=MatrixKind.GOE)
    # unfold "analytically" using Wigner semi-circle
    unfolded = eigs.unfold(smoother=SmoothMethod.GOE)
    # visualize core spectral observables and unfolding fit
    unfolded.plot_observables(
        rigidity_L=np.arange(2, 20, 0.5),
        levelvar_L=np.arange(2, 20, 0.5),
        title="GOE Spectral Observables - Analytic Unfolding",
        ensembles=["goe"],
        mode=PlotMode.Save,
        outfile=OUTDIR / "observables.png",
    )


if __name__ == "__main__":
    make_observables_plot()
