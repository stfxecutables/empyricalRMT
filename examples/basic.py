try:  # just a hack to make running these in development easier
    import empyricalRMT
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np

from empyricalRMT._types import MatrixKind
from empyricalRMT.eigenvalues import Eigenvalues
from empyricalRMT.smoother import SmoothMethod

# generate eigenvalues from a 2000x2000 sample from the Gaussian Orthogonal Ensemble
eigs = Eigenvalues.generate(matsize=2000, kind=MatrixKind.GOE)
# unfold "analytically" using Wigner semi-circle
unfolded = eigs.unfold(smoother=SmoothMethod.GOE)
# visualize core spectral observables and unfolding fit
unfolded.plot_observables(
    rigidity_L=np.arange(2, 20, 0.5),
    levelvar_L=np.arange(2, 20, 0.5),
    title="GOE Spectral Observables - Analytic Unfolding",
    ensembles=["goe"],
)
