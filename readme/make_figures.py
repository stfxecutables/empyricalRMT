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


def make_unfolding_compare_plots() -> None:
    import matplotlib.pyplot as plt

    from empyricalRMT.eigenvalues import Eigenvalues
    from empyricalRMT.smoother import SmoothMethod

    eigs = Eigenvalues.generate(1000, kind="goe")
    unfoldings = {
        "Exponential": eigs.unfold(smoother=SmoothMethod.Exponential),
        "Polynomial": eigs.unfold(smoother=SmoothMethod.Polynomial, degree=5),
        "Gompertz": eigs.unfold(smoother=SmoothMethod.Gompertz),
        "GOE": eigs.unfold(smoother=SmoothMethod.GOE),
    }
    N = len(unfoldings)
    fig, axes = plt.subplots(ncols=2, nrows=N)
    for i, (label, unfolded) in enumerate(unfoldings.items()):
        title = f"{label} Unfolding"
        unfolded.plot_fit(title=title, fig=fig, axes=axes[i][0])
        unfolded.plot_nnsd(
            title=title,
            brody=True,
            brody_fit="mle",
            ensembles=["goe", "poisson"],
            fig=fig,
            axes=axes[i][1],
        )
        axes[i][0].legend().set_visible(False) if i != 0 else None
        axes[i][1].legend().set_visible(False) if i != 0 else None
    fig.subplots_adjust(top=0.95, bottom=0.05, left=0.08, right=0.95, hspace=0.383)
    fig.set_size_inches(w=8.5, h=11)
    fig.savefig(OUTDIR / "unfold_compare.png", dpi=300)


if __name__ == "__main__":
    # make_observables_plot()
    make_unfolding_compare_plots()
