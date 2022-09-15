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
    fig, axes = plt.subplots(ncols=3, nrows=N)
    for i, (label, unfolded) in enumerate(unfoldings.items()):
        title = f"{label} Unfolding"
        unfolded.plot_nnsd(
            title=title,
            brody=True,
            brody_fit="mle",
            ensembles=["goe", "poisson"],
            fig=fig,
            axes=axes[i][0],
        )
        unfolded.plot_spectral_rigidity(title=title, ensembles=["goe"], fig=fig, axes=axes[i][1])
        unfolded.plot_level_variance(title=title, ensembles=["goe"], fig=fig, axes=axes[i][2])
        axes[i][0].legend().set_visible(False) if i != 0 else None
        axes[i][1].legend().set_visible(False) if i != 0 else None
    fig.subplots_adjust(top=0.95, bottom=0.05, left=0.08, right=0.95, hspace=0.383)
    fig.set_size_inches(w=8.5 * 1.5, h=11 * 1.5)
    fig.savefig(OUTDIR / "unfold_compare.png", dpi=300)
    plt.show()


def make_unfolding_detrend_compare_plots() -> None:
    import matplotlib.pyplot as plt

    from empyricalRMT.eigenvalues import Eigenvalues
    from empyricalRMT.signalproc.detrend import DetrendMethod, detrend
    from empyricalRMT.smoother import SmoothMethod

    eigs = Eigenvalues.generate(1000, kind="goe")
    # goe_unfolds = {
    #     method.name: eigs.detrend(method).unfold(smoother="gue") for method in DetrendMethod
    # }
    # goe_unfolds[""] = eigs.unfold(smoother="gue")
    poly_unfolds = {
        method.name: eigs.detrend(method).unfold(smoother="poly", degree=5)
        for method in DetrendMethod
    }
    poly_unfolds[""] = eigs.unfold(smoother="poly", degree=5)
    N = len(DetrendMethod) + 1
    fig, axes = plt.subplots(ncols=2, nrows=N)
    for i, (label, unfolded) in enumerate(poly_unfolds.items()):
        title = f"{label} De-trending"
        unfolded.plot_nnsd(
            title=title,
            brody=True,
            brody_fit="mle",
            ensembles=["goe", "poisson"],
            fig=fig,
            axes=axes[i][0],
        )
        unfolded.plot_spectral_rigidity(title=title, ensembles=["goe"], fig=fig, axes=axes[i][1])
        unfolded.plot_level_variance(title=title, ensembles=["goe"], fig=fig, axes=axes[i][2])
        axes[i][0].legend().set_visible(False) if i != 0 else None
        axes[i][1].legend().set_visible(False) if i != 0 else None
    fig.subplots_adjust(top=0.95, bottom=0.05, left=0.08, right=0.95, hspace=0.383)
    fig.set_size_inches(w=8.5 * 1.5, h=11 * 1.5)
    # fig.savefig(OUTDIR / "unfold_compare_detrended.png", dpi=300)
    plt.show()


def make_single_plots() -> None:
    from empyricalRMT._types import MatrixKind
    from empyricalRMT.eigenvalues import Eigenvalues
    from empyricalRMT.plot import PlotMode
    from empyricalRMT.smoother import SmoothMethod

    ensembles = ["poisson", "goe"]  # theoretically expected curves to plot
    eigs = Eigenvalues.generate(matsize=5000, kind=MatrixKind.GOE)
    unfolded = eigs.unfold(smoother=SmoothMethod.GOE)
    unfolded.plot_nnsd(
        ensembles=ensembles, mode=PlotMode.Save, outfile=OUTDIR / "nnsd.png"
    )  # nearest neighbours spacings
    unfolded.plot_nnnsd(
        ensembles=["goe"], mode=PlotMode.Save, outfile=OUTDIR / "nnnsd.png"
    )  # next-nearest neighbours spacings
    unfolded.plot_spectral_rigidity(
        ensembles=ensembles, mode=PlotMode.Save, outfile=OUTDIR / "rigidity.png"
    )
    unfolded.plot_level_variance(
        ensembles=ensembles, mode=PlotMode.Save, outfile=OUTDIR / "variance.png"
    )


def make_unfolding_fit_plot() -> None:
    import matplotlib.pyplot as plt

    from empyricalRMT.eigenvalues import Eigenvalues
    from empyricalRMT.smoother import SmoothMethod

    # generate time series data
    T = np.random.standard_normal([1000, 250])
    eigs = Eigenvalues.from_time_series(T, trim_zeros=False)
    exp_unfolded = eigs.unfold(smoother=SmoothMethod.Exponential)
    poly_unfolded = eigs.unfold(smoother=SmoothMethod.Polynomial, degree=5)
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
    exp_unfolded.plot_fit(fig=fig, axes=ax1, title="Exponential Unfolding")
    poly_unfolded.plot_fit(fig=fig, axes=ax2, title="Polynomial Degree 5 Unfolding")
    fig.savefig(OUTDIR / "unfoldfit.png")


def time_rigidity() -> None:
    from timeit import repeat

    import numpy as np

    from empyricalRMT.eigenvalues import Eigenvalues

    unfolded = Eigenvalues.generate(5000, kind="goe").unfold(smoother="goe")
    L = np.arange(2, 50, 1, dtype=np.float64)
    results = repeat(
        "unfolded.spectral_rigidity(L)", number=10, globals=dict(unfolded=unfolded, L=L), repeat=10
    )
    print(
        f"Mean: {np.mean(results):0.2f}s. Range: [{np.min(results):0.2f}, {np.max(results):0.2f}]"
    )


def time_levelvar() -> None:
    from timeit import repeat

    import numpy as np

    from empyricalRMT.eigenvalues import Eigenvalues

    unfolded = Eigenvalues.generate(5000, kind="goe").unfold(smoother="goe")
    L = np.arange(2, 20, 1, dtype=np.float64)
    results = repeat(
        "unfolded.level_variance(L)", number=10, globals=dict(unfolded=unfolded, L=L), repeat=10
    )
    print(
        f"Mean: {np.mean(results):0.2f}s. Range: [{np.min(results):0.2f}, {np.max(results):0.2f}]"
    )


if __name__ == "__main__":
    # make_observables_plot()
    # make_unfolding_compare_plots()
    # make_unfolding_detrend_compare_plots()
    make_unfolding_fit_plot()
    # make_single_plots()
    # time_rigidity()
    # time_levelvar()
