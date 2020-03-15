import matplotlib.pyplot as plt
import numpy as np
import pytest

from empyricalRMT.eigenvalues import Eigenvalues
from empyricalRMT.construct import generate_eigs
from empyricalRMT.plot import PlotMode, _handle_plot_mode, _configure_sbn_style


@pytest.mark.plot
def test_brody_fit() -> None:
    # test GOE eigs
    bw = 0.2
    # bw = "scott"
    mode: PlotMode = "test"
    ensembles = ["goe", "poisson"]
    for N in [100, 250, 500, 1000]:
        _configure_sbn_style()
        fig, axes = plt.subplots(2, 2)
        for i in range(4):
            eigs = generate_eigs(N)
            Eigenvalues(eigs).unfold(degree=7).plot_nnsd(
                brody=True,
                kde_bw=bw,
                title=f"GOE N={N}",
                ensembles=ensembles,
                mode="return",
                fig=fig,
                axes=axes.flat[i],
            )
        _handle_plot_mode(mode, fig, axes)

    # test time series
    _configure_sbn_style()
    fig, axes = plt.subplots(2, 2)
    for i in range(4):
        eigs = np.linalg.eigvalsh(np.corrcoef(np.random.standard_normal([1000, 250])))
        Eigenvalues(eigs).unfold(degree=7).plot_nnsd(
            brody=True,
            brody_fit="mle",
            mode=mode,
            title=f"t-series (untrimmed)(MLE)",
            ensembles=ensembles,
            kde_bw=bw,
            fig=fig,
            axes=axes.flat[i],
        )
    _handle_plot_mode(mode, fig, axes)

    _configure_sbn_style()
    fig, axes = plt.subplots(2, 2)
    for i in range(4):
        eigs = np.linalg.eigvalsh(np.corrcoef(np.random.standard_normal([1000, 250])))
        Eigenvalues(eigs).unfold(degree=7).plot_nnsd(
            brody=True,
            brody_fit="spacings",
            mode=mode,
            title=f"t-series (untrimmed)(spacings)",
            ensembles=ensembles,
            kde_bw=bw,
            fig=fig,
            axes=axes.flat[i],
        )
    _handle_plot_mode(mode, fig, axes)

    _configure_sbn_style()
    fig, axes = plt.subplots(2, 2)
    for i in range(4):
        eigs = np.linalg.eigvalsh(np.corrcoef(np.random.standard_normal([1000, 250])))
        eigs = eigs[eigs > 100 * np.abs(eigs.min())]
        Eigenvalues(eigs).unfold(degree=7).plot_nnsd(
            brody=True,
            brody_fit="spacings",
            mode=mode,
            title="t-series (trimmed)(spacings)",
            ensembles=ensembles,
            kde_bw=bw,
            fig=fig,
            axes=axes.flat[i],
        )
    _handle_plot_mode(mode, fig, axes)

    _configure_sbn_style()
    fig, axes = plt.subplots(2, 2)
    for i in range(4):
        eigs = np.linalg.eigvalsh(np.corrcoef(np.random.standard_normal([1000, 250])))
        eigs = eigs[eigs > 100 * np.abs(eigs.min())]
        Eigenvalues(eigs).unfold(degree=7).plot_nnsd(
            brody=True,
            brody_fit="mle",
            mode=mode,
            title="t-series (trimmed)(MLE)",
            ensembles=ensembles,
            kde_bw=bw,
            fig=fig,
            axes=axes.flat[i],
        )
    _handle_plot_mode(mode, fig, axes)

    if mode != "test":
        plt.show()
