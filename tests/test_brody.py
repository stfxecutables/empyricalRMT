import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes

from empyricalRMT.construct import generate_eigs
from empyricalRMT.eigenvalues import Eigenvalues
from empyricalRMT.plot import PlotMode, _handle_plot_mode


@pytest.mark.fast
def test_brody_fit() -> None:
    for N in [100, 250, 500, 1000]:
        unfolded = Eigenvalues(generate_eigs(N)).unfold(degree=7)
        # test fitting via max spacing
        res = unfolded.fit_brody(method="spacing")
        spacings = res["spacings"]
        if not np.all(np.isfinite(spacings)):
            raise ValueError("Return spacings contains infinities.")
        ecdf = res["ecdf"]
        if np.sum(ecdf < 0) > 0 or np.sum(ecdf > 1):
            raise ValueError("Invalid values in empirical cdf.")
        brody_cdf = res["brody_cdf"]
        if np.sum(brody_cdf < 0) > 0 or np.sum(brody_cdf > 1):
            raise ValueError("Invalid values in brody cdf.")

        # test fitting via mle
        res = unfolded.fit_brody(method="mle")
        spacings = res["spacings"]
        if not np.all(np.isfinite(spacings)):
            raise ValueError("Return spacings contains infinities.")
        ecdf = res["ecdf"]
        if np.sum(ecdf < 0) > 0 or np.sum(ecdf > 1):
            raise ValueError("Invalid values in empirical cdf.")
        brody_cdf = res["brody_cdf"]
        if np.sum(brody_cdf < 0) > 0 or np.sum(brody_cdf > 1):
            raise ValueError("Invalid values in brody cdf.")


test_brody_fit()


@pytest.mark.plot
def test_brody_plot() -> None:
    # test GOE eigs
    bw = 0.2
    # bw = "scott"
    mode: PlotMode = PlotMode.Test
    ensembles = ["goe", "poisson"]
    for N in [100, 250, 500, 1000]:
        axes: Axes
        fig, axes = plt.subplots(2, 2)  # type: ignore
        for i in range(4):
            eigs = generate_eigs(N)
            Eigenvalues(eigs).unfold(degree=7).plot_nnsd(
                brody=True,
                kde_bw=bw,
                title=f"GOE N={N}",
                ensembles=ensembles,
                mode=PlotMode.Return,
                fig=fig,
                axes=axes.flat[i],  # type: ignore
            )
        _handle_plot_mode(mode, fig, axes)

    # test time series
    fig, axes = plt.subplots(2, 2)  # type: ignore
    for i in range(4):
        eigs = np.linalg.eigvalsh(np.corrcoef(np.random.standard_normal([1000, 250])))
        Eigenvalues(eigs).unfold(degree=7).plot_nnsd(
            brody=True,
            brody_fit="mle",
            mode=mode,
            title="t-series (untrimmed)(MLE)",
            ensembles=ensembles,
            kde_bw=bw,
            fig=fig,
            axes=axes.flat[i],  # type: ignore
        )
    _handle_plot_mode(mode, fig, axes)

    fig, axes = plt.subplots(2, 2)  # type: ignore
    for i in range(4):
        eigs = np.linalg.eigvalsh(np.corrcoef(np.random.standard_normal([1000, 250])))
        Eigenvalues(eigs).unfold(degree=7).plot_nnsd(
            brody=True,
            brody_fit="spacings",
            mode=mode,
            title="t-series (untrimmed)(spacings)",
            ensembles=ensembles,
            kde_bw=bw,
            fig=fig,
            axes=axes.flat[i],  # type: ignore
        )
    _handle_plot_mode(mode, fig, axes)

    fig, axes = plt.subplots(2, 2)  # type: ignore
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
            axes=axes.flat[i],  # type: ignore
        )
    _handle_plot_mode(mode, fig, axes)

    fig, axes = plt.subplots(2, 2)  # type: ignore
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
            axes=axes.flat[i],  # type: ignore
        )
    _handle_plot_mode(mode, fig, axes)

    if mode != "test":
        plt.show()
