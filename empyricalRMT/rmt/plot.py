import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sbn
import warnings

from colorama import Fore, Style
from matplotlib.pyplot import Figure, Axes
from numpy import ndarray
from pathlib import Path
from scipy.integrate import quad
from scipy.special import sici
from statsmodels.nonparametric.kde import KDEUnivariate as KDE
from typing import Any, List, Optional, Tuple, Union
from typing_extensions import Literal
from warnings import warn

from empyricalRMT.rmt.ensemble import Poisson, GOE
from empyricalRMT.rmt.observables.step import _step_function_fast
from empyricalRMT.utils import make_parent_directories

PlotResult = Optional[Tuple[Figure, Axes]]
PlotMode = Union[
    Literal["block"], Literal["noblock"], Literal["save"], Literal["return"]
]

RESET = Style.RESET_ALL
PLOTTING_READY = False


def _raw_eig_dist(
    eigs: ndarray,
    bins: int = 50,
    title: str = "Raw Eigenvalue Distribution",
    kde: bool = True,
    mode: PlotMode = "block",
    outfile: Path = None,
    fig: Figure = None,
    axes: Axes = None,
) -> PlotResult:
    """Plot a histogram of the raw eigenvalues.

    Parameters
    ----------
    eigs: ndarray
        The eigenvalues to plot.
    bins: int
        the number of (equal-sized) bins to display and use for the histogram
    kde: boolean
        If False (default), do not display a kernel density estimate. If true, use
        [statsmodels.nonparametric.kde.KDEUnivariate](https://www.statsmodels.org/stable/generated/statsmodels.nonparametric.kde.KDEUnivariate.html#statsmodels.nonparametric.kde.KDEUnivariate)
        with arguments {kernel="gau", bw="scott", cut=0} to compute and display the kde
    title: string
        The plot title string
    mode: "block" (default) | "noblock" | "save" | "return"
        If "block", call plot.plot() and display plot in a blocking fashion.
        If "noblock", attempt to generate plot in nonblocking fashion.
        If "save", save plot to pathlib Path specified in `outfile` argument
        If "return", return (fig, axes), the matplotlib figure and axes object for modification.
    outfile: Path
        If mode="save", save generated plot to Path specified in `outfile` argument.
        Intermediate directories will be created if needed.

    Returns
    -------
    (fig, axes): (Figure, Axes)
        The handles to the matplotlib objects, only if `mode` is "return".
    """
    fig, axes = _setup_plotting(fig, axes)
    sbn.distplot(
        eigs,
        norm_hist=True,
        bins=bins,  # doane
        kde=False,
        label="Raw Eigenvalue Distribution",
        axlabel="Eigenvalue",
        color="black",
        ax=axes,
    )
    if kde:
        grid = np.linspace(eigs.min(), eigs.max(), 10000)
        _kde_plot(eigs, grid, axes)

    axes.set(title=title, ylabel="Density")
    axes.legend().set_visible(True)
    return _handle_plot_mode(mode, fig, axes, outfile)


def _step_function(
    eigs: ndarray,
    gridsize: int = 100000,
    title: str = "Eigenvalue Step Function",
    mode: PlotMode = "block",
    outfile: Path = None,
    fig: Figure = None,
    axes: Axes = None,
) -> PlotResult:
    """Compute the step function vaues over a grid, and plot the resulting curve.

    Parameters
    ----------
    eigs: ndarray
        The eigenvalues to plot.
    gridsize: int
        The number of points to evaluate the step function over. The grid will be
        generated as np.linspace(eigs.min(), eigs.max(), gridsize).
    title: string
        The plot title string
    mode: "block" (default) | "noblock" | "save" | "return"
        If "block", call plot.plot() and display plot in a blocking fashion.
        If "noblock", attempt to generate plot in nonblocking fashion.
        If "save", save plot to pathlib Path specified in `outfile` argument
        If "return", return (fig, axes), the matplotlib figure and axes object for modification.
    outfile: Path
        If mode="save", save generated plot to Path specified in `outfile` argument.
        Intermediate directories will be created if needed.

    Returns
    -------
    (fig, axes): (Figure, Axes)
        The handles to the matplotlib objects, only if `mode` is "return".
    """
    fig, axes = _setup_plotting(fig, axes)
    grid = np.linspace(eigs.min(), eigs.max(), gridsize)
    steps = _step_function_fast(eigs, grid)
    df = pd.DataFrame({"Cumulative Value": steps, "Raw eigenvalues λ": grid})
    axes = sbn.lineplot(data=df, x="Raw eigenvalues λ", y="Cumulative Value", ax=axes)
    axes.set(title=title)
    return _handle_plot_mode(mode, fig, axes, outfile)


def _raw_eig_sorted(
    eigs: ndarray,
    title: str = "Raw Eigenvalues",
    mode: PlotMode = "block",
    outfile: Path = None,
    kind: Union[Literal["scatter"], Literal["line"]] = "scatter",
    fig: Figure = None,
    axes: Axes = None,
) -> PlotResult:
    """Plot a curve or scatterplot of the raw eigenvalues.

    Parameters
    ----------
    eigs: ndarray
        The eigenvalues to plot.
    title: string
        The plot title string
    mode: "block" (default) | "noblock" | "save" | "return"
        If "block", call plot.plot() and display plot in a blocking fashion.
        If "noblock", attempt to generate plot in nonblocking fashion.
        If "save", save plot to pathlib Path specified in `outfile` argument
        If "return", return (fig, axes), the matplotlib figure and axes object for modification.
    outfile: Path
        If mode="save", save generated plot to Path specified in `outfile` argument.
        Intermediate directories will be created if needed.
    kind: "scatter" (default) | "line"
        Whether to use a scatterplot or line plot.

    Returns
    -------
    (fig, axes): (Figure, Axes)
        The handles to the matplotlib objects, only if `mode` is "return".
    """
    fig, axes = _setup_plotting(fig, axes)
    if kind == "scatter":
        sbn.scatterplot(data=eigs, ax=axes)
    elif kind == "line":
        sbn.lineplot(data=eigs, ax=axes)
    else:
        raise ValueError("Invalid plot kind. Must be 'scatter' or 'line'.")
    axes.set(title=title, xlabel="Eigenvalue index", ylabel="Eigenvalue")
    return _handle_plot_mode(mode, fig, axes, outfile)


def _unfolded_dist(
    unfolded: ndarray,
    bins: int = 50,
    kde: bool = True,
    title: str = "Unfolded Eigenvalues",
    mode: PlotMode = "block",
    outfile: Path = None,
    fig: Figure = None,
    axes: Axes = None,
) -> PlotResult:
    """Plot a histogram of the unfolded eigenvalues.

    Parameters
    ----------
    unfolded: ndarray
        The unfolded eigenvalues to plot.
    bins: int
        the number of (equal-sized) bins to display and use for the histogram
    kde: boolean
        If False (default), do not display a kernel density estimate. If true, use
        [statsmodels.nonparametric.kde.KDEUnivariate](https://www.statsmodels.org/stable/generated/statsmodels.nonparametric.kde.KDEUnivariate.html#statsmodels.nonparametric.kde.KDEUnivariate)
        with arguments {kernel="gau", bw="scott", cut=0} to compute and display the kde
    title: string
        The plot title string
    mode: "block" (default) | "noblock" | "save" | "return"
        If "block", call plot.plot() and display plot in a blocking fashion.
        If "noblock", attempt to generate plot in nonblocking fashion.
        If "save", save plot to pathlib Path specified in `outfile` argument
        If "return", return (fig, axes), the matplotlib figure and axes object for modification.
    outfile: Path
        If mode="save", save generated plot to Path specified in `outfile` argument.
        Intermediate directories will be created if needed.

    Returns
    -------
    (fig, axes): (Figure, Axes)
        The handles to the matplotlib objects, only if `mode` is "return".
    """
    fig, axes = _setup_plotting(fig, axes)
    sbn.distplot(
        unfolded,
        norm_hist=True,
        bins=bins,  # doane
        kde=False,
        label="Unfolded Eigenvalue Distribution",
        axlabel="Eigenvalue",
        color="black",
        ax=axes,
    )
    if kde:
        grid = np.linspace(unfolded.min(), unfolded.max(), 10000)
        _kde_plot(unfolded, grid, axes)

    axes.set(title=title, ylabel="Density")
    axes.legend().set_visible(True)
    return _handle_plot_mode(mode, fig, axes, outfile)


def _unfolded_fit(
    unfolded: ndarray,
    title: str = "Unfolding Fit",
    mode: PlotMode = "block",
    outfile: Path = None,
    fig: Figure = None,
    axes: Axes = None,
) -> PlotResult:
    """Plot the unfolding fit against the step function.

    Parameters
    ----------
    unfolded: ndarray
        The unfolded eigenvalues to plot.
    title: string
        The plot title string
    mode: "block" (default) | "noblock" | "save" | "return"
        If "block", call plot.plot() and display plot in a blocking fashion.
        If "noblock", attempt to generate plot in nonblocking fashion.
        If "save", save plot to pathlib Path specified in `outfile` argument
        If "return", return (fig, axes), the matplotlib figure and axes object for modification.
    outfile: Path
        If mode="save", save generated plot to Path specified in `outfile` argument.
        Intermediate directories will be created if needed.
    kind: "scatter" (default) | "line"
        Whether to use a scatterplot or line plot.

    Returns
    -------
    (fig, axes): (Figure, Axes)
        The handles to the matplotlib objects, only if `mode` is "return".
    """
    fig, axes = _setup_plotting(fig, axes)
    N = len(unfolded)
    df = pd.DataFrame({"Step Function": np.arange(1, N + 1), "Unfolded λ": unfolded})
    axes = sbn.lineplot(data=df, ax=axes)
    axes.set(title=title)
    return _handle_plot_mode(mode, fig, axes, outfile)


# this essentially plots the nearest-neighbors spacing distribution
def _spacings(
    unfolded: ndarray,
    bins: int = 50,
    kde: bool = True,
    trim: float = 0.0,
    trim_kde: bool = False,
    title: str = "Unfolded Spacing Distribution",
    mode: PlotMode = "block",
    outfile: Path = None,
    ensembles: List[str] = ["poisson", "goe", "gue", "gse"],
    fig: Figure = None,
    axes: Axes = None,
) -> PlotResult:
    """Plots a histogram of the Nearest-Neighbors Spacing Distribution

    Parameters
    ----------
    unfolded: ndarray
        the unfolded eigenvalues
    bins: int
        the number of (equal-sized) bins to display and use for the histogram
    kde: boolean
        If False (default), do not display a kernel density estimate. If true, use
        [statsmodels.nonparametric.kde.KDEUnivariate](https://www.statsmodels.org/stable/generated/statsmodels.nonparametric.kde.KDEUnivariate.html#statsmodels.nonparametric.kde.KDEUnivariate)
        with arguments {kernel="gau", bw="scott", cut=0} to compute and display the kde
    trim: float
        If True, only use spacings <= `trim` for computing the KDE and plotting.
        Useful for when large spacings distort the histogram.
    trim_kde: bool
        If True, fit the KDE using only spacings <= `trim`. Otherwise, fit the
        KDE using all available spacings.
    title: string
        The plot title string
    mode: "block" | "noblock" | "save" | "return"
        If "block", call plot.plot() and display plot in a blocking fashion.
        If "noblock", attempt to generate plot in nonblocking fashion.
        If "save", save plot to pathlib Path specified in `outfile` argument
        If "return", return (fig, axes), the matplotlib figure and axes object for modification.
    outfile: Path
        If mode="save", save generated plot to Path specified in `outfile` argument.
        Intermediate directories will be created if needed.
    ensembles: ["poisson", "goe", "gue", "gse"]
        Which ensembles to display the expected NNSD curves for.

    Returns
    -------
    (fig, axes): (Figure, Axes)
        The handles to the matplotlib objects, only if `mode` is "return".
    """
    fig, axes = _setup_plotting(fig, axes)
    _spacings = np.sort(unfolded[1:] - unfolded[:-1])
    all_spacings = np.copy(_spacings)
    if trim > 0.0:
        _spacings = _spacings[_spacings <= trim]
    # Generate expected distributions for classical ensembles
    p = np.pi
    s = np.linspace(_spacings.min(), _spacings.max(), 10000)
    axes = sbn.distplot(
        _spacings,
        norm_hist=True,
        bins=bins,  # doane
        kde=False,
        label="Empirical Spacing Distribution",
        axlabel="spacing (s)",
        color="black",
        ax=axes,
    )

    # fmt: off
    if "poisson" in ensembles:
        poisson = np.exp(-s)
        poisson = axes.plot(s, poisson, label="Poisson")
        plt.setp(poisson, color="#08FD4F")
    if "goe" in ensembles:
        goe = ((p * s) / 2) * np.exp(-(p / 4) * s * s)
        goe = axes.plot(s, goe, label="Gaussian Orthogonal")
        plt.setp(goe, color="#FD8208")
    if "gue" in ensembles:
        gue = (32 / p**2) * (s * s) * np.exp(-(4 * s * s) / p)
        gue = axes.plot(s, gue, label="Gaussian Unitary")
        plt.setp(gue, color="#0066FF")
    if "gse" in ensembles:
        gse = (2**18 / (3**6 * p**3)) * (s**4) * np.exp(-((64 / (9 * p)) * (s * s)))
        gse = axes.plot(s, gse, label="Gaussian Symplectic")
        plt.setp(gse, color="#EA00FF")
    # fmt: on

    if kde is True:
        if trim_kde:
            _kde_plot(_spacings, s, axes)
        else:
            _kde_plot(all_spacings, s, axes)

    # adjusting the right bounds can be necessary when / if there are
    # many large eigenvalue spacings
    # axes.set_xlim(left=0, right=np.percentile(_spacings, 99))
    axes.set_ylim(top=1.5, bottom=0)
    axes.set_xlim(left=0, right=2.5)
    axes.set(title=title, ylabel="Density p(s)")
    axes.legend().set_visible(True)

    return _handle_plot_mode(mode, fig, axes, outfile)


def _next_spacings(
    unfolded: ndarray,
    bins: int = 50,
    kde: bool = True,
    trim: float = 0.0,
    trim_kde: bool = False,
    title: str = "next Nearest-Neigbors Spacing Distribution",
    mode: PlotMode = "block",
    outfile: Path = None,
    ensembles: List[str] = ["goe", "poisson"],
    fig: Figure = None,
    axes: Axes = None,
) -> PlotResult:
    """Plots a histogram of the next Nearest-Neighbors Spacing Distribution

    Parameters
    ----------
    unfolded: ndarray
        the unfolded eigenvalues
    bins: int
        the number of (equal-sized) bins to display and use for the histogram
    kde: boolean
        If False (default), do not display a kernel density estimate. If true, use
        [statsmodels.nonparametric.kde.KDEUnivariate](https://www.statsmodels.org/stable/generated/statsmodels.nonparametric.kde.KDEUnivariate.html#statsmodels.nonparametric.kde.KDEUnivariate)
        with arguments {kernel="gau", bw="scott", cut=0} to compute and display the kde
    trim: float
        If True, only use spacings <= `trim` for computing the KDE and plotting.
        Useful for when large spacings distort the histogram.
    trim_kde: bool
        If True, fit the KDE using only spacings <= `trim`. Otherwise, fit the
        KDE using all available spacings.
    title: string
        The plot title string
    mode: "block" | "noblock" | "save" | "return"
        If "block", call plot.plot() and display plot in a blocking fashion.
        If "noblock", attempt to generate plot in nonblocking fashion.
        If "save", save plot to pathlib Path specified in `outfile` argument
        If "return", return (fig, axes), the matplotlib figure and axes object for modification.
    outfile: Path
        If mode="save", save generated plot to Path specified in `outfile` argument.
        Intermediate directories will be created if needed.
    ensembles: List["poisson", "goe"]
        Which ensembles to display the expected next-NNSD curves for.

    Returns
    -------
    (fig, axes): (Figure, Axes)
        The handles to the matplotlib objects, only if `mode` is "return".
    """
    fig, axes = _setup_plotting(fig, axes)
    _spacings = np.sort((unfolded[2:] - unfolded[:-2]) / 2)
    all_spacings = np.copy(_spacings)
    if trim > 0.0:
        _spacings = _spacings[_spacings <= trim]
    # Generate expected distributions for classical ensembles
    s_min, s_max = _spacings.min(), _spacings.max()
    s = np.linspace(s_min, s_max, 10000)

    axes = sbn.distplot(
        _spacings,
        norm_hist=True,
        bins=bins,  # doane
        kde=False,
        label="next NNSD",
        axlabel="spacing (s_2)",
        color="black",
        ax=axes,
    )

    if kde is True:
        if trim_kde:
            _kde_plot(_spacings, s, axes)
        else:
            _kde_plot(all_spacings, s, axes)

    if "goe" in ensembles:
        goe = GOE.nnnsd(spacings=s)
        goe = axes.plot(s, goe, label="Gaussian Orthogonal")
        plt.setp(goe, color="#FD8208")

    if "poisson" in ensembles:
        poisson = Poisson.nnnsd(spacings=s)
        poisson = axes.plot(s, poisson, label="Poisson")
        plt.setp(poisson, color="#08FD4F")

    # adjusting the right bounds can be necessary when / if there are
    # many large eigenvalue spacings
    axes.set_ylim(top=2.0, bottom=0)
    axes.set_xlim(left=0, right=2.5)
    axes.set(title=title, ylabel="Density p(s)")
    axes.legend().set_visible(True)

    return _handle_plot_mode(mode, fig, axes, outfile)


def _spectral_rigidity(
    unfolded: Optional[ndarray],
    data: pd.DataFrame,
    title: str = "Spectral Rigidity",
    mode: PlotMode = "block",
    outfile: Path = None,
    ensembles: List[str] = ["poisson", "goe", "gue", "gse"],
    fig: Figure = None,
    axes: Axes = None,
) -> PlotResult:
    """Plot the computed spectral rigidity against the various expected spectral
    rigidity curves for the classical ensembles.

    Parameters
    ----------
    unfolded: ndarray
        The unfolded eigenvalues to plot.
    data: DataFrame
        `data` argument is pd.DataFrame({"L": L_vals, "delta": delta3})
        TODO: fix this
    title: string
        The plot title string
    mode: "block" (default) | "noblock" | "save" | "return"
        If "block", call plot.plot() and display plot in a blocking fashion.
        If "noblock", attempt to generate plot in nonblocking fashion.
        If "save", save plot to pathlib Path specified in `outfile` argument
        If "return", return (fig, axes), the matplotlib figure and axes object for modification.
    outfile: Path
        If mode="save", save generated plot to Path specified in `outfile` argument.
        Intermediate directories will be created if needed.
    ensembles: ["poisson", "goe", "gue", "gse"]
        Which ensembles to display the expected spectral rigidity curves for comparison against.

    Returns
    -------
    (fig, axes): (Figure, Axes)
        The handles to the matplotlib objects, only if `mode` is "return".
    """
    fig, axes = _setup_plotting(fig, axes)
    df = pd.DataFrame(data, columns=["L", "delta"])
    # sbn.relplot(x="L", y="delta", data=df, ax=axes)
    sbn.scatterplot(x="L", y="delta", data=df, ax=axes)
    ensembles = set(ensembles)  # type: ignore

    # _, right = plt.xlim()
    _, right = axes.get_xlim()

    L = df["L"]
    p, y = np.pi, np.euler_gamma

    # see pg 290 of Mehta (2004) for definition of s
    s = L / np.mean(unfolded[1:] - unfolded[:-1]) if unfolded is not None else L

    if "poisson" in ensembles:
        poisson = L / 15 / 2
        poisson = axes.plot(L, poisson, label="Poisson")
        plt.setp(poisson, color="#08FD4F")
    if "goe" in ensembles:
        goe = (1 / (p ** 2)) * (np.log(2 * p * s) + y - 5 / 4 - (p ** 2) / 8)
        goe = axes.plot(L, goe, label="Gaussian Orthogonal")
        plt.setp(goe, color="#FD8208")
    if "gue" in ensembles:
        gue = (1 / (2 * (p ** 2))) * (np.log(2 * p * s) + y - 5 / 4)
        gue = axes.plot(L, gue, label="Gaussian Unitary")
        plt.setp(gue, color="#0066FF")
    if "gse" in ensembles:
        gse = (1 / (4 * (p ** 2))) * (np.log(4 * p * s) + y - 5 / 4 + (p ** 2) / 8)
        gse = axes.plot(L, gse, label="Gaussian Symplectic")
        plt.setp(gse, color="#EA00FF")

    axes.set(title=title, xlabel="L", ylabel="∆3(L)")
    axes.legend().set_visible(True)
    return _handle_plot_mode(mode, fig, axes, outfile)


def _level_number_variance(
    unfolded: ndarray,
    data: pd.DataFrame,
    title: str = "Level Number Variance",
    mode: PlotMode = "block",
    outfile: Path = None,
    ensembles: List[str] = ["poisson", "goe", "gue", "gse"],
    fig: Figure = None,
    axes: Axes = None,
) -> PlotResult:
    """Plot the computed level number variance against the various expected number
    level variance curves for the classical ensembles.

    Parameters
    ----------
    unfolded: ndarray
        The unfolded eigenvalues to plot.
    data: DataFrame
        `data` argument is pd.DataFrame({"L": L_vals, "sigma": sigma}), where sigma
        are the values computed from observables.levelvariance.level_number_variance
    title: string
        The plot title string
    mode: "block" (default) | "noblock" | "save" | "return"
        If "block", call plot.plot() and display plot in a blocking fashion.
        If "noblock", attempt to generate plot in nonblocking fashion.
        If "save", save plot to pathlib Path specified in `outfile` argument
        If "return", return (fig, axes), the matplotlib figure and axes object for modification.
    outfile: Path
        If mode="save", save generated plot to Path specified in `outfile` argument.
        Intermediate directories will be created if needed.
    ensembles: ["poisson", "goe", "gue", "gse"]
        Which ensembles to display the expected number level variance curves for comparison against.

    Returns
    -------
    (fig, axes): (Figure, Axes)
        The handles to the matplotlib objects, only if `mode` is "return".
    """
    fig, axes = _setup_plotting(fig, axes)
    df = pd.DataFrame(data, columns=["L", "sigma"])
    # sbn.relplot(x="L", y="sigma", data=df, ax=axes)
    sbn.scatterplot(x="L", y="sigma", data=df, ax=axes)
    ensembles = set(ensembles)  # type: ignore

    # _, right = plt.xlim()
    _, right = axes.get_xlim()

    L = df["L"]
    p, y = np.pi, np.euler_gamma
    # s = L / np.mean(unfolded[1:] - unfolded[:-1])
    s = L

    def exact(x: float) -> float:
        def f1(r: float) -> Any:
            return (np.sin(r) / r) ** 2

        # re-arranging the formula for sici from
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.sici.html
        # to match Mehta (2004) p595, A.38, we get:
        int_2 = y + np.log(2 * p * x) - sici(2 * p * x)[1]
        # sici(x) returns (Sine integral from 0 to x, gamma + log(x) + Cosine integral from 0 to x)
        int_3 = (sici(np.inf)[0] - sici(p * x)[0]) ** 2
        t1 = 4 * x / p
        t2 = 2 / p ** 2
        t3 = t2 / 2
        res = (
            t1 * quad(f1, p * x, np.inf, limit=100)[0] + t2 * int_2 - 0.25 + t3 * int_3
        )
        return float(res)

    if "poisson" in ensembles:
        poisson = L / 2  # waste of time, too large very often
        poisson = axes.plot(L, poisson, label="Poisson")
        plt.setp(poisson, color="#08FD4F")
    if "goe" in ensembles:
        goe = np.zeros(s.shape)
        with warnings.catch_warnings():  # ignore integration, divide-by-zero warnings
            warnings.simplefilter("ignore")
            for i, s_val in enumerate(s):
                if L[i] < 10:
                    goe[i] = exact(s_val)
                else:
                    goe[i] = (2 / (p ** 2)) * (
                        np.log(2 * p * s_val) + y + 1 - (p ** 2) / 8
                    )
        goe = axes.plot(L, goe, label="Gaussian Orthogonal")
        plt.setp(goe, color="#FD8208")
    if "gue" in ensembles:
        gue = (1 / (p ** 2)) * (np.log(2 * p * s) + y + 1)
        gue = axes.plot(L, gue, label="Gaussian Unitary")
        plt.setp(gue, color="#0066FF")
    if "gse" in ensembles:
        gse = (1 / (2 * (p ** 2))) * (np.log(4 * p * s) + y + 1 + (p ** 2) / 8)
        gse = axes.plot(L, gse, label="Gaussian Symplectic")
        plt.setp(gse, color="#EA00FF")

    axes.set(
        title=f"Level Number Variance - {title} unfolding",
        xlabel="L",
        ylabel="Sigma^2(L)",
    )
    axes.legend().set_visible(True)
    return _handle_plot_mode(mode, fig, axes, outfile)


def _observables(
    unfolded: ndarray,
    rigidity_df: pd.DataFrame,
    levelvar_df: pd.DataFrame,
    ensembles: List[str] = ["goe", "poisson"],
    suptitle: str = "Spectral Observables",
    mode: PlotMode = "block",
    outfile: Path = None,
) -> PlotResult:
    _setup_plotting(init=True)
    fig, axes = plt.subplots(2, 2, sharex="none", sharey="none")
    fig.set_size_inches(fig.get_size_inches()*2)
    fig.suptitle(suptitle)
    _unfolded_fit(unfolded=unfolded, fig=fig, axes=axes[0, 0], mode="noblock")
    _spacings(unfolded, fig=fig, axes=axes[0, 1], mode="noblock", ensembles=ensembles)
    _spectral_rigidity(
        unfolded, data=rigidity_df, fig=fig, axes=axes[1, 0], mode="noblock", ensembles=ensembles
    )
    _level_number_variance(
        unfolded, data=levelvar_df, fig=fig, axes=axes[1, 1], mode="noblock", ensembles=ensembles
    )
    return _handle_plot_mode(mode, fig, axes, outfile)


def _setup_plotting(
    fig: Figure = None, axes: Axes = None, init: bool = False
) -> Tuple[Optional[Figure], Optional[Axes]]:
    global PLOTTING_READY
    if not PLOTTING_READY:
        PALETTE = sbn.color_palette("dark").copy()
        PALETTE.insert(0, (0.0, 0.0, 0.0))
        sbn.set()
        sbn.set_palette(PALETTE)
        PLOTTING_READY = True
    if init:
        return None, None
    if fig is None or axes is None:
        fig, axes = plt.subplots()
        return fig, axes
    else:
        return fig, axes


def _kde_plot(values: ndarray, grid: ndarray, axes: Axes) -> None:
    """
    calculate KDE for observed spacings
    we are doing this manually because we want to ensure consistency of the KDE
    calculation and remove Seaborn control over the process, while also avoiding
    inconsistent behaviours like https://github.com/mwaskom/seaborn/issues/938 and
    https://github.com/mwaskom/seaborn/issues/796

    Parameters
    ----------
    values: ndarray
        the values used to compute (fit) the kernel density estimate
    grid: ndarray
        the grid of values over which to evaluate the computed KDE curve
    axes: pyplot.Axes
        the current axes object to be modified
    """
    kde = KDE(values)
    kde.fit(kernel="gau", bw="scott", cut=0)
    evaluated = np.empty_like(grid)
    for i, _ in enumerate(evaluated):
        evaluated[i] = kde.evaluate(grid[i])
    kde_curve = axes.plot(grid, evaluated, label="Kernel Density Estimate")
    plt.setp(kde_curve, color="black")


def _handle_plot_mode(
    mode: str, fig: Figure, axes: Axes, outfile: Path = None
) -> Union[Optional[PlotResult], Axes]:
    if mode == "block":
        plt.show(block=True)
    elif mode == "noblock":
        plt.show(block=False)
        plt.pause(0.001)
    elif mode == "save":
        if outfile is None:
            raise ValueError("Path not specified for `outfile`.")
        try:
            outfile = Path(outfile)
        except BaseException as e:
            raise ValueError("Cannot interpret outfile path.") from e
        make_parent_directories(outfile)
        print(f"Saving figure to {outfile}")
        fig.savefig(outfile)
    elif mode == "return":
        return fig, axes
    else:
        raise ValueError("Invalid plotting mode.")
    return None


def _validate_bin_sizes(vals: ndarray, bins: int) -> None:
    vals = np.sort(vals)
    L = len(vals)
    bin_ends = np.linspace(vals[0], vals[-1], bins, endpoint=True)[1:]
    counts = np.empty(bin_ends.shape)
    for i, endpoint in enumerate(bin_ends):
        if i == 0:
            counts[i] = len(vals[vals < endpoint])
        else:
            counts[i] = len(vals[vals < endpoint]) - np.sum(counts[:i])
        if counts[i] / L > 0.4:
            print(
                f"{Fore.YELLOW}Overfull bin {i}: {Fore.RED}{np.round(counts[i]/L, 2)}% {RESET} of values."
            )
            warn("Distribution likely too skewed to generate interpretable histogram")
