import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sbn
import warnings

from colorama import Fore, Style
from matplotlib.pyplot import Figure, Axes
from matplotlib.patches import Rectangle
from matplotlib.legend_handler import HandlerBase
from numpy import ndarray
from pathlib import Path
from scipy.integrate import quad
from scipy.special import sici
from scipy.stats import trim_mean
from statsmodels.nonparametric.kde import KDEUnivariate as KDE
from typing import Any, List, Optional, Tuple, Union
from typing_extensions import Literal
from warnings import warn

from empyricalRMT.brody import brody_dist, brody_fit_evaluate, fit_brody
from empyricalRMT.ensemble import Poisson, GOE
from empyricalRMT.observables.step import _step_function_fast
from empyricalRMT.utils import make_parent_directories

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from empyricalRMT.trim import TrimIter  # noqa: F401

PlotResult = Optional[Tuple[Figure, Axes]]
PlotMode = Union[
    Literal["block"],
    Literal["noblock"],
    Literal["save"],
    Literal["return"],
    Literal["test"],
]

RESET = Style.RESET_ALL
PLOTTING_READY = False


def _raw_eig_dist(
    eigs: ndarray,
    bins: int = 50,
    kde: bool = True,
    title: str = "Raw Eigenvalue Distribution",
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
        with arguments {kernel="gau", bw="scott", cut=0} to compute and display
        the kde

    title: string
        The plot title string

    mode: "block" (default) | "noblock" | "save" | "return"
        If "block", call plot.plot() and display plot in a blocking fashion.
        If "noblock", attempt to generate plot in nonblocking fashion.
        If "save", save plot to pathlib Path specified in `outfile` argument
        If "return", return (fig, axes), the matplotlib figure and axes object
        for modification.

    outfile: Path
        If mode="save", save generated plot to Path specified in `outfile` argument.
        Intermediate directories will be created if needed.

    fig: Figure
        If provided with `axes`, configure plotting with the provided `fig`
        object instead of creating a new figure. Useful for creating subplots.

    axes: Axes
        If provided with `fig`, plot to the provided `axes` object. Useful for
        creating subplots.


    Returns
    -------
    (fig, axes): (Figure, Axes)
        The handles to the matplotlib objects, only if `mode` is "return".
    """
    _configure_sbn_style()
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
        If "return", return (fig, axes), the matplotlib figure and axes object
        for modification.

    outfile: Path
        If mode="save", save generated plot to Path specified in `outfile` argument.
        Intermediate directories will be created if needed.

    fig: Figure
        If provided with `axes`, configure plotting with the provided `fig`
        object instead of creating a new figure. Useful for creating subplots.

    axes: Axes
        If provided with `fig`, plot to the provided `axes` object. Useful for
        creating subplots.


    Returns
    -------
    (fig, axes): (Figure, Axes)
        The handles to the matplotlib objects, only if `mode` is "return".
    """
    _configure_sbn_style()
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
        If "return", return (fig, axes), the matplotlib figure and axes object
        for modification.

    outfile: Path
        If mode="save", save generated plot to Path specified in `outfile` argument.
        Intermediate directories will be created if needed.

    kind: "scatter" (default) | "line"
        Whether to use a scatterplot or line plot.

    fig: Figure
        If provided with `axes`, configure plotting with the provided `fig`
        object instead of creating a new figure. Useful for creating subplots.

    axes: Axes
        If provided with `fig`, plot to the provided `axes` object. Useful for
        creating subplots.


    Returns
    -------
    (fig, axes): (Figure, Axes)
        The handles to the matplotlib objects, only if `mode` is "return".
    """
    _configure_sbn_style()
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
        with arguments {kernel="gau", bw="scott", cut=0} to compute and display
        the kde

    title: string
        The plot title string

    mode: "block" (default) | "noblock" | "save" | "return"
        If "block", call plot.plot() and display plot in a blocking fashion.
        If "noblock", attempt to generate plot in nonblocking fashion.
        If "save", save plot to pathlib Path specified in `outfile` argument
        If "return", return (fig, axes), the matplotlib figure and axes object
        for modification.

    outfile: Path
        If mode="save", save generated plot to Path specified in `outfile` argument.
        Intermediate directories will be created if needed.

    fig: Figure
        If provided with `axes`, configure plotting with the provided `fig`
        object instead of creating a new figure. Useful for creating subplots.

    axes: Axes
        If provided with `fig`, plot to the provided `axes` object. Useful for
        creating subplots.


    Returns
    -------
    (fig, axes): (Figure, Axes)
        The handles to the matplotlib objects, only if `mode` is "return".
    """
    _configure_sbn_style()
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
    eigs: ndarray,
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
        If "return", return (fig, axes), the matplotlib figure and axes object
        for modification.

    outfile: Path
        If mode="save", save generated plot to Path specified in `outfile` argument.
        Intermediate directories will be created if needed.

    kind: "scatter" (default) | "line"
        Whether to use a scatterplot or line plot.

    fig: Figure
        If provided with `axes`, configure plotting with the provided `fig`
        object instead of creating a new figure. Useful for creating subplots.

    axes: Axes
        If provided with `fig`, plot to the provided `axes` object. Useful for
        creating subplots.


    Returns
    -------
    (fig, axes): (Figure, Axes)
        The handles to the matplotlib objects, only if `mode` is "return".
    """
    _configure_sbn_style()
    # cmap = plt.cm.cividis
    cmap = plt.cm.gist_heat
    fig, axes = _setup_plotting(fig, axes)
    N = len(unfolded)
    step_vals = np.arange(0, N)
    df_line = pd.DataFrame({"Step Function": step_vals, "Unfolded λ": unfolded})
    # df_scatter = pd.DataFrame({"Step Function": steps, "Outlier": np.abs(unfolded)})
    sbn.lineplot(data=df_line, ax=axes)
    mean = np.mean(np.abs(unfolded))

    # 0 is left of cmap, color_max is right of cmap
    color = (5 * np.abs(unfolded) / mean) ** 2
    size = (10 * (np.abs(unfolded) - mean) / mean) ** 2
    inlier = np.abs(unfolded - step_vals) < 5
    color[inlier] = color.min()
    min_size = 0.5
    size[inlier] = min_size
    size[size < 1] = min_size
    axes.scatter(
        x=step_vals,
        y=unfolded,
        s=size,  # size of points
        c=color,  # color of points
        cmap=cmap,  # should be color blind safe
        marker="o",
        # color="red",
        edgecolors="white",
        linewidths=0.1,
        label="Outlier",
        alpha=0.4,
    )
    # plt.setp(ax_scatter, label="Outlier")
    axes.set(title=title, xlabel="Eigenvalue Index", ylabel="Unfolded Value")
    handles, labels = axes.get_legend_handles_labels()
    line_handles, line_labels = handles[:-1], labels[:-1]
    # axes.legend(line_handles, line_labels)
    cmap_handles = [Rectangle((0, 0), 1, 1)]
    # seems you only need a handler map for legend element that need a custom handler
    handler_map = dict(zip(cmap_handles, [HandlerColormap(cmap, num_stripes=16)]))
    labels = [line_labels[0], line_labels[1], "Inlier / Outlier"]
    axes.legend(
        handles=line_handles + cmap_handles,
        labels=labels,
        handler_map=handler_map,
        fontsize=12,
    ).set_visible(True)
    return _handle_plot_mode(mode, fig, axes, outfile)


# this essentially plots the nearest-neighbors spacing distribution
def _spacings(
    unfolded: ndarray,
    bins: int = 50,
    kde: bool = True,
    trim: float = 5.0,
    trim_kde: bool = False,
    kde_bw: Union[float, str] = "scott",
    brody: bool = False,
    brody_fit: str = "spacing",
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
        with arguments {kernel="gau", bw="scott", cut=0} to compute and display
        the kde

    trim: float
        If True, only use spacings <= `trim` for computing the KDE and plotting.
        Useful for when large spacings distort the histogram.

    trim_kde: bool
        If True, fit the KDE using only spacings <= `trim`. Otherwise, fit the
        KDE using all available spacings.

    kde_bw: float
        The bandwidth to use for kernel density estimation.

    brody: bool
        If True, compute the best-fitting Brody distribution via MLE, and plot
        that distribution.

    brody_fit: "spacing" | "mle"
        Method for estimating parameter of the Brody distribution. If "spacing", use
        [maximum spacing estimation](https://en.wikipedia.org/wiki/Maximum_spacing_estimation).
        If "mle", use maximum likelihood. The default is "spacing",
        as this may be preferable for the J-shape of the Brody distribution.

    title: string
        The plot title string

    mode: "block" | "noblock" | "save" | "return"
        If "block", call plot.plot() and display plot in a blocking fashion.
        If "noblock", attempt to generate plot in nonblocking fashion.
        If "save", save plot to pathlib Path specified in `outfile` argument
        If "return", return (fig, axes), the matplotlib figure and axes object
        for modification.

    outfile: Path
        If mode="save", save generated plot to Path specified in `outfile` argument.
        Intermediate directories will be created if needed.

    ensembles: ["poisson", "goe", "gue", "gse"]
        Which ensembles to display the expected NNSD curves for.

    fig: Figure
        If provided with `axes`, configure plotting with the provided `fig`
        object instead of creating a new figure. Useful for creating subplots.

    axes: Axes
        If provided with `fig`, plot to the provided `axes` object. Useful for
        creating subplots.


    Returns
    -------
    (fig, axes): (Figure, Axes)
        The handles to the matplotlib objects, only if `mode` is "return".
    """
    _configure_sbn_style()
    fig, axes = _setup_plotting(fig, axes)
    unfolded = np.sort(
        unfolded
    )  # solve issues where flexible smoothers cause reversals
    _spacings = np.diff(unfolded)
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
            _kde_plot(_spacings, s, axes, kde_bw)
        else:
            _kde_plot(all_spacings, s, axes, kde_bw)

    if brody is True:
        beta = fit_brody(_spacings)
        brody_vals = brody_dist(s, beta=beta)
        brod = axes.plot(s, brody_vals, label="Brody dist.")
        plt.setp(brod, color="#9d0000", linestyle="--")

    # adjusting the right bounds can be necessary when / if there are
    # many large eigenvalue spacings
    # axes.set_xlim(left=0, right=np.percentile(_spacings, 99))
    axes.set_ylim(top=1.5, bottom=0)
    axes.set_xlim(left=0, right=2.5)
    axes.set(title=title, ylabel="Density p(s)")
    axes.legend().set_visible(True)

    return _handle_plot_mode(mode, fig, axes, outfile)


def _brody_fit(
    unfolded: ndarray,
    method: str = "spacing",
    title: str = "Brody distribution fit",
    mode: PlotMode = "block",
    outfile: Path = None,
    save_dpi: int = None,
    ensembles: List[str] = ["goe", "poisson"],
    bins: int = 50,
    kde: bool = True,
    trim: float = 5.0,
    trim_kde: bool = False,
    kde_bw: Union[float, str] = "scott",
) -> PlotResult:
    _configure_sbn_style()
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    _spacings(
        unfolded=unfolded,
        bins=bins,
        kde=kde,
        trim=trim,
        trim_kde=trim_kde,
        kde_bw=kde_bw,
        brody=True,
        brody_fit=method,
        title="Brody distribution fit: density",
        mode="return",
        ensembles=ensembles,
        fig=fig,
        axes=axes[0],
    )
    spacings = np.diff(unfolded)
    s = spacings[spacings > 0]
    res = brody_fit_evaluate(s, method=method)
    x = res["spacings"]
    ecdf, bcdf = res["ecdf"], res["brody_cdf"]
    ax = axes.flat[1]
    ax_e = ax.plot(x, ecdf)
    plt.setp(ax_e, label="Empirical CDF")
    ax_b = ax.plot(x, bcdf)
    plt.setp(ax_b, label="Brody CDF", color="#9d0000", linestyle="--")
    if "goe" in ensembles:
        ax_goe = ax.plot(x, GOE.nnsd_cdf(spacings=x))
        plt.setp(ax_goe, label="GOE", color="#FD8208")
    if "poisson" in ensembles:
        ax_poi = ax.plot(x, Poisson.nnsd_cdf(spacings=x))
        plt.setp(ax_poi, label="Poisson", color="#08FD4F")
    ax.legend().set_visible(True)
    ax.set_title("Brody distribution fit: cumulative density")
    return _handle_plot_mode(mode, fig, axes, outfile, save_dpi)


def _next_spacings(
    unfolded: ndarray,
    bins: int = 50,
    kde: bool = True,
    trim: float = 0.0,
    trim_kde: bool = False,
    brody: bool = False,
    brody_fit: str = "spacing",
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
        with arguments {kernel="gau", bw="scott", cut=0} to compute and display
        the kde

    trim: float
        If True, only use spacings <= `trim` for computing the KDE and plotting.
        Useful for when large spacings distort the histogram.

    trim_kde: bool
        If True, fit the KDE using only spacings <= `trim`. Otherwise, fit the
        KDE using all available spacings.

    brody: bool
        If True, compute the best-fitting Brody distribution via MLE, and plot
        that distribution.

    brody_fit: "spacing" | "mle"
        Method for parametric distribution fitting of the Brody distribution to
        the data. If "spacing", use
        [maximum spacing estimation](https://en.wikipedia.org/wiki/Maximum_spacing_estimation).
        If "mle", use the maximum likelihood method. The default is "spacing",
        as this may be preferable for the J-shape of the Brody distribution.

    title: string
        The plot title string

    mode: "block" | "noblock" | "save" | "return"
        If "block", call plot.plot() and display plot in a blocking fashion.
        If "noblock", attempt to generate plot in nonblocking fashion.
        If "save", save plot to pathlib Path specified in `outfile` argument
        If "return", return (fig, axes), the matplotlib figure and axes object
        for modification.

    outfile: Path
        If mode="save", save generated plot to Path specified in `outfile` argument.
        Intermediate directories will be created if needed.

    ensembles: List["poisson", "goe"]
        Which ensembles to display the expected next-NNSD curves for.

    fig: Figure
        If provided with `axes`, configure plotting with the provided `fig`
        object instead of creating a new figure. Useful for creating subplots.

    axes: Axes
        If provided with `fig`, plot to the provided `axes` object. Useful for
        creating subplots.


    Returns
    -------
    (fig, axes): (Figure, Axes)
        The handles to the matplotlib objects, only if `mode` is "return".
    """
    _configure_sbn_style()
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

    if brody is True:
        beta = fit_brody(_spacings)
        brody_vals = brody_dist(s, beta=beta)
        brod = axes.plot(s, brody_vals, label="Brody dist.")
        plt.setp(brod, color="#9d0000", linestyle="--")

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
        If "return", return (fig, axes), the matplotlib figure and axes object
        for modification.

    outfile: Path
        If mode="save", save generated plot to Path specified in `outfile` argument.
        Intermediate directories will be created if needed.

    ensembles: ["poisson", "goe", "gue", "gse"]
        Which ensembles to display the expected spectral rigidity curves for comparison against.

    fig: Figure
        If provided with `axes`, configure plotting with the provided `fig`
        object instead of creating a new figure. Useful for creating subplots.

    axes: Axes
        If provided with `fig`, plot to the provided `axes` object. Useful for
        creating subplots.


    Returns
    -------
    (fig, axes): (Figure, Axes)
        The handles to the matplotlib objects, only if `mode` is "return".
    """
    _configure_sbn_style()
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
        are the values computed from
        observables.levelvariance.level_number_variance

    title: string
        The plot title string

    mode: "block" (default) | "noblock" | "save" | "return"
        If "block", call plot.plot() and display plot in a blocking fashion.
        If "noblock", attempt to generate plot in nonblocking fashion.
        If "save", save plot to pathlib Path specified in `outfile` argument
        If "return", return (fig, axes), the matplotlib figure and axes object
        for modification.

    outfile: Path
        If mode="save", save generated plot to Path specified in `outfile` argument.
        Intermediate directories will be created if needed.

    ensembles: ["poisson", "goe", "gue", "gse"]
        Which ensembles to display the expected number level variance curves for comparison against.

    fig: Figure
        If provided with `axes`, configure plotting with the provided `fig`
        object instead of creating a new figure. Useful for creating subplots.

    axes: Axes
        If provided with `fig`, plot to the provided `axes` object. Useful for
        creating subplots.


    Returns
    -------
    (fig, axes): (Figure, Axes)
        The handles to the matplotlib objects, only if `mode` is "return".
    """
    _configure_sbn_style()
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
        title=title,
        xlabel="L",
        ylabel="Sigma^2(L)",
    )
    axes.legend().set_visible(True)
    return _handle_plot_mode(mode, fig, axes, outfile)


def _observables(
    eigs: ndarray,
    unfolded: ndarray,
    rigidity_df: pd.DataFrame,
    levelvar_df: pd.DataFrame,
    suptitle: str = "Spectral Observables",
    mode: PlotMode = "block",
    outfile: Path = None,
    ensembles: List[str] = ["goe", "poisson"],
) -> PlotResult:
    """Plot some popular spectral observables, as well as a plot of the unfolding
    fit. For public use, use `Unfolded.plot_observables()`.

    eigs: ndarray
        The original eigenvalues (for plotting the unfolding fit).

    unfolded: ndarray
        The unfolded eigenvalues.

    rigidity_df: DataFrame
        The dataframe returned from unfolded.spectral_rigidity().

    levelvar_df: DataFrame
        The dataframe returned from unfolded.level_variance().

    suptitle: string
        The plot title string

    mode: "block" (default) | "noblock" | "save" | "return"
        If "block", call plot.plot() and display plot in a blocking fashion.
        If "noblock", attempt to generate plot in nonblocking fashion.
        If "save", save plot to pathlib Path specified in `outfile` argument
        If "return", return (fig, axes), the matplotlib figure and axes object
        for modification.

    outfile: Path
        If mode="save", save generated plot to Path specified in `outfile` argument.
        Intermediate directories will be created if needed.

    ensembles: ["poisson", "goe", "gue", "gse"]
        Which ensembles to display the expected number level variance curves for comparison against.
    """
    _configure_sbn_style()
    fig, axes = plt.subplots(2, 2, sharex="none", sharey="none")
    fig.set_size_inches(fig.get_size_inches() * 2)
    fig.suptitle(suptitle)
    _unfolded_fit(eigs=eigs, unfolded=unfolded, fig=fig, axes=axes[0, 0], mode="return")
    _spacings(unfolded, fig=fig, axes=axes[0, 1], mode="return", ensembles=ensembles)
    _spectral_rigidity(
        unfolded,
        data=rigidity_df,
        fig=fig,
        axes=axes[1, 0],
        mode="return",
        ensembles=ensembles,
    )
    _level_number_variance(
        unfolded,
        data=levelvar_df,
        fig=fig,
        axes=axes[1, 1],
        mode="return",
        ensembles=ensembles,
    )
    return _handle_plot_mode(mode, fig, axes, outfile)


def _plot_trim_iters(
    trims: List["TrimIter"],
    width: int = 4,
    title: str = "Trim fits",
    mode: PlotMode = "block",
    outfile: Path = None,
) -> PlotResult:
    """Plot the trim regions.

    Parameters
    ----------
    trims: List[TrimIter]
        The trim iterations.

    width: int
        The desired number of columns in the multiplot.

    title: str
        The 'suptitle' for all the subplots.

    mode: "block" (default) | "noblock" | "save" | "return"
        If "block", call plot.plot() and display plot in a blocking fashion.
        If "noblock", attempt to generate plot in nonblocking fashion.
        If "save", save plot to pathlib Path specified in `outfile` argument
        If "return", return (fig, axes), the matplotlib figure and axes object
        for modification.

    outfile: Path
        If mode="save", save generated plot to Path specified in `outfile` argument.
        Intermediate directories will be created if needed.


    """
    _configure_sbn_style()
    height = int(np.ceil(len(trims) / width))
    width = int(width)
    fig, axes = plt.subplots(height, width)
    for trim, ax in zip(trims, axes.flat):
        start, end = trim.trim_indices
        mean = float(trim_mean(trim.spacings.mean(), 0.2))
        var = float(trim_mean(trim.spacings.var(ddof=1), 0.2))
        cut = trim.percent_removed
        subtitle = "No trim" if trim.id == 0 else "{:.2f}% removed".format(cut)
        info = "<s> {:.4f} var(s) {:.4f}".format(mean, var)
        ax_title = f"{subtitle}\n{info}"
        df = pd.DataFrame(
            {
                "λ": trim.eigs,
                "N(λ)": np.arange(1, len(trim.eigs) + 1),
                "Cluster": trim.clusters,
            }
        )
        sbn.scatterplot(
            data=df,
            x="λ",
            y="N(λ)",
            hue="Cluster",
            style="Cluster",
            style_order=["inlier", "outlier"],
            linewidth=0,
            markers=[".", "X"],
            palette=["black", "red"],
            hue_order=["inlier", "outlier"],
            legend="brief",
            ax=ax,
        )
        ax.set(title=ax_title)
    for i in range(len(trims), len(axes.flat)):
        fig.delaxes(axes.flat[i])
    fig.subplots_adjust(wspace=0.8, hspace=0.8)
    fig.suptitle(title)
    fig.set_size_inches(width * 3, height * 3)
    return _handle_plot_mode(mode, fig, axes, outfile, 100)


def _configure_sbn_style() -> None:
    """Ensure *once* that seaborn style has been set.

    NOTE
    ----
    This must be called every time *before* creating new figs and axes. In order
    to fully ensure all plots get the correct style.
    """
    global PLOTTING_READY
    if not PLOTTING_READY:
        PALETTE = sbn.color_palette("dark").copy()
        PALETTE.insert(0, (0.0, 0.0, 0.0))
        sbn.set()
        sbn.set_palette(PALETTE)
        PLOTTING_READY = True


def _setup_plotting(fig: Figure = None, axes: Axes = None) -> Tuple[Figure, Axes]:
    """Get new axes and figure objects, or return passed in."""
    if fig is None or axes is None:
        fig, axes = plt.subplots()
        return fig, axes
    else:
        return fig, axes


def _kde_plot(
    values: ndarray, grid: ndarray, axes: Axes, bw: Union[float, str] = "scott"
) -> None:
    """Calculate KDE for observed spacings.

    Parameters
    ----------
    values: ndarray
        the values used to compute (fit) the kernel density estimate

    grid: ndarray
        the grid of values over which to evaluate the computed KDE curve

    axes: pyplot.Axes
        the current axes object to be modified

    bw: bandwidh
        The `bw` argument for statsmodels KDEUnivariate .fit

    Notes
    -----
    we are doing this manually because we want to ensure consistency of the KDE
    calculation and remove Seaborn control over the process, while also avoiding
    inconsistent behaviours like https://github.com/mwaskom/seaborn/issues/938
    and https://github.com/mwaskom/seaborn/issues/796
    """
    values = values[values > 0]  # prevent floating-point bad behaviour
    kde = KDE(values)
    # kde.fit(kernel="gau", bw="scott", cut=0)
    kde.fit(kernel="gau", bw=bw, cut=0)
    evaluated = np.empty_like(grid)
    for i, _ in enumerate(evaluated):
        evaluated[i] = kde.evaluate(grid[i])
    kde_curve = axes.plot(grid, evaluated, label="Kernel Density Estimate")
    plt.setp(kde_curve, color="black")


def _handle_plot_mode(
    mode: str, fig: Figure, axes: Axes, outfile: Path = None, save_dpi: int = None
) -> Union[Optional[PlotResult], Axes]:
    """Handle the various combinations of plotting arguments."""
    if mode == "block":
        plt.show(block=True)
    elif mode == "noblock":
        plt.show(block=False)
        plt.pause(0.001)
    elif mode == "test":
        plt.show(block=False)
        plt.close(fig)
    elif mode == "save":
        if outfile is None:
            raise ValueError("Path not specified for `outfile`.")
        try:
            outfile = Path(outfile)
        except BaseException as e:
            raise ValueError("Cannot interpret outfile path.") from e
        make_parent_directories(outfile)
        print(f"Saving figure to {outfile}")
        fig.savefig(outfile, dpi=save_dpi)
        plt.close(fig)
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


# shameless theft from https://stackoverflow.com/a/55501861, i.e.
# https://stackoverflow.com/questions/55501860/how-to-put-multiple-colormap-patches-in-a-matplotlib-legend
class HandlerColormap(HandlerBase):
    def __init__(self, cmap: plt.cm, num_stripes: int = 8, **kw: Any):
        HandlerBase.__init__(self, **kw)
        self.cmap = cmap
        self.num_stripes = num_stripes

    def create_artists(  # type: ignore
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        stripes = []
        for i in range(self.num_stripes):
            s = Rectangle(
                [xdescent + i * width / self.num_stripes, ydescent],
                width / self.num_stripes,
                height,
                fc=self.cmap((2 * i + 1) / (2 * self.num_stripes)),
                transform=trans,
                edgecolor="none",
            )
            stripes.append(s)
        return stripes
