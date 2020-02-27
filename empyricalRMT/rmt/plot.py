import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sbn
import warnings

from colorama import Fore, Style
from numpy import ndarray
from pathlib import Path
from scipy.integrate import quad
from scipy.special import sici
from statsmodels.nonparametric.kde import KDEUnivariate as KDE
from typing import Any, List, Optional, Tuple, Union
from typing_extensions import Literal
from warnings import warn


from empyricalRMT.rmt.observables.step import _step_function_fast
from empyricalRMT.utils import make_parent_directories

PlotResult = Optional[Tuple[plt.Figure, plt.Axes]]
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
    _setup_plotting()
    axes = sbn.distplot(
        eigs,
        norm_hist=True,
        bins=bins,  # doane
        kde=False,
        label="Raw Eigenvalue Distribution",
        axlabel="Eigenvalue",
        color="black",
    )
    if kde:
        grid = np.linspace(eigs.min(), eigs.max(), 10000)
        _kde_plot(eigs, grid, axes)

    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    return _handle_plot_mode(mode, axes, outfile)


def _step_function(
    eigs: ndarray,
    gridsize: int = 100000,
    title: str = "Eigenvalue Step Function",
    mode: PlotMode = "block",
    outfile: Path = None,
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
    _setup_plotting()
    grid = np.linspace(eigs.min(), eigs.max(), gridsize)
    steps = _step_function_fast(eigs, grid)
    df = pd.DataFrame({"Cumulative Value": steps, "Raw eigenvalues λ": grid})
    axes = sbn.lineplot(data=df, x="Raw eigenvalues λ", y="Cumulative Value")
    plt.title(title)
    return _handle_plot_mode(mode, axes, outfile)


def _raw_eig_sorted(
    eigs: ndarray,
    title: str = "Raw Eigenvalues",
    mode: PlotMode = "block",
    outfile: Path = None,
    kind: Union[Literal["scatter"], Literal["line"]] = "scatter",
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
    _setup_plotting()
    if kind == "scatter":
        axes = sbn.scatterplot(data=eigs)
    elif kind == "line":
        axes = sbn.lineplot(data=eigs)
    else:
        raise ValueError("Invalid plot kind. Must be 'scatter' or 'line'.")
    plt.xlabel("Eigenvalue index")
    plt.ylabel("Eigenvalue")
    plt.title(title)
    return _handle_plot_mode(mode, axes, outfile)


def _unfolded_dist(
    unfolded: ndarray,
    bins: int = 50,
    kde: bool = True,
    title: str = "Unfolded Eigenvalues",
    mode: PlotMode = "block",
    outfile: Path = None,
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
    _setup_plotting()
    axes = sbn.distplot(
        unfolded,
        norm_hist=True,
        bins=bins,  # doane
        kde=False,
        label="Unfolded Eigenvalue Distribution",
        axlabel="Eigenvalue",
        color="black",
    )
    if kde:
        grid = np.linspace(unfolded.min(), unfolded.max(), 10000)
        _kde_plot(unfolded, grid, axes)

    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    return _handle_plot_mode(mode, axes, outfile)


def _unfolded_fit(
    unfolded: ndarray,
    title: str = "Unfolding Fit",
    mode: PlotMode = "block",
    outfile: Path = None,
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
    _setup_plotting()
    N = len(unfolded)
    df = pd.DataFrame({"Step Function": np.arange(1, N + 1), "Unfolded λ": unfolded})
    axes = sbn.lineplot(data=df)
    plt.title(title)
    return _handle_plot_mode(mode, axes, outfile)


# this essentially plots the nearest-neighbors spacing distribution
def _spacings(
    unfolded: ndarray,
    bins: int = 50,
    kde: bool = True,
    title: str = "Unfolded Spacing Distribution",
    mode: PlotMode = "block",
    outfile: Path = None,
    ensembles: List[str] = ["poisson", "goe", "gue", "gse"],
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
        Which ensembles to display the expected spectral rigidity curves for comparison against.

    Returns
    -------
    (fig, axes): (Figure, Axes)
        The handles to the matplotlib objects, only if `mode` is "return".
    """
    _setup_plotting()
    _spacings = np.sort(unfolded[1:] - unfolded[:-1])
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
        gse = plt.plot(s, gse, label="Gaussian Symplectic")
        plt.setp(gse, color="#EA00FF")
    # fmt: on

    if kde is True:
        _kde_plot(_spacings, s, axes)

    plt.ylabel("Density p(s)")
    plt.title(title)
    plt.legend()
    # adjusting the right bounds can be necessary when / if there are
    # many large eigenvalue spacings
    axes.set_xlim(left=0, right=np.percentile(_spacings, 99))

    return _handle_plot_mode(mode, axes, outfile)


def _next_spacings(
    unfolded: ndarray,
    bins: int = 50,
    kde: bool = True,
    title: str = "next Nearest-Neigbors Spacing Distribution",
    mode: PlotMode = "block",
    outfile: Path = None,
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

    Returns
    -------
    (fig, axes): (Figure, Axes)
        The handles to the matplotlib objects, only if `mode` is "return".
    """
    _setup_plotting()
    _spacings = np.sort((unfolded[2:] - unfolded[:-2]) / 2)
    # Generate expected distributions for classical ensembles
    p = np.pi
    s = np.linspace(_spacings.min(), _spacings.max(), 10000)
    # see:
    # Dettmann, C. P., Georgiou, O., & Knight, G. (2017).
    # Spectral statistics of random geometric graphs.
    # EPL (Europhysics Letters), 118(1), 18003.
    # doi:10.1209/0295-5075/118/18003, pp10, Equation. 11
    # for this expected distribution formula
    goe = (2 ** 18 / (3 ** 6 * p ** 3)) * (s ** 4) * np.exp(-((64 / (9 * p)) * (s * s)))

    axes = sbn.distplot(
        _spacings,
        norm_hist=True,
        bins=bins,  # doane
        kde=False,
        label="next NNSD",
        axlabel="spacing (s_2)",
        color="black",
    )

    if kde is True:
        _kde_plot(_spacings, s, axes)

    goe = axes.plot(s, goe, label="Gaussian Orthogonal")
    plt.setp(goe, color="#FD8208")

    plt.ylabel("Density p(s)")
    plt.title(title)
    plt.legend()
    # adjusting the right bounds can be necessary when / if there are
    # many large eigenvalue spacings
    axes.set_xlim(left=0, right=np.percentile(_spacings, 99))

    return _handle_plot_mode(mode, axes, outfile)


def _spectral_rigidity(
    unfolded: Optional[ndarray],
    data: pd.DataFrame,
    title: str = "Spectral Rigidity",
    mode: PlotMode = "block",
    outfile: Path = None,
    ensembles: List[str] = ["poisson", "goe", "gue", "gse"],
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
    _setup_plotting()
    df = pd.DataFrame(data, columns=["L", "delta"])
    axes = sbn.relplot(x="L", y="delta", data=df)
    ensembles = set(ensembles)  # type: ignore

    _, right = plt.xlim()

    L = df["L"]
    p, y = np.pi, np.euler_gamma

    # see pg 290 of Mehta (2004) for definition of s
    s = L / np.mean(unfolded[1:] - unfolded[:-1]) if unfolded is not None else L

    if "poisson" in ensembles:
        poisson = L / 15 / 2
        poisson = plt.plot(L, poisson, label="Poisson")
        plt.setp(poisson, color="#08FD4F")
    if "goe" in ensembles:
        goe = (1 / (p ** 2)) * (np.log(2 * p * s) + y - 5 / 4 - (p ** 2) / 8)
        goe = plt.plot(L, goe, label="Gaussian Orthogonal")
        plt.setp(goe, color="#FD8208")
    if "gue" in ensembles:
        gue = (1 / (2 * (p ** 2))) * (np.log(2 * p * s) + y - 5 / 4)
        gue = plt.plot(L, gue, label="Gaussian Unitary")
        plt.setp(gue, color="#0066FF")
    if "gse" in ensembles:
        gse = (1 / (4 * (p ** 2))) * (np.log(4 * p * s) + y - 5 / 4 + (p ** 2) / 8)
        gse = plt.plot(L, gse, label="Gaussian Symplectic")
        plt.setp(gse, color="#EA00FF")

    plt.xlabel("L")
    plt.ylabel("∆3(L)")
    plt.title(title)
    plt.legend()

    return _handle_plot_mode(mode, axes, outfile)


def _level_number_variance(
    unfolded: ndarray,
    data: pd.DataFrame,
    title: str = "Level Number Variance",
    mode: PlotMode = "block",
    outfile: Path = None,
    ensembles: List[str] = ["poisson", "goe", "gue", "gse"],
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
    _setup_plotting()
    df = pd.DataFrame(data, columns=["L", "sigma"])
    axes = sbn.relplot(x="L", y="sigma", data=df)
    ensembles = set(ensembles)  # type: ignore

    _, right = plt.xlim()

    L = df["L"]
    p, y = np.pi, np.euler_gamma
    s = L / np.mean(unfolded[1:] - unfolded[:-1])

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
        poisson = plt.plot(L, poisson, label="Poisson")
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
        goe = plt.plot(L, goe, label="Gaussian Orthogonal")
        plt.setp(goe, color="#FD8208")
    if "gue" in ensembles:
        gue = (1 / (p ** 2)) * (np.log(2 * p * s) + y + 1)
        gue = plt.plot(L, gue, label="Gaussian Unitary")
        plt.setp(gue, color="#0066FF")
    if "gse" in ensembles:
        gse = (1 / (2 * (p ** 2))) * (np.log(4 * p * s) + y + 1 + (p ** 2) / 8)
        gse = plt.plot(L, gse, label="Gaussian Symplectic")
        plt.setp(gse, color="#EA00FF")

    plt.xlabel("L")
    plt.ylabel("Sigma^2(L)")
    plt.title(f"Level Number Variance - {title} unfolding")
    plt.legend()

    return _handle_plot_mode(mode, axes, outfile)


def _setup_plotting() -> None:
    global PLOTTING_READY
    if PLOTTING_READY:
        return
    PALETTE = sbn.color_palette("dark").copy()
    PALETTE.insert(0, (0.0, 0.0, 0.0))
    sbn.set()
    sbn.set_palette(PALETTE)
    PLOTTING_READY = True


def _kde_plot(values: ndarray, grid: ndarray, axes: plt.Axes) -> None:
    """
    calculate KDE for observed spacings
    we are doing this manually because we want to ensure consistency of the KDE
    calculation and remove Seaborn control over the process, while also avoiding
    inconsistent behaviours like https://github.com/mwaskom/seaborn/issues/938 and
    https://github.com/mwaskom/seaborn/issues/796

    Parameters
    ----------
    values: ndarray
        the values used to compute the kernel density estimate
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
    mode: str, axes: plt.Axes, outfile: Path = None
) -> Optional[PlotResult]:
    if mode == "block":
        plt.show(block=True)
    elif mode == "noblock":
        plt.show(block=False)
    elif mode == "save":
        if outfile is None:
            raise ValueError("Path not specified for `outfile`.")
        try:
            outfile = Path(outfile)
        except BaseException as e:
            raise ValueError("Cannot interpret outfile path.") from e
        make_parent_directories(outfile)
        print(f"Saving figure to {outfile}")
        plt.savefig(outfile)
    elif mode == "return":
        fig = plt.gcf()
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
