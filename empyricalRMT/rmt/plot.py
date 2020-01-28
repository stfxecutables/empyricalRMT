import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sbn

from colorama import Fore, Style
from pathlib import Path
from statsmodels.nonparametric.kde import KDEUnivariate as KDE
from warnings import warn

from empyricalRMT.rmt._eigvals import stepFunctionVectorized
from empyricalRMT.rmt.observables.spacings import computeSpacings
from empyricalRMT.utils import make_parent_directories

RESET = Style.RESET_ALL
PLOTTING_READY = False


def rawEigDist(
    eigs: np.array,
    bins=50,
    title="Raw Eigenvalue Distribution",
    kde=True,
    mode="block",
    outfile: Path = None,
):
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


def stepFunction(
    eigs: np.array,
    gridsize=100000,
    title="Eigenvalue Step Function",
    mode="block",
    outfile: Path = None,
):
    _setup_plotting()
    grid = np.linspace(eigs.min(), eigs.max(), gridsize)
    steps = stepFunctionVectorized(eigs, grid)
    df = pd.DataFrame({"Cumulative Value": steps, "Raw eigenvalues λ": grid})
    axes = sbn.lineplot(data=df, x="Raw eigenvalues λ", y="Cumulative Value")
    plt.title(title)
    return _handle_plot_mode(mode, axes, outfile)


def rawEigSorted(
    eigs: np.array, title="Raw Eigenvalues", mode="block", outfile: Path = None
):
    _setup_plotting()
    axes = sbn.scatterplot(data=eigs)
    plt.xlabel("Eigenvalue index")
    plt.ylabel("Eigenvalue")
    plt.title(title)
    return _handle_plot_mode(mode, axes, outfile)


def unfoldedDist(
    unfolded: np.array,
    bins=50,
    kde=True,
    title="Unfolded Eigenvalues",
    mode="block",
    outfile=None,
):
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


def unfoldedFit(
    unfolded: np.array, title="Unfolding Fit", mode="block", outfile: Path = None
):
    _setup_plotting()
    N = len(unfolded)
    df = pd.DataFrame({"Step Function": np.arange(1, N + 1), "Unfolded λ": unfolded})
    axes = sbn.lineplot(data=df)
    plt.title(title)
    return _handle_plot_mode(mode, axes, outfile)


# this essentially plots the nearest-neighbors spacing distribution
def spacings(
    unfolded: np.array,
    bins=50,
    kde=True,
    title="Unfolded Spacing Distribution",
    mode="block",
    outfile: Path = None,
):
    """Plots a histogram of the Nearest-Neighbors Spacing Distribution

    Parameters
    ----------
    unfolded: np.array
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
    spacings = np.sort(unfolded[1:] - unfolded[:-1])
    # Generate expected distributions for classical ensembles
    p = np.pi
    s = np.linspace(spacings.min(), spacings.max(), 10000)
    poisson = np.exp(-s)
    goe = ((p * s) / 2) * np.exp(-(p / 4) * s * s)
    gue = (32 / p ** 2) * (s * s) * np.exp(-(4 * s * s) / p)
    gse = (2 ** 18 / (3 ** 6 * p ** 3)) * (s ** 4) * np.exp(-((64 / (9 * p)) * (s * s)))

    axes = sbn.distplot(
        spacings,
        norm_hist=True,
        bins=bins,  # doane
        kde=False,
        label="Empirical Spacing Distribution",
        axlabel="spacing (s)",
        color="black",
    )

    if kde is True:
        _kde_plot(spacings, s, axes)

    poisson = axes.plot(s, poisson, label="Poisson")
    goe = axes.plot(s, goe, label="Gaussian Orthogonal")
    gue = axes.plot(s, gue, label="Gaussian Unitary")
    gse = plt.plot(s, gse, label="Gaussian Symplectic")
    plt.setp(poisson, color="#08FD4F")
    plt.setp(goe, color="#FD8208")
    plt.setp(gue, color="#0066FF")
    plt.setp(gse, color="#EA00FF")

    plt.ylabel("Density p(s)")
    plt.title(title)
    plt.legend()
    # adjusting the right bounds can be necessary when / if there are
    # many large eigenvalue spacings
    axes.set_xlim(left=0, right=np.percentile(spacings, 99))

    return _handle_plot_mode(mode, axes, outfile)


def spectralRigidity(
    unfolded: np.array,
    data: pd.DataFrame,
    title="Spectral Rigidity",
    mode="block",
    outfile: Path = None,
):
    """
    `data` argument is such that:
        df = pd.DataFrame({"L": L_vals, "∆3(L)": delta3})
    """
    _setup_plotting()
    # L = pd.DataFrame({"L", L})
    # delta3 = pd.DataFrame({"∆3(L)", delta3})
    df = pd.DataFrame(data, columns=["L", "∆3(L)"])
    axes = sbn.relplot(x="L", y="∆3(L)", data=df)
    # sbn.scatterplot(x=data., y="∆3(L)", data=df)

    _, right = plt.xlim()

    L = df["L"]
    poisson = L / 15 / 2
    p, y = np.pi, np.euler_gamma

    # see pg 290 of Mehta (2004) for definition of s
    s = L / np.mean(computeSpacings(unfolded, trim=False))
    # goe = (1/(pi**2)) * (np.log(2*pi*L) + np.euler_gamma - 5/4 - (pi**2)/8)
    # gue = (1/(2*(pi**2))) * (np.log(2*pi*L) + np.euler_gamma - 5/4)
    # gse = (1/(4*(pi**2))) * (np.log(4*pi*L) + np.euler_gamma - 5/4 + (pi**2)/8)
    goe = (1 / (p ** 2)) * (np.log(2 * p * s) + y - 5 / 4 - (p ** 2) / 8)
    gue = (1 / (2 * (p ** 2))) * (np.log(2 * p * s) + y - 5 / 4)
    gse = (1 / (4 * (p ** 2))) * (np.log(4 * p * s) + y - 5 / 4 + (p ** 2) / 8)

    poisson = plt.plot(L, poisson, label="Poisson")
    goe = plt.plot(L, goe, label="Gaussian Orthogonal")
    gue = plt.plot(L, gue, label="Gaussian Unitary")
    gse = plt.plot(L, gse, label="Gaussian Symplectic")
    plt.setp(poisson, color="#08FD4F")
    plt.setp(goe, color="#FD8208")
    plt.setp(gue, color="#0066FF")
    plt.setp(gse, color="#EA00FF")

    plt.xlabel("L")
    plt.ylabel("∆3(L)")
    plt.title(title)
    plt.legend()

    _handle_plot_mode(mode, axes, outfile)


def levelNumberVariance(
    unfolded: np.array,
    data: pd.DataFrame,
    title="Level Number Variance",
    mode="block",
    outfile: Path = None,
):
    _setup_plotting()
    df = pd.DataFrame(data, columns=["L", "∑²(L)"])
    axes = sbn.relplot(x="L", y="∑²(L)", data=df)

    _, right = plt.xlim()

    L = df["L"]
    p, y = np.pi, np.euler_gamma
    s = L / np.mean(computeSpacings(unfolded, trim=False))

    poisson = L / 2  # waste of time, too large very often
    goe = (2 / (p ** 2)) * (np.log(2 * p * s) + y + 1 - (p ** 2) / 8)
    gue = (1 / (p ** 2)) * (np.log(2 * p * s) + y + 1)
    gse = (1 / (2 * (p ** 2))) * (np.log(4 * p * s) + y + 1 + (p ** 2) / 8)

    poisson = plt.plot(L, poisson, label="Poisson")
    goe = plt.plot(L, goe, label="Gaussian Orthogonal")
    gue = plt.plot(L, gue, label="Gaussian Unitary")
    gse = plt.plot(L, gse, label="Gaussian Symplectic")
    plt.setp(poisson, color="#08FD4F")
    plt.setp(goe, color="#FD8208")
    plt.setp(gue, color="#0066FF")
    plt.setp(gse, color="#EA00FF")

    plt.xlabel("L")
    plt.ylabel("∑²(L)")
    plt.title(f"Level Number Variance - {title} unfolding")
    plt.legend()

    _handle_plot_mode(mode, axes, outfile)


def _setup_plotting():
    global PLOTTING_READY
    if PLOTTING_READY:
        return
    PALETTE = sbn.color_palette("dark").copy()
    PALETTE.insert(0, (0.0, 0.0, 0.0))
    sbn.set()
    sbn.set_palette(PALETTE)
    PLOTTING_READY = True


def _kde_plot(values: np.array, grid: np.array, axes):
    """
    calculate KDE for observed spacings
    we are doing this manually because we want to ensure consistency of the KDE
    calculation and remove Seaborn control over the process, while also avoiding
    inconsistent behaviours like https://github.com/mwaskom/seaborn/issues/938 and
    https://github.com/mwaskom/seaborn/issues/796

    Parameters
    ----------
    values: np.array
        the values used to compute the kernel density estimate
    grid: np.array
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


def _handle_plot_mode(mode, axes, outfile):
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


def _validate_bin_sizes(vals, bins):
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
