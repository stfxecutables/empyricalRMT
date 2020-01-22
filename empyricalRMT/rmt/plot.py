import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import platform
import seaborn as sbn

from colorama import Fore, Style
from pathlib import Path
from statsmodels.nonparametric.kde import KDEUnivariate as KDE

from ..rmt.eigenvalues import stepFunctionVectorized, trim_largest
from ..rmt.observables.spacings import computeSpacings
from ..utils import make_parent_directories

RESET = Style.RESET_ALL


def setup_plotting(check_linux=False):
    PALETTE = sbn.color_palette("dark").copy()
    PALETTE.insert(0, (0.0, 0.0, 0.0))
    sbn.set()
    sbn.set_palette(PALETTE)

    # hack to make text larger on Ubuntu high DPI screen
    if check_linux:
        if platform.system() == "Linux":
            sbn.set_context("poster")


setup_plotting(False)


def validate_bin_sizes(vals, bins):
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
            raise Exception(
                "Distribution too skewed to generate interpretable histogram"
            )


def rawEigDist(
    eigs: np.array,
    bins=200,
    title="Raw Eigenvalue Distribution",
    kde=None,
    block=False,
    xlims=None,
):
    axes = sbn.distplot(eigs, bins=bins, kde=kde, axlabel="Eigenvalue", color="black")
    plt.ylabel("Density")
    plt.title(title)
    if xlims is not None:
        axes.set_xlim(xlims)
    if type(bins) is int:
        validate_bin_sizes(eigs, bins)
    plt.show(block=block)


def stepFunction(eigs: np.array, trim=True, percentile=97.5, block=False, xlims=None):
    if trim:
        eigs = trim_largest(eigs, percentile)
    grid = np.linspace(eigs.min(), eigs.max(), 100000)
    step_values = stepFunctionVectorized(eigs, grid)
    df = pd.DataFrame({"Cumulative Value": step_values, "Raw eigenvalues λ": grid})
    axes = sbn.lineplot(data=df, x="Raw eigenvalues λ", y="Cumulative Value")
    if xlims is not None:
        axes.set_xlim(xlims)
    plt.title("Step function of raw eigenvalues")
    plt.show(block=block)
    plt.clf()


def rawEigSorted(eigs: np.array, block=False):
    sbn.scatterplot(data=eigs)
    plt.xlabel("Eigenvalue index")
    plt.ylabel("Eigenvalue")
    plt.title("Raw Eigenvalue Distribution")
    plt.show(block=block)


def unfoldedDist(unfolded: np.array, method="Spline", block=False):
    sbn.distplot(unfolded, bins="doane", axlabel="Unfolded Eigenvalues", color="black")
    plt.ylabel("Density")
    plt.title(f"{method} Unfolded Eigenvalue Distribution")
    plt.show(block=block)


def unfoldedFit(unfolded: np.array, title="Spline Fit (Default)", block=False):
    N = len(unfolded)
    df = pd.DataFrame({"Cumulative Value": np.arange(1, N + 1), "Unfolded λ": unfolded})
    sbn.lineplot(data=df)
    plt.title(title)
    plt.show(block=block)


# this essentially plots the nearest-neighbors spacing distribution
def spacings(
    unfolded: np.array,
    bins=100,
    kde=False,
    title="Unfolded Spacing Distribution",
    mode="block",
    outfile: Path = Path("plots/spacings.png"),
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
        If "return", return the matplotlib axes object for modification.
    outfile: Path
        If mode="save", save generated plot to Path specified in `outfile` argument.
    """
    spacings = np.sort(unfolded[1:] - unfolded[:-1])
    # Generate expected distributions for classical ensembles
    pi = np.pi
    s = np.linspace(spacings.min(), spacings.max(), 10000)
    poisson = np.exp(-s)
    goe = ((np.pi * s) / 2) * np.exp(-(np.pi / 4) * s * s)
    gue = (32 / pi ** 2) * (s * s) * np.exp(-(4 * s * s) / pi)
    gse = (
        (2 ** 18 / (3 ** 6 * pi ** 3)) * (s ** 4) * np.exp(-((64 / (9 * pi)) * (s * s)))
    )

    axes = sbn.distplot(
        spacings,
        norm_hist=True,
        bins=bins,  # doane
        kde=False,
        label="Empirical Spacing Distribution",
        axlabel="spacing (s)",
        color="black",
    )

    # calculate KDE for observed spacings
    # we are doing this manually because we want to ensure consistecny of the KDE
    # calculation and remove Seaborn control over the process, while also avoiding
    # inconsistent behaviours like https://github.com/mwaskom/seaborn/issues/938 and
    # https://github.com/mwaskom/seaborn/issues/796
    if kde is True:
        kde = KDE(spacings)
        kde.fit(kernel="gau", bw="scott", cut=0)
        evaluated = np.empty_like(s)
        for i, _ in enumerate(evaluated):
            evaluated[i] = kde.evaluate(s[i])
        kde_curve = axes.plot(s, evaluated, label="Kernel Density Estimate")
        plt.setp(kde_curve, color="black")

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

    if mode == "block":
        plt.show(block=True)
    elif mode == "noblock":
        plt.show(block=False)
    elif mode == "save":
        print("Making parent directories")
        make_parent_directories(outfile)
        print("Saving figure")
        plt.savefig(outfile)
        # plt.clf()
    elif mode == "return":
        return axes
    else:
        raise Exception("Invalid plotting mode.")


def spectralRigidity(
    unfolded, data, title="Default", mode="block", outfile=Path("plots/rigidity")
):
    """
    `data` argument is such that:
        df = pd.DataFrame({"L": L_vals, "∆3(L)": delta3})
    """
    # L = pd.DataFrame({"L", L})
    # delta3 = pd.DataFrame({"∆3(L)", delta3})
    df = pd.DataFrame(data, columns=["L", "∆3(L)"])
    sbn.relplot(x="L", y="∆3(L)", data=df)
    # sbn.scatterplot(x=data., y="∆3(L)", data=df)

    _, right = plt.xlim()

    L = df["L"]
    poisson = L / 15 / 2
    pi = np.pi

    # see pg 290 of Mehta (2004) for definition of s
    s = L / np.mean(computeSpacings(unfolded, trim=False))
    # goe = (1/(pi**2)) * (np.log(2*pi*L) + np.euler_gamma - 5/4 - (pi**2)/8)
    # gue = (1/(2*(pi**2))) * (np.log(2*pi*L) + np.euler_gamma - 5/4)
    # gse = (1/(4*(pi**2))) * (np.log(4*pi*L) + np.euler_gamma - 5/4 + (pi**2)/8)
    goe = (1 / (pi ** 2)) * (
        np.log(2 * pi * s) + np.euler_gamma - 5 / 4 - (pi ** 2) / 8
    )
    gue = (1 / (2 * (pi ** 2))) * (np.log(2 * pi * s) + np.euler_gamma - 5 / 4)
    gse = (1 / (4 * (pi ** 2))) * (
        np.log(4 * pi * s) + np.euler_gamma - 5 / 4 + (pi ** 2) / 8
    )

    poisson = plt.plot(L, poisson, label="Poisson")
    plt.setp(poisson, color="#08FD4F")

    goe = plt.plot(L, goe, label="Gaussian Orthogonal")
    plt.setp(goe, color="#FD8208")

    gue = plt.plot(L, gue, label="Gaussian Unitary")
    plt.setp(gue, color="#0066FF")

    gse = plt.plot(L, gse, label="Gaussian Symplectic")
    plt.setp(gse, color="#EA00FF")

    plt.xlabel("L")
    plt.ylabel("∆3(L)")
    plt.title(f"Spectral Rigidity - {title} Unfolding")
    plt.legend()

    if mode == "block":
        plt.show(block=True)
    elif mode == "noblock":
        plt.show(block=False)
    elif mode == "save":
        make_parent_directories(outfile)
        plt.savefig(outfile)
    else:
        raise Exception("Invalid plotting mode.")


def levelNumberVariance(
    unfolded, data, title="Default", mode="block", outfile=Path("plots/levelnumber")
):
    df = pd.DataFrame(data, columns=["L", "∑²(L)"])
    sbn.relplot(x="L", y="∑²(L)", data=df)

    _, right = plt.xlim()

    L = df["L"]
    pi = np.pi
    s = L / np.mean(computeSpacings(unfolded, trim=False))

    poisson = L / 2  # waste of time, too large very often
    goe = (2 / (pi ** 2)) * (np.log(2 * pi * s) + np.euler_gamma + 1 - (pi ** 2) / 8)
    gue = (1 / (pi ** 2)) * (np.log(2 * pi * s) + np.euler_gamma + 1)
    gse = (1 / (2 * (pi ** 2))) * (
        np.log(4 * pi * s) + np.euler_gamma + 1 + (pi ** 2) / 8
    )

    poisson = plt.plot(L, poisson, label="Poisson")
    plt.setp(poisson, color="#08FD4F")

    goe = plt.plot(L, goe, label="Gaussian Orthogonal")
    plt.setp(goe, color="#FD8208")

    gue = plt.plot(L, gue, label="Gaussian Unitary")
    plt.setp(gue, color="#0066FF")

    gse = plt.plot(L, gse, label="Gaussian Symplectic")
    plt.setp(gse, color="#EA00FF")

    plt.xlabel("L")
    plt.ylabel("∑²(L)")
    plt.title(f"Level Number Variance - {title} unfolding")
    plt.legend()
    if mode == "block":
        plt.show(block=True)
    elif mode == "noblock":
        plt.show(block=False)
    elif mode == "save":
        make_parent_directories(outfile)
        plt.savefig(outfile)
    else:
        raise Exception("Invalid plotting mode.")
