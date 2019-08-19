import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import platform
import seaborn as sbn

from colorama import Fore, Style
from pathlib import Path

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
    spacings: np.array,
    bins=100,
    kde=False,
    title=None,
    mode="block",
    outfile: Path = Path("plots/spacings.png"),
):
    # print("Computing kernel density estimate")
    axes = sbn.distplot(
        spacings,
        norm_hist=True,
        bins=bins,  # doane
        kde=kde,
        axlabel="spacing (s)",
        color="black",
        label="Empirical Spacing Distribution",
        kde_kws={
            "label": "Kernel Density Estimate",
            # "bw": "silverman"
        },
    )
    # _, right = plt.xlim()
    # x = np.linspace(0.01, right, 1000)

    # print("Computing predicted curves")
    pi = np.pi
    # x = np.linspace(spacings.min(), spacings[spacings < 10].max(), 1000)
    x = np.linspace(spacings.min(), spacings.max(), 10000)
    poisson = np.exp(-x)
    goe = ((np.pi * x) / 2) * np.exp(-(np.pi / 4) * x * x)
    gue = (32 / pi ** 2) * (x * x) * np.exp(-(4 * x * x) / pi)
    gse = (
        (2 ** 18 / (3 ** 6 * pi ** 3)) * (x ** 4) * np.exp(-((64 / (9 * pi)) * (x * x)))
    )

    # poisson = plt.plot(x, poisson, label="Poisson")
    poisson = axes.plot(x, poisson, label="Poisson")
    plt.setp(poisson, color="#08FD4F")

    # goe = plt.plot(x, goe, label="Gaussian Orthogonal")
    goe = axes.plot(x, goe, label="Gaussian Orthogonal")
    plt.setp(goe, color="#FD8208")

    gue = axes.plot(x, gue, label="Gaussian Unitary")
    plt.setp(gue, color="#0066FF")

    gse = plt.plot(x, gse, label="Gaussian Symplectic")
    plt.setp(gse, color="#EA00FF")

    plt.ylabel("Density p(s)")
    if title is None:
        plt.title(f"Spacing Distribution - {title} Unfolding")
    else:
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
    else:
        raise Exception("Invalid plotting mode.")


def spectralRigidity(
    unfolded, data, title="Default", mode="block", outfile=Path("plots/rigidity")
):
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
