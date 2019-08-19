import gc
import numpy as np
import pandas as pd

from colorama import init, Fore, Back, Style
from glob import glob
from platform import system
from pathlib import Path

import rmt.plot

from rmt import plot
from rmt.eigenvalues import trim_iteratively, trim_largest, trim_middle
from rmt.plot import spacings as plotSpacings
from rmt.observables.levelvariance import sigma_squared
from rmt.observables.spacings import computeSpacings
from rmt.observables.rigidity import spectralRigidityRewrite as computeRigidity
from rmt.unfold import polynomial as poly_unfold
from rmt.unfold import spline
from utils import make_directory, res


def group_eig_dists(
    bids_root: Path,
    bids_outdir_name: str = "rmt",
    eig_prefix: str = "eigs",
    group_regex: str = "bold.npy",
    title: str = "Raw Eigenvalue Dist.",
    group_label: str = "Task",
    outdir: Path = Path("plots/eigdists"),
    bins="auto",
    trim_method="none",
    trim_percent=98,
    block=True,
):
    eigs = get_eig_files(bids_root, eig_prefix, group_regex, log=False)

    for i, eig in enumerate(eigs):
        eigvals = np.load(eig)
        eigvals = trim_via_method(eigvals, trim_method, trim_percent)

        rmt.plot.rawEigDist(eigvals, bins=bins, kde=False, block=True, xlims=None)
        rmt.plot.stepFunction(eigvals, trim=False, block=True, xlims=None)


def group_spacings(
    bids_root: Path,
    bids_outdir_name: str = "rmt",
    eig_prefix: str = "eigs",
    group_regex: str = "bold.npy",
    title: str = "Nearest Neighbours Spacing Dist.",
    group_label: str = "Task",
    outdir: Path = Path("plots/spacings"),
    degree=5,
    grid_length=10000,
    trim_method="largest",
    trim_percent=98,
    detrend=None,
    percent=None,
    block=False,
    dry_run=False,
):
    eigs = get_eig_files(bids_root, eig_prefix, group_regex, log=False)
    if dry_run:
        print("Got eigenvalue files: ")
        [print("\t", eig) for eig in eigs]
        return

    for i, eig in enumerate(eigs):
        eigvals = np.load(eig)
        # eigvals = trim_via_method(eigvals, trim_method, trim_percent)
        # unfolded = poly_unfold(eigvals, degree, grid_length, detrend, percent, plot=True)
        unfolded = spline(eigvals, degree, detrend=None, plot=True)
        spacings = computeSpacings(unfolded, trim=True)
        print(f"Computed spacings. After unfolding:")
        print(
            f"mean spacing: {np.mean(spacings)}, median: {np.median(spacings)} var: {np.var(spacings)}"
        )
        print(f"[min, max] spacing: [{spacings.min(), spacings.max()}]")

        if detrend:
            make_directory(outdir / "detrended")
            outfile = outdir / "detrended" / f"{eig.stem}.png"
        outfile = outdir / f"{eig.stem}.png"
        plotSpacings(
            spacings=spacings,
            bins=200,
            kde=True,
            title=f"{title}: {group_label} S{i+1}",
            mode="block" if block else "save",
            outfile=outfile,
        )

        # raise Exception("Done")
        print(
            f"{Fore.GREEN}Saved spacing distribution plot to {outfile}{Style.RESET_ALL}"
        )
        gc.collect()


def group_rigidity(
    bids_root: Path,
    bids_outdir_name: str = "rmt",
    eig_prefix: str = "eigs",
    group_regex: str = "bold.npy",
    title: str = "Spectral Rigidity ∆3(L) - ",
    group_label: str = "Task",
    outdir: Path = Path("plots/rigidity"),
    degree=5,
    grid_length=10000,
    trim_method="largest",
    trim_percent=98,
    detrend=None,
):
    eigs = get_eig_files(bids_root, eig_prefix, group_regex, log=False)

    for i, eig in enumerate(eigs):
        eigvals = np.load(eig)
        eigvals = trim_via_method(eigvals, trim_method, trim_percent)
        unfolded = poly_unfold(eigvals, degree, grid_length, detrend)
        L, rigidity = computeRigidity(
            eigvals, unfolded, c_iters=1000, L_grid_size=100, min_L=1, max_L=25
        )

        df = pd.DataFrame({"L": L, "∆3(L)": rigidity})
        # outfile = outdir / f"{eig.stem}.png".replace(eig_prefix + "_corrmat", "rigidity")
        outfile = outdir / f"{eig.stem}.pdf".replace(
            eig_prefix + "_corrmat", "rigidity"
        )
        plot.spectralRigidity(
            unfolded,
            df,
            title=f"{title}: {group_label} S{i+1}",
            mode="save",
            outfile=outfile,
        )

        print(f"{Fore.GREEN}Saved spectral rigidity plot to {outfile}{Style.RESET_ALL}")
        raise Exception("Done")
        gc.collect()


def group_levelvariance(
    bids_root: Path,
    bids_outdir_name: str = "rmt",
    eig_prefix: str = "eigs",
    group_regex: str = "bold.npy",
    title: str = "Level Number Variance ∑²(L) - ",
    group_label: str = "Task",
    outdir: Path = Path("plots/levelnumber"),
    degree=5,
    grid_length=10000,
    trim_method="largest",
    trim_percent=98,
    fit_percent=100,
    block=False,
):
    eigs = get_eig_files(bids_root, eig_prefix, group_regex, log=False)

    for i, eig in enumerate(eigs):
        eigvals = np.load(eig)
        eigvals = trim_via_method(eigvals, trim_method, trim_percent)
        # unfolded = poly_unfold(eigvals, degree, grid_length, detrend=None, percent=fit_percent)
        unfolded = spline(eigvals, degree, detrend=None, plot=True)
        L, sigma_sq = sigma_squared(
            eigvals, unfolded, c_iters=1000, L_grid_size=500, min_L=1, max_L=25
        )

        df = pd.DataFrame({"L": L, "∑²(L)": sigma_sq})

        outfile = outdir / f"{eig.stem}.png".replace(
            eig_prefix + "_corrmat", f"lvar_d{degree}"
        )
        plot.levelNumberVariance(
            unfolded,
            df,
            title=f"{title}: {group_label} S{i+1}",
            mode="block" if block else "save",
            outfile=outfile,
        )

        print(
            f"{Fore.GREEN}Saved level number variance plot to {outfile}{Style.RESET_ALL}"
        )
        gc.collect()


def get_eig_files(bids_root: Path, eig_prefix: str, group_regex: str, log=False):
    eig_regex = None
    if system() != "Windows":
        eig_regex = f"{res(bids_root)}/**/{eig_prefix}*{group_regex}"
    else:
        eig_regex = f"{res(bids_root)}\\**\\{eig_prefix}*{group_regex}"
    eigs = [Path(file) for file in glob(eig_regex, recursive=True)]
    if len(eigs) == 0:
        raise Exception(
            f"Bad eigenvalue data regex {eig_regex}. No saved .npy data files found"
        )
    eigs.sort()

    if log:
        print("Found files:")
        for eig in eigs:
            print(eig.name)
    return eigs


def trim_via_method(
    eigvals: np.array, method: str, percent: float, print_summary=True
) -> np.array:
    if method == "largest":
        eigvals = trim_largest(eigvals, percent)
    elif method == "middle":
        eigvals = trim_middle(eigvals, percent)
    elif method == "iter":
        eigvals = trim_iteratively(eigvals)
    elif method == "none":
        pass

    if print_summary:
        print(f"Unique eigenvalues: {len(np.unique(eigvals))}")
        print(
            f"min: {eigvals.min()}, median: {np.median(eigvals)}, max: {eigvals.max()}"
        )
        print(f"mean: {np.mean(eigvals)}, var: {np.var(eigvals)}")

    return eigvals
