import numpy as np

from colorama import Fore, Style
from glob import glob
from pathlib import Path
from platform import system

from signalproc.detrend import linear_detrend, mean_detrend
from fmri.downsample import downsample_brain
from fmri.flatten import flatten_4D
from fmri.loadnii import load_nii
from signalproc.clean import get_signals
from rmt.correlater import compute_correlations
from rmt.eigenvalues import getEigs
from utils import make_directory, res, setup_progressbar

RESET = Style.RESET_ALL


def generate_correlations(
    bids_root: Path,
    regex: str,
    prefix: str = "corrmat",
    bids_outdir_name: str = "rmt",
    downsample_factor: float = None,
    detrend=False,
    dry_run=False,
):
    """compute and save correlation matrices for nii files matching regex
    e.g. generate_correlations(Path("ds000222-download"), "memorywords_bold.nii.gz", "corrmat")
    """
    func_regex = None
    if system() != "Windows":
        func_regex = f"{res(bids_root)}/**/sub*{regex}"
    else:
        func_regex = f"{res(bids_root)}\\**\\sub*{regex}"
    funcs = [Path(file) for file in glob(func_regex, recursive=True)]
    if len(funcs) == 0:
        raise Exception("Bad functional image regex. No functional images found")
    funcs.sort()
    print("Found files:\n", funcs)

    if dry_run:
        print(f"{Fore.GREEN}Found files:{RESET}")
        for func in funcs:
            print("\t", func)
        print(f"{Fore.YELLOW}Will output files to:{RESET}")
        for func in funcs:
            derivatives = bids_root / "derivatives"
            rmt_dir = derivatives / bids_outdir_name
            subj_dir = rmt_dir / find_subj_folder_name(mat).name
            if detrend:
                outfile = (
                    subj_dir / f"{prefix}_{func.stem.replace('.nii', '')}_detrended.npy"
                )
            else:
                outfile = subj_dir / f"{prefix}_{func.stem.replace('.nii', '')}.npy"
            print("\t", str(outfile.absolute()))
        return

    corrmats = []
    pbar = setup_progressbar("Computing group correlations", len(funcs)).start()
    for i, func in enumerate(funcs):
        img = None
        if downsample_factor is not None:
            pbar.update(i)
            img = downsample_brain(func, downsample_factor)
            pbar.update(i)
        else:
            pbar.update(i)
            img = load_nii(res(func))
            pbar.update(i)

        pbar.update(i)
        arr = flatten_4D(img)  # flatten
        pbar.update(i)
        arr = get_signals(arr)  # remove air / constant signals
        pbar.update(i)

        derivatives = bids_root / "derivatives"
        rmt_dir = derivatives / bids_outdir_name
        subj_dir = rmt_dir / find_subj_folder_name(func).name
        pbar.update(i)

        if detrend:
            pbar.update(i)
            ret = np.empty(arr.shape, dtype=np.float64)
            pbar.update(i)
            # linear_detrend(arr, ret)
            mean_detrend(arr, ret)
            outfile = (
                subj_dir / f"{prefix}_{func.stem.replace('.nii', '')}_detrended.npy"
            )
        else:
            outfile = subj_dir / f"{prefix}_{func.stem.replace('.nii', '')}.npy"
        make_directory(derivatives)
        make_directory(rmt_dir)
        make_directory(subj_dir)

        pbar.update(i)
        corrmat = compute_correlations(arr, outfile, detrend, i, len(funcs))
        pbar.update(i)
        corrmats.append(corrmat)

    pbar.finish(dirty=True)
    return corrmats


def generate_eigs(
    bids_root: Path,
    regex: str,
    corrmat_prefix: str = "corrmat",
    eig_prefix: str = "eigs",
    bids_outdir_name: str = "rmt",
    dry_run=False,
):
    """compute and save eigenvalues for nii files matching regex
    e.g. generate_eigs(Path("ds000222-download"), "memorywords_bold.nii.gz", "corrmat")
    """
    mat_regex = None
    if system() != "Windows":
        mat_regex = f"{res(bids_root)}/**/{corrmat_prefix}*{regex}"
    else:
        mat_regex = f"{res(bids_root)}\\**\\{corrmat_prefix}*{regex}"
    mats = [Path(file) for file in glob(mat_regex, recursive=True)]
    if len(mats) == 0:
        raise Exception("Bad matrix data regex. No saved .npy data files found")
    mats.sort()

    if dry_run:
        print(f"{Fore.GREEN}Found files:{RESET}")
        for mat in mats:
            print("\t", mat)
        print(f"{Fore.YELLOW}Will output files to:{RESET}")
        for mat in mats:
            derivatives = bids_root / "derivatives"
            rmt_dir = derivatives / bids_outdir_name
            subj_dir = rmt_dir / find_subj_folder_name(mat).name
            outfile = subj_dir / f"{eig_prefix}_{mat.stem}.npy".replace(
                f"_{corrmat_prefix}", ""
            )
            print("\t", str(outfile.absolute()))
        return

    eigs = []
    pbar = setup_progressbar("Computing eigenvalues", len(mats))
    pbar.init()
    for i, mat in enumerate(mats):
        derivatives = bids_root / "derivatives"
        rmt_dir = derivatives / bids_outdir_name
        subj_dir = rmt_dir / find_subj_folder_name(mat).name
        outfile = subj_dir / f"{eig_prefix}_{mat.stem}.npy".replace(
            f"_{corrmat_prefix}", ""
        )
        make_directory(derivatives)
        make_directory(rmt_dir)
        make_directory(subj_dir)
        pbar.update(i)
        arr = np.load(mat)
        pbar.update(i)
        eig = getEigs(arr)
        pbar.update(i)
        np.save(outfile, eig)
        pbar.update(i)
        eigs.append(eig)
        pbar.update(i)
    pbar.finish(dirty=True)

    return eigs


# keep looking up until you find the subject folder
def find_subj_folder_name(func: Path) -> str:
    path = func.parent  # don't start with the file itself
    while not path.name.lower().startswith("sub"):
        path = path.parent
    return path

