from pathlib import Path
from platform import system

from batch.plot import group_eig_dists

REST_PATH = None
TASK_PATH = None
REST_REG = "rest_bold.npy"
TASK_REG = "memorywords_bold.npy"
if system() == "Linux":
    REST_PATH = Path("/home/derek/Desktop/fMRI_Data/Rest+Various/ds000224-download")
    TASK_PATH = Path("/home/derek/Desktop/fMRI_Data/Rest+Various/ds000224-download")
elif system() == "Windows":
    REST_PATH = None
    TASK_PATH = None
elif system() == "Darwin" or system() == "MacOS":
    REST_PATH = Path("/Users/derek/Desktop/Rest+Various/ds000224-download")
    TASK_PATH = Path("/Users/derek/Desktop/Rest+Various/ds000224-download")
print("Resting state regex: ", REST_REG)
print("REsting state path: ", REST_PATH)

group_eig_dists(
    bids_root=REST_PATH,
    bids_outdir_name="rmt",
    eig_prefix="eigs",
    group_regex=REST_REG,
    group_label="Rest",
    outdir=Path("plots/eigdists"),
    bins="doane",
    trim_method="none",
    block=True,
)

group_eig_dists(
    bids_root=TASK_PATH,
    bids_outdir_name="rmt",
    eig_prefix="eigs",
    group_regex=TASK_REG,
    group_label="Memorywords",
    outdir=Path("plots/eigdists"),
    bins="doane",
    trim_method="none",
    block=True,
)
