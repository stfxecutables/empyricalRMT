from pathlib import Path
from platform import system

from batch.plot import group_spacings

REST_PATH = None
TASK_PATH = None
REST_REG = "rest_bold.npy"
REST_REG = "rest_bold_detrended.npy"
TASK_REG = "memorywords_bold.npy"
TASK_REG = "memorywords_bold_detrended.npy"
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

group_spacings(
    bids_root=REST_PATH,
    bids_outdir_name="rmt",
    eig_prefix="eigs",
    group_regex=REST_REG,
    title="NNSD",
    group_label="Rest",
    outdir=Path("plots/spacings"),
    degree=5,
    grid_length=20000,
    trim_method="none",
    trim_percent=95,
    detrend=True,
    percent=None,
    block=True,
    dry_run=True,
)

# group_spacings(
#     bids_root=TASK_PATH,
#     bids_outdir_name="rmt",
#     eig_prefix="eigs",
#     group_regex=TASK_REG,
#     title="NNSD",
#     group_label="Memorywords",
#     outdir=Path("plots/spacings"),
#     degree=5,
#     grid_length=20000,
#     trim_method="none",
#     trim_percent=98,
#     detrend=True,
#     percent=None,
#     block=True,
# )
