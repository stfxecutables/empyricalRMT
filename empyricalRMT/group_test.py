from pathlib import Path
from platform import system

from fmri.preprocess import generate_correlations, generate_eigs

REST_PATH = None
TASK_PATH = None
REST_REG = "**/*ses-func03*/**/*rest_bold.nii.gz*"
TASK_REG = "**/*ses-func03*/**/*memorywords_bold.nii.gz*"
if system() == "Linux":
    REST_PATH = Path("/home/derek/Desktop/fMRI_Data/Rest+Various/ds000224-download")
    TASK_PATH = Path("/home/derek/Desktop/fMRI_Data/Rest+Various/ds000224-download")
elif system() == "Windows":
    REST_PATH = None
    TASK_PATH = None
elif system() == "Darwin" or system() == "MacOS":
    REST_PATH = Path("/Users/derek/Desktop/Rest+Various/ds000224-download")
    TASK_PATH = Path("/Users/derek/Desktop/Rest+Various/ds000224-download")

# generate_correlations(
#     bids_root=TASK_PATH,
#     regex=TASK_REG,
#     prefix="corrmat",
#     bids_outdir_name="rmt",
#     downsample_factor=0.5,
#     detrend=True,
# )

# generate_correlations(
#     bids_root=REST_PATH,
#     regex=REST_REG,
#     prefix="corrmat",
#     bids_outdir_name="rmt",
#     downsample_factor=0.5,
#     detrend=True,
# )


generate_eigs(
    bids_root=REST_PATH,
    regex="detrended.npy",
    corrmat_prefix="corrmat",
    eig_prefix="eigs",
    bids_outdir_name="rmt",
    dry_run=False,
)
