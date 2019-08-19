from pathlib import Path

from fmri.downsample import downsample_brain_to_size


# test_semicircle()
# test_spectral_rigidity(20000, unfold_degree=3)
# test_levelvariance()

FUNC = Path(
    "/home/derek/Desktop/fMRI_Data/Rest+Various/ds000224-download/sub-MSC01/"
    "ses-func03/func/sub-MSC01_ses-func03_task-rest_bold.nii.gz"
)

downsample_brain_to_size(FUNC, (40, 40, 40))
