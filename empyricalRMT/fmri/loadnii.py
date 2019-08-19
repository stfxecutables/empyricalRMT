import nibabel as nib
import numpy as np


def load_nii(path: str) -> np.array:
    img = nib.load(path)
    raw_data = img.get_fdata()
    return raw_data
