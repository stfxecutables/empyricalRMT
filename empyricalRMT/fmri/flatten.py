import nibabel as nib
import numpy as np

def flatten_4D(img4D: np.ndarray or nib.Nifti1Image) -> np.ndarray:
    if type(img4D) == np.ndarray:
        return img4D.reshape((np.prod(img4D.shape[0:-1]),) + (img4D.shape[-1],))
    else:
        arr = img4D.get_fdata()
        return arr.reshape((np.prod(arr.shape[0:-1]),) + (arr.shape[-1],))