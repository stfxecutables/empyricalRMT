import nibabel as nib
import numpy as np

from nilearn.image import resample_img
from pathlib import Path
from scipy.ndimage import zoom

from utils import make_cheaty_nii, res

def downsample_brain(func: Path, factor: float = 0.5) -> nib.Nifti1Image:
    img = nib.load(res(func))
    # new_3d_shape = np.array(img.shape[0:-1])
    # new_3d_shape = np.floor(new_3d_shape * factor).astype(int)
    # new_shape = new_3d_shape + img.shape[-1]
    img_data = img.get_fdata()
    # print("Starting rescaling")
    rescaled = zoom(img_data, (factor, factor, factor, 1))
    # print("Done rescaling")
    return make_cheaty_nii(img, rescaled)

def downsample_brain_to_size(func: Path, shape: (int, int, int)) -> nib.Nifti1Image:
    img = nib.load(res(func))
    old_shape = img.shape[0:-1]
    factor = [0, 0, 0, 1]
    for i, (old, new) in enumerate(zip(old_shape, shape)):
        factor[i] = new / old
    img_data = img.get_fdata()

    print(f"Rescaling to factor {tuple(factor)}")
    rescaled = zoom(img_data, tuple(factor))
    print("Done rescaling")
    return make_cheaty_nii(img, rescaled)

    # return resample_img(img, target_affine=None, target_shape=new_shape, copy=True)

def estimate_downsample_factor(func: Path) -> nib.Nifti1Image:
    img = nib.load(res(func))
    data = img.get_fdata()
    shape = np.array(data.shape[0:-1])

    # TODO
    factor = 1.0
