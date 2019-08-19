import gc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import platform
import seaborn as sbn

from numba import jit, prange, typeof, cuda
from os import path
from pathlib import Path
from scipy.stats import pearsonr
from tempfile import mkdtemp, TemporaryFile

import rmt.unfold as unfold

from .loadnii import load_nii
from rmt.construct import getEigs
from rmt.observables.levelvariance import sigma_squared_exhaustive
from rmt.observables.rigidity import spectralRigidity
from rmt.observables.spacings import computeSpacings
from rmt.plot import setup_plotting
from test import test_spline_unfold
from utils import eprint, log, is_symmetric


setup_plotting()

NORM_SUBJ_1: np.array = load_nii("fmri/sampledata/sub-cntrl03_task-rest_bold.nii.gz")
NORM_SUBJ_2: np.array = load_nii("fmri/sampledata/sub-cntrl04_task-rest_bold.nii.gz")
PD_SUBJ_1: np.array   = load_nii("fmri/sampledata/sub-pddn03_task-rest_bold.nii.gz")
PD_SUBJ_2: np.array   = load_nii("fmri/sampledata/sub-pddn04_task-rest_bold.nii.gz")

HOME = Path.home()
OUTDIR = Path("Desktop")
OUTFILE = Path("reshapedFMRI.npy")
OUT = Path.joinpath(HOME, OUTDIR, OUTFILE)

TEMPCORRS = path.join(mkdtemp(), 'corrstemp.dat')
TEMPEIGS = path.join(mkdtemp(), 'eigstemp.dat')


def get_subj_corrmat(subj: np.array = NORM_SUBJ_1, subsample=True, sample_size: int = 5000):
    print(subj.shape)
    # get value for resizing
    # e.g. if data has shape (16, 16, 32, 300), we want 16*16*32
    # see e.g. https://bic-berkeley.github.io/psych-214-fall-2016/reshape_and_4d.html
    voxel_count = np.prod(subj.shape[0:-1])

    # An array of all the time series
    # first axis is the voxels, second is time
    # e.g. at each i, reshaped[i,:] is the BOLD time-series
    reshaped: np.array = subj.reshape((voxel_count, subj.shape[-1]))
    print("reshaped.shape", reshaped.shape)
    np.save(OUT, reshaped)

    # take a random subsample that is more manageable
    if subsample:
        np.random.shuffle(reshaped)
        reshaped = reshaped[0:sample_size, :]
        print(reshaped.dtype)  # should be float64
        gc.collect()  # free up memory

    # compute correlations
    # (!!), inefficient, upper triangle is sufficient
    iters = sample_size if subsample else voxel_count
    corrs = None
    if subsample is True:
        corrs = compute_corrs(reshaped, iters)
    else:
        # use memory maps
        corrs = np.memmap(TEMPCORRS, dtype="float32", mode="w+", shape=(iters, iters))
        print("corrs: ", corrs)
        for i in range(iters):
            compute_column_corrs(corrs, reshaped, i)

            columns_done_percent = 100 * (i + 1) / voxel_count
            print("Computed ", columns_done_percent, " of column correlations")

        corrs[:] = corrs + corrs.T + np.identity(voxel_count)

    # sample_series = pd.DataFrame({"BOLD": subj[16,16,16,:]})
    # sbn.relplot(data=sample_series, kind="line")
    # plt.show()
    return corrs


# @jit(nopython=True, cache=False, parallel=True, fastmath=True)
@jit(nopython=True, cache=False, parallel=True)
def compute_corrs(reshaped, iters):
    # compute correlations
    # (!!), inefficient, upper triangle is sufficient
    corrs = np.empty((iters, iters), dtype=np.float64)

    for i in prange(iters):
        print("Computing correlation matrix column ", i)
        for j in prange(iters):
            if i >= j:  # ignore lower triangle and diagonal
                corrs[i, j] = 0
                continue
            else:
                corr_mat = np.corrcoef(reshaped[i, :], reshaped[j, :])
                corrs[i, j] = corr_mat[1,0]
                # R = np.correlate(reshaped[i, :], reshaped[j, :])[0]
                # corrs[i, j] = R

    corrs = corrs + corrs.T + np.identity(iters)
    return corrs


@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def compute_column_corrs(corrs, reshaped, i):
    # compute correlations
    # (!!), inefficient, upper triangle is sufficient
    voxel_count = reshaped.shape[0]

    for j in prange(voxel_count):
        if i >= j:  # ignore lower triangle and diagonal
            corrs[i, j] = 0
        else:
            corr_mat = np.corrcoef(reshaped[i, :], reshaped[j, :])
            corrs[i, j] = corr_mat[1,0]
            # corrs[i, j] = np.correlate(reshaped[i, :], reshaped[j, :])[0]


def show_RMT_summary(corrmat: np.array, unfold_method="spline"):
    test_spline_unfold(corrmat, knots=11, percent=98, detrend=True, raws=True)
