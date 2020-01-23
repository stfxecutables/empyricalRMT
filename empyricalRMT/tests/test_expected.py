import numpy as np
import pytest

from scipy.stats import ks_2samp, kstest
from statsmodels.nonparametric.kde import KDEUnivariate as KDE

import empyricalRMT.rmt.expected as expected

from empyricalRMT.rmt.construct import generateGOEMatrix
from empyricalRMT.rmt.plot import spacings as plotSpacings
from empyricalRMT.rmt.unfold import Unfolder


@pytest.mark.plot
def test_GOE():
    for i in range(1):
        M = generateGOEMatrix(1000)
        eigs = np.sort(np.linalg.eigvals(M))
        unfolded = Unfolder(eigs).unfold(trim=False)
        plotSpacings(unfolded, bins=20, kde=True, mode="block")


@pytest.mark.expected
def test_nnsd_mad_msd(capsys):
    def _get_kde_values(spacings: np.array, n_points: int = 1000) -> np.array:
        spacings = np.sort(spacings)
        kde = KDE(spacings)
        kde.fit(kernel="gau", bw="scott", cut=0, fft=False, gridsize=n_points)
        s = np.linspace(spacings[0], spacings[-1], n_points)
        evaluated = np.empty_like(s)
        for i, _ in enumerate(evaluated):
            evaluated[i] = kde.evaluate(s[i])
        return evaluated

    def _mad(arr1, arr2):
        return np.mean(np.abs(arr1 - arr2))

    def _msd(arr1, arr2):
        return np.mean((arr1 - arr2) ** 2)

    def _evaluate_distances(sizes=[50, 100, 200, 500, 1000, 2000, 4000], **kwargs):
        log = []
        for size in sizes:
            mads, msqds = [], []
            for i in range(10):
                M = generateGOEMatrix(size)
                eigs = np.sort(np.linalg.eigvalsh(M))
                unfolded = Unfolder(eigs).unfold(trim=False, **kwargs)
                spacings = unfolded[1:] - unfolded[:-1]
                obs = _get_kde_values(spacings, 10000)
                exp = expected.GOE.spacing_distribution(unfolded, 10000)
                mad, msd = _mad(obs, exp), _msd(obs, exp)
                mads.append(mad), msqds.append(msd)
            mean_mad, mean_msqd = np.mean(mads), np.mean(msqds)

            log.append(f"\nDeviations for {size}x{size} GOE matrices:")
            log.append(f"mean MAD: {mean_mad}, ({np.percentile(mads, [5, 95])})")
            log.append(f"mean MSqD: {mean_msqd}, ({np.percentile(msqds, [5, 95])})")
            log.append(f"MAD z-score: {mean_mad / np.std(mads, ddof=1)}")
            log.append(f"MSqD z-score: {mean_msqd / np.std(msqds, ddof=1)}")

        for size in sizes:
            mads, msqds = [], []
            for i in range(10):
                eigs = np.sort(np.random.uniform(-1, 1, size))
                unfolded = Unfolder(eigs).unfold(trim=False, **kwargs)
                spacings = unfolded[1:] - unfolded[:-1]
                obs = _get_kde_values(spacings, 10000)
                exp = expected.GOE.spacing_distribution(unfolded, 10000)
                mad, msd = _mad(obs, exp), _msd(obs, exp)
                mads.append(mad), msqds.append(msd)
            mean_mad, mean_msqd = np.mean(mads), np.mean(msqds)

            log.append(f"\nDeviations for {size}x{size} Standard Uniform matrices:")
            log.append(f"\nDeviations for {size}x{size} square GOE matrices:")
            log.append(f"mean MAD: {mean_mad}, ({np.percentile(mads, [5, 95])})")
            log.append(f"mean MSqD: {mean_msqd}, ({np.percentile(msqds, [5, 95])})")
            log.append(f"MAD z-score: {mean_mad / np.std(mads, ddof=1)}")
            log.append(f"MSqD z-score: {mean_msqd / np.std(msqds, ddof=1)}")
        return log

    degrees = [5, 7, 9, 11]
    for degree in degrees:
        with capsys.disabled():
            print(
                f"\n{'='*80}\nComputing distances for polynomial degree {degree}\n{'='*80}"
            )
            log = _evaluate_distances(degree=degree)
            for line in log:
                print(line)
    for smoothing in np.linspace(1, 2, num=5):
        with capsys.disabled():
            print(
                f"\n{'='*80}\nComputing distances for splines with smoothing {smoothing}\n{'='*80}"
            )
            log = _evaluate_distances(
                degree=3, smoother="spline", spline_smooth=smoothing
            )
            for line in log:
                print(line)
    with capsys.disabled():
        print(
            f"\n{'='*80}\nComputing distances for gompertz function {smoothing}\n{'='*80}"
        )
        log = _evaluate_distances(smoother="gompertz")
        for line in log:
            print(line)


def test_nnsd_kolmogorov(capsys=None):
    """To compare an observed NNSD to the expected NNSDs, we should not only
    perform a two-sample Kolmogorov-Smirnov test, but also simply measure the distances
    between the observed and expected distribution values.

    However, we also know that, depending on the choice of unfolding, and matrix size,
    there are going to be different cutpoints for when an observed NNSD "looks like" a
    GOE. We also know that bad the approximators we are using to do the unfolding also
    have limited ability to result in the correct observables even for matrices we *know*
    to be GOE. So we need to generate some example GOE data and see what the KS tests and
    various mean distance functions look like.
    """
    raise NotImplementedError("Python kstest implementations are terrible")

    def get_kde_values(spacings: np.array, n_points: int = 1000) -> np.array:
        spacings = np.sort(spacings)
        kde = KDE(spacings)
        kde.fit(kernel="gau", bw="scott", cut=0, fft=False, gridsize=n_points)
        s = np.linspace(spacings[0], spacings[-1], n_points)
        # evaluated = np.empty_like(s)
        # for i, _ in enumerate(evaluated):
        #     evaluated[i] = kde.evaluate(s[i])
        # return evaluated
        return kde.cdf

    sizes = [50, 100, 200, 500, 1000, 2000]
    for size in sizes:
        ks_vals = []
        for i in range(10):
            M = generateGOEMatrix(size)
            eigs = np.sort(np.linalg.eigvals(M))
            unfolded = Unfolder(eigs).unfold(trim=False)
            spacings = unfolded[1:] - unfolded[:-1]
            # density_obs = get_kde_values(spacings, 1000)
            density_exp = expected.GOE.spacing_distribution(unfolded, 1000)
            # rvs = lambda: density_exp  # workaround for terrible implementation
            cdf = lambda x: get_kde_values(spacings)
            rvs = lambda **kwargs: spacings
            ks_vals.append(kstest(rvs, cdf, N=len(spacings)))
            # ks_vals.append(ks_2samp(density_obs, density_exp))
            # ks_vals.append(kstest(density_obs, density_exp))
        # with capsys.disabled():
        #     print(f"KS-test values observed for square GOE matrix of size {size}:")
        #     for val in ks_vals:
        #         print(val)

    pass

