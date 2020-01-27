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

    def _evaluate_distances(
        sizes=[50, 100, 200, 500, 1000, 2000, 4000], reps=10, **kwargs
    ):
        log = []
        all_msqds = []
        for size in sizes:
            mads, msqds = [], []
            for i in range(reps):
                M = generateGOEMatrix(size)
                eigs = np.sort(np.linalg.eigvalsh(M))
                unfolded = Unfolder(eigs).unfold(trim=False, **kwargs)
                spacings = unfolded[1:] - unfolded[:-1]
                obs = _get_kde_values(spacings, 10000)
                exp = expected.GOE.spacing_distribution(unfolded, 10000)
                mad, msd = _mad(obs, exp), _msd(obs, exp)
                mads.append(mad), msqds.append(msd), all_msqds.append(msd)

            mean_mad, mean_msqd = np.mean(mads), np.mean(msqds)

            mad_z = mean_mad / np.std(mads, ddof=1)
            msqd_z = mean_msqd / np.std(msqds, ddof=1)
            mad_perc = np.percentile(mads, [5, 95])
            msqd_perc = np.percentile(msqds, [5, 95])
            msqds = np.array(msqds)
            msqds_below_threshold = len(msqds[msqds <= 0.01]) / len(msqds)

            log.append(f"\nDeviations for {size}x{size} GOE matrices:")
            log.append(
                "mean MAD:  {:06.5f}, [{:06.5f},{:06.5f}]: z == {:04.3f}".format(
                    mean_mad, mad_perc[0], mad_perc[1], mad_z
                )
            )
            log.append(
                "mean MSqD: {:06.5f}, [{:06.5f},{:06.5f}]: z == {:06.5f}".format(
                    mean_msqd, msqd_perc[0], msqd_perc[1], msqd_z
                )
            )
            log.append(
                "Percent identified as GOE: {:03.1f}".format(
                    100 * msqds_below_threshold
                )
            )
        all_msqds = np.array(all_msqds)
        all_msqds_below_threshold = len(all_msqds[all_msqds <= 0.01]) / len(all_msqds)
        log.append(f"{'-'*80}")
        log.append(
            "Percent identified as GOE across sizes: {:03.1f}".format(
                100 * all_msqds_below_threshold
            )
        )
        log.append(f"{'-'*80}")

        # compare to eigenvalues randomly selected from a uniform distribution
        all_msqds = []
        for size in sizes:
            mads, msqds = [], []
            for i in range(reps):
                eigs = np.sort(np.random.uniform(-1, 1, size))
                unfolded = Unfolder(eigs).unfold(trim=False, **kwargs)
                spacings = unfolded[1:] - unfolded[:-1]
                obs = _get_kde_values(spacings, 10000)
                exp = expected.GOE.spacing_distribution(unfolded, 10000)
                mad, msd = _mad(obs, exp), _msd(obs, exp)
                mads.append(mad), msqds.append(msd), all_msqds.append(msd)
            mean_mad, mean_msqd = np.mean(mads), np.mean(msqds)

            mad_z = mean_mad / np.std(mads, ddof=1)
            msqd_z = mean_msqd / np.std(msqds, ddof=1)
            mad_perc = np.percentile(mads, [5, 95])
            msqd_perc = np.percentile(msqds, [5, 95])
            msqds = np.array(msqds)
            msqds_below_threshold = len(msqds[msqds <= 0.01]) / len(msqds)

            log.append(f"\nDeviations for {size} eigenvalues from U(-1, 1):")
            log.append(
                "mean MAD:  {:06.5f}, [{:06.5f},{:06.5f}]: z == {:04.3f}".format(
                    mean_mad, mad_perc[0], mad_perc[1], mad_z
                )
            )
            log.append(
                "mean MSqD: {:06.5f}, [{:06.5f},{:06.5f}]: z == {:06.5f}".format(
                    mean_msqd, msqd_perc[0], msqd_perc[1], msqd_z
                )
            )
            log.append(
                "Percent identified as GOE: {:03.1f}".format(
                    100 * msqds_below_threshold
                )
            )
        all_msqds = np.array(all_msqds)
        all_msqds_below_threshold = len(all_msqds[all_msqds <= 0.01]) / len(all_msqds)
        log.append(f"{'-'*80}")
        log.append(
            "Percent random uniform eigenvalues identified as from GOE across sizes: {:03.1f}".format(
                100 * all_msqds_below_threshold
            )
        )
        log.append(f"{'-'*80}")

        # compare to eigenvalues randomly selected from a standard normal distribution
        all_msqds = []
        for size in sizes:
            mads, msqds = [], []
            for i in range(reps):
                eigs = np.sort(np.random.standard_normal(size))
                unfolded = Unfolder(eigs).unfold(trim=False, **kwargs)
                spacings = unfolded[1:] - unfolded[:-1]
                obs = _get_kde_values(spacings, 10000)
                exp = expected.GOE.spacing_distribution(unfolded, 10000)
                mad, msd = _mad(obs, exp), _msd(obs, exp)
                mads.append(mad), msqds.append(msd), all_msqds.append(msd)
            mean_mad, mean_msqd = np.mean(mads), np.mean(msqds)

            mad_z = mean_mad / np.std(mads, ddof=1)
            msqd_z = mean_msqd / np.std(msqds, ddof=1)
            mad_perc = np.percentile(mads, [5, 95])
            msqd_perc = np.percentile(msqds, [5, 95])
            msqds = np.array(msqds)
            msqds_below_threshold = len(msqds[msqds <= 0.01]) / len(msqds)

            log.append(f"\nDeviations for {size} eigenvalues from N(0, 1):")
            log.append(
                "mean MAD:  {:06.5f}, [{:06.5f},{:06.5f}]: z == {:04.3f}".format(
                    mean_mad, mad_perc[0], mad_perc[1], mad_z
                )
            )
            log.append(
                "mean MSqD: {:06.5f}, [{:06.5f},{:06.5f}]: z == {:06.5f}".format(
                    mean_msqd, msqd_perc[0], msqd_perc[1], msqd_z
                )
            )
            log.append(
                "Percent identified as GOE: {:03.1f}".format(
                    100 * msqds_below_threshold
                )
            )
        all_msqds = np.array(all_msqds)
        all_msqds_below_threshold = len(all_msqds[all_msqds <= 0.01]) / len(all_msqds)
        log.append(f"{'-'*80}")
        log.append(
            "Percent random standard normal eigenvalues identified as from GOE across sizes: {:03.1f}".format(
                100 * all_msqds_below_threshold
            )
        )
        log.append(f"{'-'*80}")

        return log

    degrees = [5, 7, 9, 11]
    for degree in degrees:
        with capsys.disabled():
            print(
                f"\n{'='*80}\nComputing distances for polynomial degree {degree}\n{'='*80}"
            )
            log = _evaluate_distances(degree=degree, reps=2)
            for line in log:
                print(line)
    for smoothing in np.linspace(1, 2, num=5):
        with capsys.disabled():
            print(
                f"\n{'='*80}\nComputing distances for splines with smoothing {smoothing}\n{'='*80}"
            )
            log = _evaluate_distances(
                degree=3, reps=2, smoother="spline", spline_smooth=smoothing
            )
            for line in log:
                print(line)
    with capsys.disabled():
        print(f"\n{'='*80}\nComputing distances for gompertz smoother\n{'='*80}")
        log = _evaluate_distances(smoother="gompertz", reps=2)
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

