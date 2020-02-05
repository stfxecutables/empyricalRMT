import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
import pytest
import seaborn as sbn

from scipy.stats import ks_2samp, kstest
from statsmodels.nonparametric.kde import KDEUnivariate as KDE
from typing import Any, List, Tuple

import empyricalRMT.rmt.ensemble as ensemble

from empyricalRMT.rmt.construct import generateGOEMatrix
from empyricalRMT.rmt.plot import spacings as plotSpacings
from empyricalRMT.rmt.unfolder import Unfolder


@pytest.mark.plot
@pytest.mark.fast
def test_GOE() -> None:
    for i in range(1):
        M = generateGOEMatrix(1000)
        eigs = np.sort(np.linalg.eigvals(M))
        unfolded = Unfolder(eigs).unfold(trim=False)
        plotSpacings(
            unfolded,
            title="Generate GOE spacing plot test",
            bins=20,
            kde=True,
            mode="block",
        )


@pytest.mark.expected
@pytest.mark.slow
def test_nnsd_mad_msd(capsys) -> None:  # type: ignore
    def _get_kde_values(spacings: np.array, n_points: int = 1000) -> np.array:
        spacings = np.sort(spacings)
        kde = KDE(spacings)
        kde.fit(kernel="gau", bw="scott", cut=0, fft=False, gridsize=n_points)
        s = np.linspace(spacings[0], spacings[-1], n_points)
        evaluated = np.empty_like(s)
        for i, _ in enumerate(evaluated):
            evaluated[i] = kde.evaluate(s[i])
        return evaluated

    def _mad(arr1: ndarray, arr2: ndarray) -> Any:
        return np.mean(np.abs(arr1 - arr2))

    def _msd(arr1: ndarray, arr2: ndarray) -> Any:
        return np.mean((arr1 - arr2) ** 2)

    def _evaluate_distances(
        sizes: List[int] = [50, 100, 200, 500, 1000, 2000, 4000],
        reps: int = 10,
        **kwargs: Any,
    ) -> List[str]:
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
                exp = ensemble.GOE.spacing_distribution(unfolded, 10000)
                mad, msd = _mad(obs, exp), _msd(obs, exp)
                mads.append(mad), msqds.append(msd), all_msqds.append(  # type: ignore
                    msd
                )

            mean_mad, mean_msqd = np.mean(mads), np.mean(msqds)

            mad_z = mean_mad / np.std(mads, ddof=1)
            msqd_z = mean_msqd / np.std(msqds, ddof=1)
            mad_perc = np.percentile(mads, [5, 95])
            msqd_perc = np.percentile(msqds, [5, 95])
            msqds = np.array(msqds)
            msqds_below_threshold = len(msqds[msqds <= 0.01]) / len(  # type: ignore
                msqds
            )

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
        all_msqds_below_threshold = len(
            all_msqds[all_msqds <= 0.01]  # type: ignore
        ) / len(all_msqds)
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
                exp = ensemble.GOE.spacing_distribution(unfolded, 10000)
                mad, msd = _mad(obs, exp), _msd(obs, exp)
                mads.append(mad), msqds.append(msd), all_msqds.append(  # type: ignore
                    msd
                )
            mean_mad, mean_msqd = np.mean(mads), np.mean(msqds)

            mad_z = mean_mad / np.std(mads, ddof=1)
            msqd_z = mean_msqd / np.std(msqds, ddof=1)
            mad_perc = np.percentile(mads, [5, 95])
            msqd_perc = np.percentile(msqds, [5, 95])
            msqds = np.array(msqds)
            msqds_below_threshold = len(msqds[msqds <= 0.01]) / len(  # type: ignore
                msqds
            )

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
        all_msqds_below_threshold = len(
            all_msqds[all_msqds <= 0.01]  # type: ignore
        ) / len(all_msqds)
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
                exp = ensemble.GOE.spacing_distribution(unfolded, 10000)
                mad, msd = _mad(obs, exp), _msd(obs, exp)
                mads.append(mad), msqds.append(msd), all_msqds.append(  # type: ignore
                    msd
                )
            mean_mad, mean_msqd = np.mean(mads), np.mean(msqds)

            mad_z = mean_mad / np.std(mads, ddof=1)
            msqd_z = mean_msqd / np.std(msqds, ddof=1)
            mad_perc = np.percentile(mads, [5, 95])
            msqd_perc = np.percentile(msqds, [5, 95])
            msqds = np.array(msqds)
            msqds_below_threshold = len(
                msqds[msqds <= 0.01]  # type: ignore
            ) / len(
                msqds
            )

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
        all_msqds_below_threshold = len(
            all_msqds[all_msqds <= 0.01]  # type: ignore
        ) / len(all_msqds)
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


@pytest.mark.slow
def test_nnsd_kolmogorov(capsys: Any) -> None:
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
    sizes = [50, 100, 200, 500, 1000, 2000, 4000]
    goe_eigs = [np.sort(np.linalg.eigvalsh(generateGOEMatrix(size))) for size in sizes]
    goe_unfolded = [Unfolder(eigs).unfold(trim=False) for eigs in goe_eigs]
    goe_spacings = np.array([unfolded[1:] - unfolded[:-1] for unfolded in goe_unfolded])

    def _compute_ks(reps: int = 10, **unfold_kwargs: Any) -> List[str]:
        log = []
        for i, size in enumerate(sizes):
            # compare to uniformly distributed eigenvalues
            stats, p_vals = [], []
            for _ in range(reps):
                eigs = np.sort(np.random.uniform(-1, 1, size))
                unfolded = Unfolder(eigs).unfold(trim=False, **unfold_kwargs)
                compare_spacings = unfolded[1:] - unfolded[:-1]
                goe = goe_spacings[i]
                D, p_val = ks_2samp(compare_spacings, goe)
                stats.append(D), p_vals.append(1000 * p_val)  # type: ignore
                plotSpacings(
                    unfolded,
                    bins=20,
                    kde=True,
                    title=f"{size} uniformly-distributed eigens",
                )
            mean_d, mean_p = np.mean(stats), np.mean(p_vals)
            d_perc = np.percentile(stats, [5, 95])
            p_perc = np.percentile(p_vals, [5, 95])
            log.append(
                f"\nComparing {size} U(-1, 1) eigenvalues to ({size}x{size}) GOE spacings"
            )
            log.append(
                "Mean D_n:   {:05.3f} [{:05.3f},{:05.3f}]".format(
                    mean_d, d_perc[0], d_perc[1]
                )
            )
            log.append(
                "Mean p-val (x1000): {:05.3f} [{:05.3f},{:05.3f}]".format(
                    mean_p, p_perc[0], p_perc[1]
                )
            )

        for i, size in enumerate(sizes):
            # compare to standard normally distributed eigenvalues (poisson)
            stats, p_vals = [], []
            for _ in range(reps):
                eigs = np.sort(np.random.standard_normal(size))
                unfolded = Unfolder(eigs).unfold(trim=False, **unfold_kwargs)
                compare_spacings = unfolded[1:] - unfolded[:-1]
                goe = goe_spacings[i]
                D, p_val = ks_2samp(compare_spacings, goe)
                stats.append(D), p_vals.append(p_val * 1000)  # type: ignore
                plotSpacings(
                    unfolded,
                    bins=20,
                    kde=True,
                    title=f"{size} normally-distributed eigens",
                )

            mean_d, mean_p = np.mean(stats), np.mean(p_vals)
            d_perc = np.percentile(stats, [5, 95])
            p_perc = np.percentile(p_vals, [5, 95])
            log.append(
                f"\nComparing {size} N(0, 1) eigenvalues to ({size}x{size}) GOE spacings"
            )
            log.append(
                "Mean D_n:   {:05.3f} [{:05.3f},{:05.3f}]".format(
                    mean_d, d_perc[0], d_perc[1]
                )
            )
            log.append(
                "Mean p-val (x1000): {:05.3f} [{:05.3f},{:05.3f}]".format(
                    mean_p, p_perc[0], p_perc[1]
                )
            )
        return log

    degrees = [5, 7, 9, 11]
    reps = 2
    for degree in degrees:
        with capsys.disabled():
            print(
                f"\n{'='*80}\nComputing KS-tests for polynomial degree {degree}\n{'='*80}"
            )
            log = _compute_ks(degree=degree, reps=reps)
            for line in log:
                print(line)
    for smoothing in np.linspace(1, 2, num=5):
        with capsys.disabled():
            print(
                f"\n{'='*80}\nComputing KS-tests for splines with smoothing {smoothing}\n{'='*80}"
            )
            log = _compute_ks(
                degree=3, reps=reps, smoother="spline", spline_smooth=smoothing
            )
            for line in log:
                print(line)
    with capsys.disabled():
        print(f"\n{'='*80}\nComputing KS-tests for gompertz smoother\n{'='*80}")
        log = _compute_ks(smoother="gompertz", reps=reps)
        for line in log:
            print(line)
