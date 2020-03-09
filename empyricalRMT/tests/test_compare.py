import numpy as np
import pytest

from typing import List

from empyricalRMT.compare import Compare, Metric
from empyricalRMT.construct import generate_eigs
from empyricalRMT.eigenvalues import Eigenvalues
from empyricalRMT.ensemble import GOE, GDE


@pytest.mark.fast
def test_validate() -> None:
    for n_curves in range(2, 7):
        n_curves = 2
        curves = [np.random.standard_normal(100) for _ in range(n_curves)]
        labels = [str(i) for i in range(n_curves)]
        compare = Compare(curves, labels)
        compare._test_validate()

    curves = [np.random.standard_normal(100) for _ in range(1)]
    labels = [str(i) for i in range(len(curves))]

    # check that can't have less than 2 curves
    with pytest.raises(ValueError):
        compare = Compare(curves, labels)

    # check that labels length must match
    labels.pop()
    with pytest.raises(ValueError):
        compare = Compare(curves, labels)

    # check that curves must be consistent
    sizes = [3, 4, 5]
    curves = [np.random.standard_normal(size) for size in sizes]
    labels = [str(i) for i in range(len(curves))]
    with pytest.raises(ValueError):
        Compare(curves, labels)._test_validate(check_all_equal=True)


@pytest.mark.math
@pytest.mark.fast
def test_correlate() -> None:
    for n_curves in range(2, 7):
        n_curves = 2
        curves = [np.random.standard_normal(100) for _ in range(n_curves)]
        labels = ["label" + str(i) for i in range(n_curves)]
        compare = Compare(curves, labels)
        n = len(labels)
        df = compare.correlate()

        # basic sanity / construction checks
        assert np.all(df.index == labels)
        assert np.all(df.columns == labels)
        assert df.shape == (n, n)

    for n_curves in range(3, 7):
        curves = [np.random.standard_normal(100) for _ in range(n_curves)]
        labels = ["label" + str(i) for i in range(n_curves)]
        base_curve = np.random.standard_cauchy(100)
        base_label = "base"
        compare = Compare(curves, labels, base_curve, base_label)
        n = len(labels)
        df = compare.correlate()

        # basic sanity / construction checks
        assert np.all(df.index == labels)
        assert np.all(df.columns == base_label)
        assert df.shape == (n, 1)


# The below might generate junk warnings about binary incompatibilities. This seems to be
# related to https://github.com/numpy/numpy/issues/14920, and is likely an incompatibility
# between the numpy version used for this library and the numpy version used internally in
# numba, as the warning only arises when using numba. Hopefully, if future numpy updates
# causes breaking changes, this warning will become an actual error.
@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
@pytest.mark.math
@pytest.mark.fast
def test_msqd() -> None:
    for n_curves in range(2, 7):
        n_curves = 2
        curves = [np.random.standard_normal(100) for _ in range(n_curves)]
        labels = ["label" + str(i) for i in range(n_curves)]
        compare = Compare(curves, labels)
        n = len(labels)
        df = compare.mean_sq_difference()

        # basic sanity / construction checks
        assert np.all(df.index == labels)
        assert np.all(df.columns == labels)
        assert df.shape == (n, n)

    for n_curves in range(3, 7):
        curves = [np.random.standard_normal(100) for _ in range(n_curves)]
        labels = ["label" + str(i) for i in range(n_curves)]
        base_curve = np.random.standard_cauchy(100)
        base_label = "base"
        compare = Compare(curves, labels, base_curve, base_label)
        n = len(labels)
        df = compare.mean_sq_difference()

        # basic sanity / construction checks
        assert np.all(df.index == labels)
        assert np.all(df.columns == base_label)
        assert df.shape == (n, 1)

    # very trivial correctness checks
    curves = [[1, 1, 1], [2, 2, 2]]
    labels = ["ones", "twos"]
    compare = Compare(curves, labels)
    df = compare.mean_sq_difference()
    assert df["ones"]["twos"] == 1.0
    assert df["twos"]["ones"] == 1.0
    assert df["ones"]["ones"] == 0.0
    assert df["twos"]["twos"] == 0.0

    curves = [[1, 1, 1], [3, 3, 3]]
    labels = ["ones", "threes"]
    compare = Compare(curves, labels)
    df = compare.mean_sq_difference()
    assert df["ones"]["threes"] == 4.0
    assert df["threes"]["ones"] == 4.0
    assert df["ones"]["ones"] == 0.0
    assert df["threes"]["threes"] == 0.0


@pytest.mark.math
@pytest.mark.fast
def test_mad() -> None:
    for n_curves in range(2, 7):
        n_curves = 2
        curves = [np.random.standard_normal(100) for _ in range(n_curves)]
        labels = ["label" + str(i) for i in range(n_curves)]
        compare = Compare(curves, labels)
        n = len(labels)
        df = compare.mean_abs_difference()

        # basic sanity / construction checks
        assert np.all(df.index == labels)
        assert np.all(df.columns == labels)
        assert df.shape == (n, n)

    for n_curves in range(3, 7):
        curves = [np.random.standard_normal(100) for _ in range(n_curves)]
        labels = ["label" + str(i) for i in range(n_curves)]
        base_curve = np.random.standard_cauchy(100)
        base_label = "base"
        compare = Compare(curves, labels, base_curve, base_label)
        n = len(labels)
        df = compare.mean_abs_difference()

        # basic sanity / construction checks
        assert np.all(df.index == labels)
        assert np.all(df.columns == base_label)
        assert df.shape == (n, 1)

    curves = [[1, 1, 1], [2, 2, 2]]
    labels = ["ones", "twos"]
    compare = Compare(curves, labels)
    df = compare.mean_abs_difference()
    assert df["ones"]["twos"] == 1.0
    assert df["twos"]["ones"] == 1.0
    assert df["ones"]["ones"] == 0.0
    assert df["twos"]["twos"] == 0.0

    curves = [[1, 1, 1], [3, 3, 3]]
    labels = ["ones", "threes"]
    compare = Compare(curves, labels)
    df = compare.mean_abs_difference()
    assert df["ones"]["threes"] == 2.0
    assert df["threes"]["ones"] == 2.0
    assert df["ones"]["ones"] == 0.0
    assert df["threes"]["threes"] == 0.0


@pytest.mark.slow
def test_unfold_compare() -> None:
    metrics: List[Metric] = ["msqd", "mad", "corr"]
    print("\n")
    print("=" * 80)
    print("Comparing a GOE matrix to GOE")
    print("=" * 80)
    eigs = Eigenvalues(generate_eigs(2000, seed=2))
    unfolded = eigs.unfold(degree=13)
    df = unfolded.ensemble_compare(ensemble=GOE, metrics=metrics, show_progress=True)
    print(df)

    print("\n")
    print("=" * 80)
    print("Comparing a Poisson / GDE matrix to GOE")
    print("=" * 80)
    eigs = Eigenvalues(generate_eigs(2000, kind="poisson", seed=2))
    unfolded = eigs.unfold(degree=13)
    df = unfolded.ensemble_compare(ensemble=GOE, metrics=metrics, show_progress=True)
    print(df)

    print("\n")
    print("=" * 80)
    print("Comparing a Poisson / GDE matrix to GOE")
    print("=" * 80)
    eigs = Eigenvalues(generate_eigs(2000, kind="poisson", seed=2))
    unfolded = eigs.unfold(degree=13)
    df = unfolded.ensemble_compare(ensemble=GOE, metrics=metrics, show_progress=True)
    print(df)

    print("\n")
    print("=" * 80)
    print("Comparing a Poisson / GDE matrix to Poisson / GDE")
    print("=" * 80)
    eigs = Eigenvalues(generate_eigs(2000, kind="poisson", seed=2))
    unfolded = eigs.unfold(degree=13)
    df = unfolded.ensemble_compare(ensemble=GDE, metrics=metrics, show_progress=True)
    print(df)
