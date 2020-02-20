import numpy as np
import pytest

from empyricalRMT.rmt.compare import Compare


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
        labels = [str(i) for i in range(n_curves)]
        compare = Compare(curves, labels)
        df = compare.correlate()

        # basic sanity / construction checks
        assert np.all(df.index == labels)
        assert np.all(df.columns == labels)

