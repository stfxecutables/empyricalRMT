import numpy as np
import pytest

from numpy import ndarray

from empyricalRMT.rmt.compare import Compare
from empyricalRMT.rmt.correlater import _compute_upper_correlations


@pytest.mark.math
@pytest.mark.fast
def test_correlate() -> None:
    n_curves = 2
    curves = [np.random.standard_normal(100) for _ in range(n_curves)]
    labels = [str(i) for i in range(n_curves)]
    compare = Compare(curves, labels)
    df = compare.correlate()

    # basic sanity / construction checks
    assert np.all(df.index == labels)
    assert np.all(df.columns == labels)

    curves = [np.random.standard_normal(100) for _ in range(1)]
    labels = [str(i) for i in range(n_curves)]

    # check that can't have less than 2 curves
    with pytest.raises(ValueError):
        compare = Compare(curves, labels)

    # check that labels length must match
    labels.pop()
    with pytest.raises(ValueError):
        compare = Compare(curves, labels)

    curves = [np.random.standard_normal(np.random.choice(100)) for _ in range(3)]
    labels = [str(i) for i in range(len(curves))]
    with pytest.raises(ValueError):
        compare = Compare(curves, labels).correlate()


@pytest.mark.fast
@pytest.mark.perf
def test_correlate_perf() -> None:

    import timeit

    n_comparisons = [10, 50, 200, 500]

    print("\n", "-" * 80)
    print("Execution times for numpy.corrcoef()")
    print("-" * 80)
    for n in n_comparisons:
        print(
            timeit.timeit(
                "numpy_correlate(curves)",
                setup=f"curves = setup({n});",
                globals=globals(),
                number=100,
            )
        )

    print("\n", "-" * 80)
    print("Execution times for _compute_upper_correlations()")
    print("-" * 80)
    for n in n_comparisons:
        print(
            timeit.timeit(
                f"data = np.zeros([{n}, {n}], dtype=float); _compute_upper_correlations(data, curves)",
                setup=f"curves = np.array(setup({n}), dtype=float);",
                globals=globals(),
                number=100,
            )
        )


def setup(n_curves: int = 2) -> ndarray:
    curves = [np.random.standard_normal(100) for _ in range(n_curves)]
    return curves


def numpy_correlate(curves: ndarray) -> ndarray:
    correlates = np.corrcoef(curves)
    correlates[np.tri(len(curves), dtype=bool)] = 0.0
    return correlates
