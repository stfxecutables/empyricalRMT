import numpy as np
import pytest

from numpy import ndarray


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
