import numpy as np
import pytest
import time

from numpy import ndarray
from scipy.integrate import simps, trapz
from typing import Any

from empyricalRMT.rmt.observables.rigidity import (
    _slope,
    _intercept,
    _integrate_fast,
    _integrate_simpsons,
    _int_simps_nonunif,
)
from empyricalRMT.rmt.observables.step import (
    _step_function_correct,
    _step_function_fast,
    _step_function_slow,
)


@pytest.mark.math
@pytest.mark.fast
def test_step_fast() -> None:
    def is_correct(eigs: ndarray, vals: ndarray) -> Any:
        return np.allclose(
            np.array(_step_function_fast(eigs, vals), dtype=int),
            np.array(_step_function_correct(eigs, vals), dtype=int),
            atol=1e-5,
        )

    def is_close(eigs: ndarray, vals: ndarray) -> bool:
        computed = _step_function_fast(eigs, vals)
        correct = _step_function_correct(eigs, vals)
        diffs = np.sum(np.abs(computed - correct)) / len(vals)
        return bool(diffs < 1e-5)

    # readable cases
    reigs = np.array([-2, -1, 0, 1, 2], dtype=float)
    x = np.array([-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0])
    assert np.allclose(
        np.array(_step_function_fast(reigs, x), dtype=int),
        np.array([0, 0, 1, 1, 2, 2, 3], dtype=int),
    )
    assert is_correct(reigs, x)
    x = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    assert np.allclose(
        np.array(_step_function_fast(reigs, x), dtype=int),
        np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5], dtype=int),
    )
    assert is_correct(reigs, x)
    x = np.array([-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    assert np.allclose(
        np.array(_step_function_fast(reigs, x), dtype=int),
        np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5], dtype=int),
    )
    assert is_correct(reigs, x)
    # this input is causing a segfault
    x = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    assert np.allclose(
        np.array(_step_function_fast(reigs, x), dtype=int),
        np.array([3, 3, 4, 4, 5, 5], dtype=int),
    )
    assert is_correct(reigs, x)

    for _ in range(1000):
        eigs = np.sort(np.random.uniform(-1000, 1000, 1000))
        # for i in range(len(eigs) - 1):
        #     if np.allclose(eigs[i], eigs[i + 1]):
        #         raise ValueError("Non-unique eigenvalues!")

        # degenerate cases
        x_0 = np.linspace(eigs[-1] + 1000, eigs[-1] + 2000, 10000)
        x_1 = np.linspace(eigs[0] - 1000, eigs[0] - 2000, 10000)
        assert is_close(eigs, x_0)
        assert is_close(eigs, x_1)

        # differing overlaps
        x_2 = np.linspace(eigs[0], eigs[-1], 10000)
        x_3 = np.linspace(eigs[0] - 500, eigs[-1], 10000)
        x_4 = np.linspace(eigs[0] - 500, eigs[-1] + 500, 10000)
        x_5 = np.linspace(eigs[0], eigs[-1] + 500, 10000)
        assert is_close(eigs, x_2)
        assert is_close(eigs, x_3)
        assert is_close(eigs, x_4)
        assert is_close(eigs, x_5)


@pytest.mark.perf
def test_step_fast_perf() -> None:
    step_fasts, step_slows, step_corrects = [], [], []
    for _ in range(5):
        eigs = np.sort(np.random.uniform(-10000, 10000, 10000))
        x = np.linspace(eigs[0], eigs[-1], 50000)

        start = time.time()
        for _ in range(100):
            _step_function_fast(eigs, x)
        step_fast = time.time() - start

        start = time.time()
        for _ in range(100):
            _step_function_slow(eigs, x)
        step_slow = time.time() - start

        start = time.time()
        for _ in range(100):
            _step_function_correct(eigs, x)
        step_correct = time.time() - start

        step_fasts.append(step_fast)
        step_slows.append(step_slow)
        step_corrects.append(step_correct)

    print("Smaller values are better (seconds)")
    print(
        "_step_function_fast:       ",
        np.mean(step_fasts),
        "+-",
        3 * np.std(step_fasts, ddof=1),
    )
    print(
        "_step_function_slow: ",
        np.mean(step_slows),
        "+-",
        3 * np.std(step_slows, ddof=1),
    )
    print(
        "_step_function_correct:    ",
        np.mean(step_corrects),
        "+-",
        3 * np.std(step_corrects, ddof=1),
    )


@pytest.mark.math
@pytest.mark.fast
def test_slope() -> None:
    for _ in range(1000):
        m = np.random.uniform(-10, 10)
        b = np.random.uniform(-10, 10)
        x = np.random.uniform(-1000, 1000, 1000)
        y = m * x + b
        m_comp = _slope(x, y)
        b_comp = _intercept(x, y, m_comp)
        assert np.allclose(m, m_comp)
        assert np.allclose(b, b_comp)


@pytest.mark.math
def test_integrate_trapz() -> None:
    """Just some extremely non-rigorous but basic sanity checks."""
    # linear functions
    for _ in range(100):
        m = np.random.uniform(-10, 10)
        b = np.random.uniform(-10, 10)
        grid = np.sort(np.random.uniform(-1000, 1000, 1000))
        y = m * grid + b
        # m*x**2/2 + bx
        int_analytic = (m * grid[-1] ** 2 / 2 + b * grid[-1]) - (
            m * grid[0] ** 2 / 2 + b * grid[0]
        )
        int_comp = _integrate_fast(grid, y)
        int_exp = trapz(y, x=grid)
        assert np.allclose(int_analytic, int_exp)
        assert np.allclose(int_comp, int_exp)

    # quadratic functions
    for _ in range(100):
        a = np.random.uniform(-10, 10)
        b = np.random.uniform(-10, 10)
        c = np.random.uniform(-10, 10)
        grid = np.sort(np.random.uniform(-1000, 1000, 1000))
        y = a * grid ** 2 + b * grid + c
        f = lambda x: a / 3 * x ** 3 + b / 2 * x ** 2 + c * x  # noqa E731
        int_analytic = f(grid[-1]) - f(grid[0])
        int_comp = _integrate_fast(grid, y)
        int_exp = trapz(y, x=grid)
        assert np.abs(int_analytic - int_comp) < 0.001 * np.abs(int_analytic)
        assert np.allclose(int_comp, int_exp)


@pytest.mark.math
def test_integrate_simps() -> None:
    """Just some extremely non-rigorous but basic sanity checks."""
    # linear functions
    for _ in range(100):
        m = np.random.uniform(-10, 10)
        b = np.random.uniform(-10, 10)
        grid = np.linspace(-500, 500, 1001)  # must be uniform grid for simpsons
        y = m * grid + b
        # m*x**2/2 + bx
        int_analytic = (m * grid[-1] ** 2 / 2 + b * grid[-1]) - (
            m * grid[0] ** 2 / 2 + b * grid[0]
        )
        # int_comp = _integrate_simpsons(grid, y)
        int_comp = _int_simps_nonunif(grid, y)
        int_exp = simps(y, x=grid)
        print("Calculated via my simpsons: ", int_comp)
        print("Calculated via analytic: ", int_analytic)
        assert np.allclose(int_analytic, int_exp)
        assert np.allclose(int_comp, int_exp)

    # quadratic functions
    for _ in range(100):
        a = np.random.uniform(-10, 10)
        b = np.random.uniform(-10, 10)
        c = np.random.uniform(-10, 10)
        grid = np.linspace(-500, 500, 1001)
        y = a * grid ** 2 + b * grid + c
        f = lambda x: a / 3 * x ** 3 + b / 2 * x ** 2 + c * x  # noqa E731
        int_analytic = f(grid[-1]) - f(grid[0])
        int_comp = _int_simps_nonunif(grid, y)
        int_exp = simps(y, x=grid)
        print("Calculated via my simpsons: ", int_comp)
        print("Calculated via analytic: ", int_analytic)
        assert np.abs(int_analytic - int_comp) < 0.001 * np.abs(int_analytic)
        assert np.allclose(int_comp, int_exp)


@pytest.mark.perf
def test_integrate_perf_trapz() -> None:
    import time

    n = 10000
    m = np.random.uniform(-10, 10, n)
    b = np.random.uniform(-10, 10, n)
    grid = np.sort(np.random.uniform(-1000, 1000, 1000))
    y = np.empty([n, len(grid)])
    for i in range(n):
        y[i, :] = m[i] * grid + b[i]

    start = time.time()
    for i in range(n):
        _integrate_fast(grid, y[i])
    total_custom = time.time() - start

    start = time.time()
    for i in range(n):
        trapz(y[i], x=grid)
    total_lib = time.time() - start

    # just make sure we are at least doing better than scipy
    assert total_custom < total_lib
    print("Custom trapz integration time: ", total_custom)
    print("Scipy trapz integration time: ", total_lib)


@pytest.mark.perf
def test_integrate_perf_simps() -> None:
    import time

    n = 10000
    m = np.random.uniform(-10, 10, n)
    b = np.random.uniform(-10, 10, n)
    grid = np.sort(np.random.uniform(-1000, 1000, 1000))
    y = np.empty([n, len(grid)])
    for i in range(n):
        y[i, :] = m[i] * grid + b[i]

    start = time.time()
    for i in range(n):
        _int_simps_nonunif(grid, y[i])
    total_custom = time.time() - start

    start = time.time()
    for i in range(n):
        simps(y[i], x=grid)
    total_lib = time.time() - start

    # just make sure we are at least doing better than scipy
    assert total_custom < total_lib
    print("Custom simps integration time: ", total_custom)
    print("Scipy simps integration time: ", total_lib)
