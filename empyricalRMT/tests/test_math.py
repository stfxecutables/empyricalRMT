import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
import pytest
import seaborn as sbn

from scipy.integrate import trapz
from typing import Any, List, Tuple

import empyricalRMT.rmt.ensemble as ensemble

from empyricalRMT.rmt.observables.rigidity import slope, intercept, integrateFast


@pytest.mark.math
@pytest.mark.fast
def test_slope() -> None:
    for _ in range(1000):
        m = np.random.uniform(-10, 10)
        b = np.random.uniform(-10, 10)
        x = np.random.uniform(-1000, 1000, 1000)
        y = m * x + b
        m_comp = slope(x, y)
        b_comp = intercept(x, y, m_comp)
        assert np.allclose(m, m_comp)
        assert np.allclose(b, b_comp)


@pytest.mark.math
def test_integrate() -> None:
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
        int_comp = integrateFast(grid, y)
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
        int_comp = integrateFast(grid, y)
        int_exp = trapz(y, x=grid)
        assert np.abs(int_analytic - int_comp) < 0.001 * np.abs(int_analytic)
        assert np.allclose(int_comp, int_exp)


@pytest.mark.math
def test_integrate_perf() -> None:
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
        integrateFast(grid, y[i])
    total_custom = time.time() - start

    start = time.time()
    for i in range(n):
        trapz(y[i], x=grid)
    total_lib = time.time() - start

    # just make sure we are at least doing better than scipy
    assert total_custom < total_lib
    print("Custom integration time: ", total_custom)
    print("Scipy integration time: ", total_lib)

