import numpy as np
import pandas as pd
import pytest

from pathlib import Path

from empyricalRMT.eigenvalues import Eigenvalues
from empyricalRMT.construct import generate_eigs
from empyricalRMT.trim import TrimIter


@pytest.mark.fast
@pytest.mark.trim
def test_init_sanity() -> None:
    eigs = Eigenvalues(generate_eigs(1000))
    report = eigs.trim_report(
        max_iters=9,
        poly_degrees=[5, 7, 9],
        spline_degrees=[],
        spline_smooths=[],
        show_progress=True,
    )
    assert np.allclose(report._untrimmed, eigs.original_eigenvalues)
    assert isinstance(report.summary, pd.DataFrame)
    assert isinstance(report._trim_iters, list)
    assert isinstance(report._trim_iters[0], TrimIter)
    path = Path(".") / "trim_report.csv"
    report.to_csv(path)
    assert path.exists()
    path.unlink()
    report.plot_trim_steps(mode="test")


@pytest.mark.fast
@pytest.mark.trim
def test_trim_manual() -> None:
    vals = generate_eigs(2000)
    for i in range(20):
        m, n = np.sort(np.array(np.random.uniform(0, len(vals), 2), dtype=int))
        raw_trimmed = np.copy(vals[m:n])
        eigenvalues = Eigenvalues(vals)
        trimmed = eigenvalues.trim_manually(m, n)
        assert np.allclose(raw_trimmed, trimmed.vals)


@pytest.mark.fast
@pytest.mark.trim
def test_trim_reports() -> None:
    eigs = Eigenvalues(generate_eigs(2000, seed=2))
    report = eigs.trim_report()
    best_smoothers, best_unfolds, best_indices, consistent_smoothers = (
        report.best_overall()
    )
    assert np.array_equal(
        np.sort(consistent_smoothers), np.sort(["poly_7", "poly_8", "poly_9"])
    )
    assert np.array_equal(best_indices, [(104, 1765), (231, 1765), (104, 2000)])

    report.plot_trim_steps(mode="test")
