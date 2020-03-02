import numpy as np
import pandas as pd
import pytest

from empyricalRMT.rmt.eigenvalues import Eigenvalues
from empyricalRMT.rmt.construct import generate_eigs
from empyricalRMT.rmt.trim import TrimReport


@pytest.mark.fast
@pytest.mark.trim
def test_init_sanity() -> None:
    for i in range(10):
        vals = generate_eigs(100)
        trim = TrimReport(vals, poly_degrees=[5, 7], max_iters=5)
        assert np.allclose(trim._untrimmed, vals)
        assert isinstance(trim.unfold_info, pd.DataFrame)
        assert isinstance(trim.unfoldings, list)
        assert isinstance(trim.unfoldings[0], pd.DataFrame)
        assert isinstance(trim._trim_steps, list)
        assert isinstance(trim._trim_steps[0], pd.DataFrame)


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
    eigs = generate_eigs(2000, seed=2)
    report = TrimReport(eigs)
    best_smoothers, best_unfolds, best_indices, consistent_smoothers = (
        report.summarize_trim_unfoldings()
    )
    assert np.array_equal(
        np.sort(consistent_smoothers), np.sort(["poly_7", "poly_8", "poly_9"])
    )
    assert np.array_equal(best_indices, [(104, 1765), (231, 1765), (104, 2000)])

    report.plot_trim_steps(mode="test")


# test_trim_reports()
