import numpy as np
from numpy import ndarray
import os
import pandas as pd
import pytest

from pathlib import Path

from empyricalRMT.rmt.eigenvalues import Eigenvalues
from empyricalRMT.rmt.construct import generateGOEMatrix, generate_eigs
from empyricalRMT.rmt.trim import Trimmed, TrimReport
from empyricalRMT.rmt.unfolder import Unfolder


@pytest.mark.fast
@pytest.mark.trim
def test_init_sanity() -> None:
    for i in range(10):
        M = generateGOEMatrix(100)
        eigs = np.sort(np.linalg.eigvalsh(M))
        trim = TrimReport(eigs)
        assert np.allclose(trim._untrimmed, eigs)
        assert isinstance(trim._unfold_info, pd.DataFrame)
        assert isinstance(trim._all_unfolds, pd.DataFrame)
        assert isinstance(trim._trim_steps, list)
        assert isinstance(trim._trim_steps[0], pd.DataFrame)


@pytest.mark.fast
@pytest.mark.trim
def test_trim_manual() -> None:
    M = generateGOEMatrix(2000)
    eigs = np.linalg.eigvalsh(M)
    for i in range(20):
        m, n = np.sort(np.array(np.random.uniform(0, len(eigs), 2), dtype=int))
        raw_trimmed = np.copy(eigs[m:n])
        eigenvalues = Eigenvalues(eigs)
        trimmed = eigenvalues.trim_manually(m, n)
        assert np.allclose(raw_trimmed, trimmed.vals)


@pytest.mark.fast
@pytest.mark.trim
def test_trim_reports() -> None:
    M = generateGOEMatrix(2000, seed=0)
    eigs = np.linalg.eigvalsh(M)
    report = TrimReport(eigs)
    best_smoothers, best_unfolds, best_indices, consistent_smoothers = (
        report.summarize_trim_unfoldings()
    )
    assert np.array_equal(
        np.sort(consistent_smoothers), np.sort(["cubic-spline_1.0", "poly_6", "poly_5"])
    )
    assert np.array_equal(best_indices, [(0, 1895), (367, 1769), (232, 1769)])

    report.plot_trim_steps(mode="noblock")
