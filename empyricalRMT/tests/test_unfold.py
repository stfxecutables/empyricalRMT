import numpy as np
import os
import pytest

from pathlib import Path

from empyricalRMT.rmt.construct import generateGOEMatrix
from empyricalRMT.rmt.unfolder import Unfolder


@pytest.mark.fast
@pytest.mark.unfolder
def test_unfold_init() -> None:
    M = generateGOEMatrix(2000)
    eigs = np.linalg.eigvalsh(M)
    unfolder = Unfolder(eigs)
    assert np.alltrue(unfolder.eigs == eigs)
    assert np.alltrue(unfolder.eigenvalues == eigs)


@pytest.mark.fast
@pytest.mark.trim
def test_trim_manual() -> None:
    M = generateGOEMatrix(2000)
    eigs = np.linalg.eigvalsh(M)
    for i in range(20):
        unfolder = Unfolder(eigs)
        m, n = np.sort(np.array(np.random.uniform(0, len(eigs), 2), dtype=int))
        raw_trimmed = np.copy(eigs[m:n])
        unfolder = Unfolder(eigs)
        trimmed = unfolder.trim_manual(m, n)
        assert np.allclose(raw_trimmed, trimmed)
        assert np.allclose(raw_trimmed, unfolder.trimmed)


@pytest.mark.fast
@pytest.mark.trim
@pytest.mark.unfolder
def test_trim_reports() -> None:
    M = generateGOEMatrix(2000)
    eigs = np.linalg.eigvalsh(M)
    unfolder = Unfolder(eigs)
    unfolder.trim()
    test_dir = Path(__file__).absolute().parent
    output_plot = test_dir / "trim_summary.png"
    report, best_smoothers, best_unfoldeds, consistent = unfolder.trim_report_summary(
        show_plot=False, save_plot=output_plot
    )
    # basic sanity check
    report_best = report.filter(regex="score").abs().min().min()
    smoother_best = best_smoothers["best"].filter(regex="score").abs().values[0]
    np.testing.assert_allclose(report_best, smoother_best)

    assert output_plot.exists
    os.remove(output_plot)

    output_csv = test_dir / "trim_report.csv"
    report.to_csv(output_csv)
    assert output_csv.exists

    # output_list_csv = test_dir / "trim_report__list.csv"
    # report.to_csv(output_list_csv)
    # assert output_list_csv.exists
    # # os.remove(output_csv)

