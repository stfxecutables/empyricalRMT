import numpy as np
import os
import pytest

from pathlib import Path

from empyricalRMT.rmt.construct import generateGOEMatrix
from empyricalRMT.rmt.unfold import Unfolder

@pytest.mark.unfolder
def test_unfold_init():
    M = generateGOEMatrix(2000)
    eigs = np.linalg.eigvalsh(M)
    unfolder = Unfolder(eigs)
    assert np.alltrue(unfolder.eigs == eigs)
    assert np.alltrue(unfolder.eigenvalues == eigs)


@pytest.mark.unfolder
def test_trim():
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

