import numpy as np
import os

from pathlib import Path

import empyricalRMT.rmt.construct as construct
import empyricalRMT.rmt as rmt
import empyricalRMT.rmt.unfold as unfold

from empyricalRMT.rmt.construct import generateGOEMatrix
from empyricalRMT.rmt.observables.levelvariance import sigmaSquared_exhaustive
from empyricalRMT.rmt.observables.rigidity import spectralRigidity
from empyricalRMT.rmt.observables.spacings import computeSpacings
from empyricalRMT.rmt.unfold import Unfolder
from empyricalRMT.utils import is_symmetric


def test_unfold_init():
    M = generateGOEMatrix(2000)
    eigs = np.linalg.eigvalsh(M)
    unfolder = Unfolder(eigs)
    assert np.alltrue(unfolder.eigs == eigs)
    assert np.alltrue(unfolder.eigenvalues == eigs)


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


def test_spline_unfold(
    M=construct.generateGOEMatrix(1000),
    matsize=1000,
    knots=5,
    percent=98,
    detrend=True,
    raws=False,
):
    if M is None:
        M = construct.generateGOEMatrix(matsize)

    eigs = construct.getEigs(M)  # already sorted ascending
    if not is_symmetric(M):
        raise ValueError("Non-symmetric matrix generated")

    unfolded = unfold.spline(eigs, knots, detrend=detrend, percent=percent)
    spacings = computeSpacings(unfolded)  # noqa: F841
    print("Average spacing: ", np.average(spacings))

    if raws is True:
        rmt.plot.rawEigDist(eigs)
        rmt.plot.unfoldedFit(unfolded, f"{knots}-knots Spline Fit, Middle {percent}%")

    rmt.plot.spacings(spacings, title=f"{knots}-knots Spline Fit, Middle {percent}%")
    spec_data = spectralRigidity(eigs, unfolded, L_grid_size=100)
    rmt.plot.spectralRigidity(
        unfolded=unfolded,
        data=spec_data,
        title=f"{knots}-knots Spline Unfolding, Middle {percent}%",
    )

    level_var_spline = sigmaSquared_exhaustive(unfolded, c_step=0.1, L_grid_size=100)
    rmt.plot.levelNumberVariance(
        level_var_spline, f"{knots}-knots Spline Unfolding, Middle {percent}%"
    )


def test_poly_unfold(matsize=1000, degree=5, percent=98, detrend=True, raws=False):
    M = construct.generateGOEMatrix(matsize)
    eigs = construct.getEigs(M)  # already sorted ascending
    if not is_symmetric(M):
        raise ValueError("Non-symmetric matrix generated")

    unfolded = unfold.polynomial(eigs, degree, detrend=detrend, percent=percent)
    spacings = computeSpacings(unfolded)  # noqa: F841
    print("Average spacing: ", np.average(spacings))

    if raws is True:
        rmt.plot.rawEigDist(eigs)
        rmt.plot.unfoldedFit(
            unfolded, f"Degree-{degree} Polynomial Fit, Middle {percent}%"
        )

    rmt.plot.spacings(spacings, f"Degree-{degree} Polynomial Fit, Middle {percent}%")

    spec_data = spectralRigidity(unfolded, L_iters=10, L_grid_size=100)
    rmt.plot.spectralRigidity(
        spec_data, f"Degree-{degree} Polynomial Unfolding, Middle {percent}%"
    )

    level_var_spline = sigmaSquared_exhaustive(unfolded, c_step=0.1, L_grid_size=100)
    rmt.plot.levelNumberVariance(
        level_var_spline, f"Degree-{degree} Polynomial Unfolding, Middle {percent}%"
    )


test_trim()
