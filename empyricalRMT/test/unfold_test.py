import numpy as np

import empyricalRMT.rmt.construct as construct
import empyricalRMT.rmt as rmt
import empyricalRMT.rmt.unfold as unfold

from empyricalRMT.rmt.observables.levelvariance import sigmaSquared_exhaustive
from empyricalRMT.rmt.observables.rigidity import spectralRigidity
from empyricalRMT.rmt.observables.spacings import computeSpacings
from empyricalRMT.rmt.unfold import UnfoldOptions, Unfolder
from empyricalRMT.utils import is_symmetric


def test_unfold_init():
    options = UnfoldOptions("poly", poly_degree=8, emd_detrend=False, method="auto")
    M = np.random.normal(2, 5, 1000000).reshape([1000, 1000])
    M = M + M.T
    eigs = np.linalg.eigvalsh(M)
    unfolder = Unfolder(eigs, options)
    assert np.alltrue(unfolder.eigs == eigs)
    assert np.alltrue(unfolder.eigenvalues == eigs)


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
