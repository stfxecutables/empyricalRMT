from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from warnings import warn

import numpy as np
import pandas as pd
from numpy.polynomial.polynomial import polyfit, polyval
from numpy.typing import ArrayLike
from pandas import DataFrame
from scipy.interpolate import UnivariateSpline as USpline
from scipy.optimize import curve_fit
from typing_extensions import Literal

from empyricalRMT._constants import (
    DEFAULT_POLY_DEGREE,
    DEFAULT_SPLINE_DEGREE,
    DEFAULT_SPLINE_DEGREES,
    DEFAULT_SPLINE_SMOOTH,
    DEFAULT_SPLINE_SMOOTHS,
)
from empyricalRMT._types import fArr
from empyricalRMT.detrend import emd_detrend
from empyricalRMT.exponentials import gompertz

SPLINE_DICT = {3: "cubic", 4: "quartic", 5: "quintic"}


class SmoothMethod(Enum):
    Polynomial = "poly"
    Spline = "spline"
    Exponential = "exp"
    Gompertz = "gompertz"
    GOE = "goe"
    Poisson = "poisson"

    @classmethod
    def validate(cls, s: str | SmoothMethod) -> SmoothMethod:
        try:
            if isinstance(s, str):
                return cls(s)
            return s
        except Exception as e:
            values = [e.value for e in cls]
            raise ValueError(f"SmoothMethod must be one of {values}") from e


SmoothArg = Union[List[float], Literal["heuristic"]]


def _spline_name(i: int) -> str:
    return SPLINE_DICT[i] if SPLINE_DICT.get(i) is not None else f"deg{i}"


class Smoother:
    def __init__(self, eigenvalues: ArrayLike):
        """Initialize a Smoother.

        Parameters
        ----------
        eigenvalues: ndarray
            Eigenvalues for fitting to the step function.
        """
        try:
            eigs = np.array(eigenvalues).astype(np.float64)
        except BaseException as e:
            raise ValueError("Could not convert eigenvalues into numpy array.") from e
        if len(eigs) != len(eigs.reshape((-1,))):
            raise ValueError("Input array must be one-dimensional.")
        eigs = eigs.flatten()
        self._eigs = np.sort(eigs)

    def fit(
        self,
        smoother: SmoothMethod = SmoothMethod.Polynomial,
        degree: int = DEFAULT_POLY_DEGREE,
        spline_smooth: float = DEFAULT_SPLINE_SMOOTH,
        detrend: bool = False,
        return_callable: bool = False,
    ) -> Tuple[fArr, fArr, Optional[Callable[[fArr], fArr]]]:
        """Computer the specified smoothing function values for a set of eigenvalues.

        Parameters
        ----------
        eigs: ndarray
            The sorted eigenvalues

        smoother: "poly" | "spline" | "gompertz" | lambda
            The type of smoothing function used to fit the step function

        degree: int
            The degree of the polynomial or spline

        spline_smooth: float
            The smoothing factors passed into scipy.interpolate.UnivariateSpline

        detrend: bool
            Whether or not to perform EMD detrending before returning the
            unfolded eigenvalues.

        return_callable: bool
            If true, return a function that closes over the fit parameters so
            that, e.g., additional values can be fit later.


        Returns
        -------
        unfolded: ndarray
            the unfolded eigenvalues

        steps: ndarray
            the step-function values
        """
        eigs = self._eigs
        # steps = _step_function_fast(eigs, eigs)
        steps = np.arange(0, len(eigs)) + 1
        self.__validate_args(smoother=smoother, degree=degree, spline_smooth=spline_smooth)

        if smoother is SmoothMethod.Polynomial:
            poly_coef = polyfit(eigs, steps, degree)
            unfolded = polyval(eigs, poly_coef)
            func = lambda x: polyval(x, poly_coef) if return_callable else None
            if detrend:
                unfolded = emd_detrend(unfolded)
            return unfolded, steps, func  # type: ignore

        if smoother is SmoothMethod.Spline:
            k = DEFAULT_SPLINE_DEGREE
            try:
                k = int(degree)
            except BaseException as e:
                print(ValueError("Cannot convert spline degree to int."))
                raise e
            if spline_smooth == "heuristic":
                s = len(eigs) * np.var(eigs, ddof=1)
                spline = USpline(eigs, steps, k=k, s=s)
            elif spline_smooth is not None:
                if not isinstance(spline_smooth, float):
                    raise ValueError("Spline smoothing factor must be a float")
                spline = USpline(eigs, steps, k=k, s=spline_smooth)
            else:
                raise ValueError(
                    "Unreachable: All possible spline_smooth arguments should have been handled."
                )
                spline = USpline(eigs, steps, k=k, s=spline_smooth)
            func = spline if return_callable else None
            unfolded = np.array(spline(eigs))
            if detrend:
                unfolded = emd_detrend(unfolded)
            return unfolded, steps, func  # type: ignore

        if smoother is SmoothMethod.Exponential:
            func = lambda t, a, b: a * np.exp(-b * t)  # type: ignore
            [a, b], cov = curve_fit(
                func,
                eigs,
                steps,
                p0=(len(eigs) / 2, 0.5),
            )
            unfolded = func(eigs, a, b)  # type: ignore
            if detrend:
                unfolded = emd_detrend(unfolded)
            return unfolded, steps, func  # type: ignore

        if smoother is SmoothMethod.Gompertz:
            # use steps[end] as guess for the asymptote, a, of gompertz curve
            [a, b, c], cov = curve_fit(gompertz, eigs, steps, p0=(steps[-1], 1, 1))
            func = lambda x: gompertz(x, a, b, c) if return_callable else None
            unfolded = gompertz(eigs, a, b, c)
            if detrend:
                unfolded = emd_detrend(unfolded)
            return unfolded, steps, func  # type: ignore
        raise RuntimeError("Unreachable!")

    def fit_all(
        self,
        poly_degrees: List[int] = [],
        spline_smooths: SmoothArg = [],
        spline_degrees: List[int] = DEFAULT_SPLINE_DEGREES,
        gompertz: bool = False,
        detrend: bool = False,
    ) -> Tuple[DataFrame, DataFrame, DataFrame, Dict[str, Callable]]:
        """unfold eigenvalues for all specified smoothers

        Parameters
        ----------
        poly_degrees: List[int]
            the polynomial degrees for which to compute fits.
            Default [3, 4, 5, 6, 7, 8, 9, 10, 11]

        spline_smooths: List[float] | "heuristic"
            If a list of floats, the smoothing factors, s, passed into
            scipy.interpolate.UnivariateSpline.
            If "heuristic", choose a set of smoothing factors scaled to the length of the
            eigenvalues, that, on GOE eigenvalues, tend to result in a range of fits
            varying from highly flexible (nearly interpolated) to about the flexibility of
            a cubic or quartic. As the number of eigenvalues starts to go below about 300,
            an increasing number of practically-identical, redundant splines will be fit
            with this option, and manual inspection or non-heuristic specification of
            spline smoothing factors is strongly recommended.

        spline_degrees: List[int]
            A list of ints determining the degrees of scipy.interpolate.UnivariateSpline
            fits. Default [3]


        Returns
        -------
        unfoldeds: DataFrame
            DataFrame of unfolded eigenvalues for each set of fit parameters, e.g. where
            each column contains a name indicating the fitting parameters, with the values
            of that column being the (sorted) unfolded eigenvalues.

        sqes: DataFrame
            DataFrame of mean-squared error of fits, where each column contains a name
            indicating the fitting parameters and smoother, with the values of
            the column being the mean of the squared residuals of the fit

        smoother_map: dict
            A dict of {col_name: closure} for accessing the fitted smoothers later.
        """
        # construct dataframes to hold all info
        col_names, unfoldeds, spacings, sqes = [], [], [], []
        smoother_map = {}
        for d in poly_degrees:
            col_name = f"poly_{d}"
            unfolded, steps, closure = self.fit(
                smoother=SmoothMethod.Polynomial, degree=d, return_callable=True, detrend=detrend
            )
            col_names.append(col_name)
            sqes.append(np.mean((unfolded - steps) ** 2))
            unfolded = np.sort(unfolded)  # Important!
            unfoldeds.append(unfolded)
            spacings.append(np.diff(unfolded))
            smoother_map[col_name] = closure

        if spline_smooths == "heuristic":
            for s in DEFAULT_SPLINE_SMOOTHS:
                for d in spline_degrees:
                    col_name = f"{_spline_name(d)}-spline_" "{:1.2f}_heuristic".format(s)
                    unfolded, steps, closure = self.fit(
                        smoother=SmoothMethod.Spline,
                        spline_smooth=len(self._eigs) ** s,
                        degree=d,
                        return_callable=True,
                        detrend=detrend,
                    )
                    col_names.append(col_name)
                    sqes.append(np.mean((unfolded - steps) ** 2))
                    unfolded = np.sort(unfolded)
                    unfoldeds.append(unfolded)
                    spacings.append(np.diff(unfolded))
                    smoother_map[col_name] = closure
        else:
            for s in spline_smooths:  # type: ignore
                for d in spline_degrees:
                    col_name = f"{_spline_name(d)}-spline_" "{:1.3f}".format(s)
                    unfolded, steps, closure = self.fit(
                        smoother=SmoothMethod.Spline,
                        spline_smooth=s,
                        degree=d,
                        return_callable=True,
                        detrend=detrend,
                    )
                    col_names.append(col_name)
                    sqes.append(np.mean((unfolded - steps) ** 2))
                    unfolded = np.sort(unfolded)
                    unfoldeds.append(unfolded)
                    spacings.append(np.diff(unfolded))
                    smoother_map[col_name] = closure
        if gompertz:
            unfolded, steps, closure = self.fit(
                smoother=SmoothMethod.Gompertz, return_callable=True, detrend=detrend
            )
            col_names.append("gompertz")
            sqes.append(np.mean((unfolded - steps) ** 2))
            unfolded = np.sort(unfolded)
            unfoldeds.append(unfolded)
            spacings.append(np.diff(unfolded))
            smoother_map["gompertz"] = closure
        unfoldeds = pd.DataFrame(data=unfoldeds, index=col_names).T
        spacings = pd.DataFrame(data=spacings, index=col_names).T
        sqes = pd.DataFrame(data=sqes, index=col_names).T
        return unfoldeds, spacings, sqes, smoother_map  # type: ignore

    @staticmethod
    def _get_smoother_names(
        poly_degrees: List[int],
        spline_smooths: SmoothArg,
        spline_degrees: List[int] = [3],
        gompertz: bool = True,
    ) -> List[str]:
        """If arguments are arrays, generate names (unique identifiers) for each smoother
        + smoother parameters. Otherwise, just return the name for indexing into the report.
        """

        col_names = []
        if isinstance(poly_degrees, list):
            for d in poly_degrees:
                col_names.append(f"poly_{d}")
        else:
            raise ValueError("poly_degrees must be a list of int values")

        if spline_smooths == "heuristic":
            for s in DEFAULT_SPLINE_SMOOTHS:
                if not isinstance(spline_degrees, list):
                    raise ValueError("spline_degrees must be a list of integer values")
                for deg in spline_degrees:
                    col_name = f"{_spline_name(deg)}-spline_" "{:1.3f}_heuristic".format(s)
                    col_names.append(col_name)
        else:
            try:
                spline_smooths = list(spline_smooths)  # type: ignore
            except Exception as e:
                raise ValueError(f"Error converting `spline_smooths` to list: {e}")
            if isinstance(spline_smooths, list):
                for s in spline_smooths:
                    if not isinstance(spline_degrees, list):
                        raise ValueError("spline_degrees must be a list of integer values")
                    for deg in spline_degrees:
                        col_name = f"{_spline_name(deg)}-spline_" "{:1.3f}".format(s)
                        col_names.append(col_name)
            else:
                raise ValueError("spline_smooths must be a list of float values")

        if gompertz is True:
            col_names.append("gompertz")
        return col_names

    def __validate_args(self, **kwargs: Any) -> None:
        """throw an error if smoother args are in any way invalid"""
        smoother = SmoothMethod.validate(kwargs.get("smoother"))  # type: ignore
        degree = kwargs.get("degree")
        spline_smooth = kwargs.get("spline_smooth")
        emd = kwargs.get("detrend")  # TODO: implement
        method = kwargs.get("method")

        if smoother is SmoothMethod.Polynomial:
            if degree is None:
                warn(
                    "No degree set for polynomial unfolding."
                    f"Will default to polynomial of degree {DEFAULT_POLY_DEGREE}.",
                    category=UserWarning,
                )
            if not isinstance(degree, int):
                raise ValueError("Polynomial degree must be of type `int`")
            if degree < 3:
                raise ValueError("Unfolding polynomial must have minimum degree 3.")
        elif smoother is SmoothMethod.Spline:
            spline_degree = degree
            if degree is None:
                warn(
                    "No degree set for spline unfolding. Will default to "
                    f"spline of degree {DEFAULT_SPLINE_DEGREE}.",
                    category=UserWarning,
                )
            if not isinstance(spline_degree, int) or spline_degree > 5:
                raise ValueError("Degree of spline must be an int <= 5")
            if spline_smooth is not None and spline_smooth != "heuristic":
                spline_smooth = float(spline_smooth)
        elif smoother in [
            SmoothMethod.Gompertz,
            SmoothMethod.Exponential,
            SmoothMethod.GOE,
            SmoothMethod.Poisson,
        ]:
            pass  # just allow this for now
        elif callable(smoother):
            # NOTE: above is not a great check, but probably good enough for our purposes
            # https://stackoverflow.com/questions/624926/how-do-i-detect-whether-a-python-variable-is-a-function#comment437753_624939
            raise NotImplementedError("Custom fit functions not currently implemented.")
        else:
            raise ValueError("Unrecognized smoother argument.")

        if emd is not None and not isinstance(emd, bool):
            raise ValueError("`detrend` can be only a boolean or undefined (None).")

        if method is None or method == "auto" or method == "manual":
            pass
        else:
            raise ValueError("`method` must be one of 'auto', 'manual', or 'None'")
