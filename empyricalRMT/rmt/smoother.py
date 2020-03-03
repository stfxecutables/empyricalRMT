import numpy as np
import pandas as pd

from numpy import ndarray
from numpy.polynomial.polynomial import polyfit, polyval
from pandas import DataFrame
from scipy.interpolate import UnivariateSpline as USpline
from scipy.optimize import curve_fit
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from typing_extensions import Literal
from warnings import warn

from empyricalRMT.rmt._constants import (
    DEFAULT_POLY_DEGREE,
    DEFAULT_SPLINE_DEGREE,
    DEFAULT_SPLINE_DEGREES,
    DEFAULT_SPLINE_SMOOTH,
    DEFAULT_SPLINE_SMOOTHS,
)
from empyricalRMT.rmt.exponentials import gompertz
from empyricalRMT.rmt.observables.step import _step_function_fast


SPLINE_DICT = {3: "cubic", 4: "quartic", 5: "quintic"}

SmoothMethod = Union[Literal["poly"], Literal["spline"], Literal["gompertz"]]
SmoothArg = Union[List[float], Literal["heuristic"]]


def _spline_name(i: int) -> str:
    return SPLINE_DICT[i] if SPLINE_DICT.get(i) is not None else f"deg{i}"


class Smoother:
    def __init__(self, eigenvalues: ndarray):
        """Initialize a Smoother.

        Parameters
        ----------
        eigenvalues: ndarray
            Eigenvalues for fitting the step function.
        """
        try:
            eigs = np.array(eigenvalues).ravel()
        except BaseException as e:
            raise ValueError("Could not convert eigenvalues into numpy array.") from e
        if len(eigs) != len(eigenvalues):
            raise ValueError("Input array must be one-dimensional.")
        self._eigs = np.sort(eigs)

    def fit(
        self,
        smoother: SmoothMethod = "poly",
        degree: int = DEFAULT_POLY_DEGREE,
        spline_smooth: float = DEFAULT_SPLINE_SMOOTH,
        emd_detrend: bool = False,
        return_callable: bool = False,
    ) -> Tuple[ndarray, ndarray, Optional[Callable[[ndarray], ndarray]]]:
        """Computer the specified smoothing function values for a set of eigenvalues.

        Parameters
        ----------
        eigs: ndarray
            sorted eigenvalues

        smoother: "poly" | "spline" | "gompertz" | lambda
            the type of smoothing function used to fit the step function

        degree: int
            the degree of the polynomial or spline

        spline_smooth: float
            the smoothing factors passed into scipy.interpolate.UnivariateSpline

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
        steps = _step_function_fast(eigs, eigs)
        self.__validate_args(
            smoother=smoother, degree=degree, spline_smooth=spline_smooth
        )

        if smoother == "poly":
            poly_coef = polyfit(eigs, steps, degree)
            unfolded = polyval(eigs, poly_coef)
            if return_callable:
                return unfolded, steps, lambda x: polyval(x, poly_coef)
            return unfolded, steps, None

        if smoother == "spline":
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
            if return_callable:
                return spline(eigs), steps, lambda x: spline(x)
            return spline(eigs), steps, None

        if smoother == "gompertz":
            # use steps[end] as guess for the asymptote, a, of gompertz curve
            [a, b, c], cov = curve_fit(gompertz, eigs, steps, p0=(steps[-1], 1, 1))
            if return_callable:
                return gompertz(eigs, a, b, c), steps, lambda x: gompertz(x, a, b, c)
            return gompertz(eigs, a, b, c), steps, None
        raise RuntimeError("Unreachable!")

    def fit_all(
        self,
        poly_degrees: List[int] = [],
        spline_smooths: SmoothArg = [],
        spline_degrees: List[int] = DEFAULT_SPLINE_DEGREES,
        gompertz: bool = False,
        dry_run: bool = False,
    ) -> Union[List[str], Tuple[DataFrame, DataFrame, Dict[str, Callable]]]:
        """unfold eigenvalues for all specified smoothers

        Parameters
        ----------
        poly_degrees: List[int]
            the polynomial degrees for which to compute fits. Default [3, 4, 5, 6, 7, 8, 9, 10, 11]
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
        dry_run: boolean
            If True, only return a list of column names needed for the current arguments.


        Returns
        -------
        unfoldeds: DataFrame
            DataFrame of unfolded eigenvalues for each set of fit parameters, e.g. where
            each column contains a name indicating the fitting parameters, with the values
            of that column being the (sorted) unfolded eigenvalues.

        sqes: DataFrame
            DataFrame of mean-squared error of fits, where each column contains a name
            indicating the fitting parameters and smoother, with the values of
            the column being the squared residuals of the fit
        col_names: list
            If `dry_run` is True, return only the column names for the given
            arguments.

        smoother_map: dict
            A dict of {col_name: closure} for accessing the fitted smoothers later.
        """
        col_names = self.__get_smoother_names(
            poly_degrees=poly_degrees,
            spline_smooths=spline_smooths,
            spline_degrees=spline_degrees,
            gompertz=gompertz,
        )
        if dry_run:  # early return strings of colums names
            return col_names

        # construct dataframes to hold all info
        unfoldeds, sqes = pd.DataFrame(), pd.DataFrame()
        smoother_map = {}
        for d in poly_degrees:
            col_name = f"poly_{d}"
            unfolded, steps, closure = self.fit(
                smoother="poly", degree=d, return_callable=True
            )
            unfoldeds[col_name] = unfolded
            sqes[col_name] = (unfolded - steps) ** 2
            smoother_map[col_name] = closure

        if spline_smooths == "heuristic":
            for s in DEFAULT_SPLINE_SMOOTHS:
                for d in spline_degrees:
                    col_name = f"{_spline_name(d)}-spline_" "{:1.2f}_heuristic".format(
                        s
                    )
                    unfolded, steps, closure = self.fit(
                        smoother="spline",
                        spline_smooth=len(self._eigs) ** s,
                        degree=d,
                        return_callable=True,
                    )
                    unfoldeds[col_name] = unfolded
                    sqes[col_name] = (unfolded - steps) ** 2
                    smoother_map[col_name] = closure
        else:
            for s in spline_smooths:  # type: ignore
                for d in spline_degrees:
                    col_name = f"{_spline_name(d)}-spline_" "{:1.3f}".format(s)
                    unfolded, steps, closure = self.fit(
                        smoother="spline",
                        spline_smooth=s,
                        degree=d,
                        return_callable=True,
                    )
                    unfoldeds[col_name] = unfolded
                    sqes[col_name] = (unfolded - steps) ** 2
                    smoother_map[col_name] = closure

        if gompertz:
            unfolded, steps, closure = self.fit(
                smoother="gompertz", return_callable=True
            )
            unfoldeds["gompertz"] = unfolded
            sqes["gompertz"] = (unfolded - steps) ** 2
            smoother_map[col_name] = closure
        return unfoldeds, sqes, smoother_map  # type: ignore

    def __get_smoother_names(
        self,
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
                    col_name = (
                        f"{_spline_name(deg)}-spline_" "{:1.3f}_heuristic".format(s)
                    )
                    col_names.append(col_name)
        else:
            try:
                spline_smooths = list(spline_smooths)  # type: ignore
            except Exception as e:
                raise ValueError(f"Error converting `spline_smooths` to list: {e}")
            if isinstance(spline_smooths, list):
                for s in spline_smooths:
                    if not isinstance(spline_degrees, list):
                        raise ValueError(
                            "spline_degrees must be a list of integer values"
                        )
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
        smoother = kwargs.get("smoother")
        degree = kwargs.get("degree")
        spline_smooth = kwargs.get("spline_smooth")
        emd = kwargs.get("emd_detrend")  # TODO: implement
        method = kwargs.get("method")

        if smoother == "poly":
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
        elif smoother == "spline":
            spline_degree = degree
            if degree is None:
                warn(
                    f"No degree set for spline unfolding. Will default to spline of degree {DEFAULT_SPLINE_DEGREE}.",
                    category=UserWarning,
                )
            if not isinstance(spline_degree, int) or spline_degree > 5:
                raise ValueError("Degree of spline must be an int <= 5")
            if spline_smooth is not None and spline_smooth != "heuristic":
                spline_smooth = float(spline_smooth)
        elif smoother == "gompertz":
            pass  # just allow this for now
        elif callable(smoother):
            # NOTE: above is not a great check, but probably good enough for our purposes
            # https://stackoverflow.com/questions/624926/how-do-i-detect-whether-a-python-variable-is-a-function#comment437753_624939
            raise NotImplementedError("Custom fit functions not currently implemented.")
        else:
            raise ValueError("Unrecognized smoother argument.")

        if emd is not None and not isinstance(emd, bool):
            raise ValueError("`emd_detrend` can be only a boolean or undefined (None).")

        if method is None or method == "auto" or method == "manual":
            pass
        else:
            raise ValueError("`method` must be one of 'auto', 'manual', or 'None'")
