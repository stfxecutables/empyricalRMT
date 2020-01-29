import numpy as np
import pandas as pd

from numpy import ndarray
from numpy.polynomial.polynomial import polyfit, polyval
from pandas import DataFrame
from scipy.interpolate import UnivariateSpline as USpline
from scipy.optimize import curve_fit
from warnings import warn

from empyricalRMT.rmt.exponentials import gompertz
from empyricalRMT.rmt.observables.step import stepFunctionVectorized

DEFAULT_POLY_DEGREE = 9
DEFAULT_SPLINE_SMOOTH = 1.4
DEFAULT_SPLINE_DEGREE = 3

DEFAULT_POLY_DEGREES = [3, 4, 5, 6, 7, 8, 9, 10, 11]
DEFAULT_SPLINE_SMOOTHS = np.linspace(1, 2, num=11)
DEFAULT_SPLINE_DEGREES = [3]

SPLINE_DICT = {3: "cubic", 4: "quartic", 5: "quintic"}


def _spline_name(i: int) -> str:
    return SPLINE_DICT[i] if SPLINE_DICT.get(i) is not None else f"deg{i}"


class Smoother:
    def __init__(self, eigenvalues: ndarray):
        """Initialize a Smoother.

        Parameters
        ----------
        eigenvalues: np.array
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
        smoother="poly",
        degree=None,
        spline_smooth=DEFAULT_SPLINE_SMOOTH,
        emd_detrend=False,
    ) -> ndarray:
        """Computer the specified smoothing function values for a set of eigenvalues.

        Parameters
        ----------
        eigs: np.array
            sorted eigenvalues
        smoother: "poly" | "spline" | "gompertz" | lambda
            the type of smoothing function used to fit the step function
        degree: int
            the degree of the polynomial or spline
        spline_smooth: float
            the smoothing factors passed into scipy.interpolate.UnivariateSpline

        Returns
        -------
        unfolded: np.array
            the unfolded eigenvalues

        steps: np.array
            the step-function values
        """
        eigs = self._eigs
        steps = stepFunctionVectorized(eigs, eigs)
        __validate_args(smoother=smoother, degree=degree, spline_smooth=spline_smooth)

        if smoother == "poly":
            if degree is None:
                degree = DEFAULT_POLY_DEGREE
            poly_coef = polyfit(eigs, steps, degree)
            unfolded = polyval(eigs, poly_coef)
            return unfolded, steps

        if smoother == "spline":
            if degree is None:
                degree = DEFAULT_SPLINE_DEGREE
            else:
                try:
                    k = int(degree)
                except BaseException as e:
                    print(ValueError("Cannot convert spline degree to int."))
                    raise e
            if spline_smooth == "heuristic":
                spline = USpline(eigs, steps, k=k, s=len(eigs) ** 1.4)
            elif spline_smooth is not None:
                if not isinstance(spline_smooth, float):
                    raise ValueError("Spline smoothing factor must be a float")
                spline = USpline(eigs, steps, k=k, s=len(eigs) ** spline_smooth)
            else:
                raise ValueError(
                    "Unreachable: All possible spline_smooth arguments should have been handled."
                )
                spline = USpline(eigs, steps, k=k, s=spline_smooth)
            return spline(eigs), steps

        if smoother == "gompertz":
            # use steps[end] as guess for the asymptote, a, of gompertz curve
            [a, b, c], cov = curve_fit(gompertz, eigs, steps, p0=(steps[-1], 1, 1))
            return gompertz(eigs, a, b, c), steps
        raise NotImplementedError

    def fit_all(
        self,
        poly_degrees=DEFAULT_POLY_DEGREES,
        spline_smooths=DEFAULT_SPLINE_SMOOTHS,
        spline_degrees=DEFAULT_SPLINE_DEGREES,
        dry_run=False,
    ) -> DataFrame:
        """unfold eigenvalues for all specified smoothers

        Parameters
        ----------
        poly_degrees: List[int]
            the polynomial degrees for which to compute fits. Default [3, 4, 5, 6, 7, 8, 9, 10, 11]
        spline_smooths: List[float]
            the smoothing factors passed into scipy.interpolate.UnivariateSpline fits.
            Default np.linspace(1, 2, num=11)
        spline_degrees: List[int]
            A list of ints determining the degrees of scipy.interpolate.UnivariateSpline
            fits. Default [3]
        dry_run: boolean
            If True, only return a list of column names needed for the current arguments.

        Returns:
        --------
        df: DataFrame
            DataFrame of unfolded eigenvalues for each set of fit parameters, e.g. where
            each column contains a name indicating the fitting parameters, with the values
            of that column being the (sorted) unfolded eigenvalues.
        col_names: list
            If `dry_run` is True, return only the column names for the given arguments.
        """
        # construct a dataframe to hold all info
        df = pd.DataFrame()
        col_names = self.__get_column_names(
            poly_degrees=poly_degrees,
            spline_smooths=spline_smooths,
            gompertz=True,
            spline_degrees=spline_degrees,
        )
        if dry_run:  # early return strings of colums names
            return col_names

        eigs = self._eigs
        for d in poly_degrees:
            col_name = f"poly_{d}"
            unfolded, _ = self.__fit(eigs, smoother="poly", degree=d)
            df[col_name] = unfolded
        for s in spline_smooths:
            for d in spline_degrees:
                col_name = f"{_spline_name(d)}-spline_" "{:1.1f}".format(s)
                unfolded, _ = self.__fit(
                    eigs, smoother="spline", spline_smooth=s, degree=d
                )
                df[col_name] = unfolded
        df["gompertz"], _ = self.__fit(eigs, smoother="gompertz")
        return df

    def __get_column_names(
        self, poly_degrees, spline_smooths, gompertz=True, spline_degrees=[3]
    ) -> str:
        """If arguments are arrays, generate names for all columns of report. Otherwise,
        just return the name for indexing into the report.
        """

        col_names = []
        if isinstance(poly_degrees, list):
            for d in poly_degrees:
                col_names.append(f"poly_{d}")
        else:
            raise ValueError("poly_degrees must be a list of int values")

        try:
            spline_smooths = list(spline_smooths)
        except Exception as e:
            raise ValueError(f"Error converting `spline_smooths` to list: {e}")
        if isinstance(spline_smooths, list):
            for s in spline_smooths:
                if not isinstance(spline_degrees, list):
                    raise ValueError("spline_degrees must be a list of integer values")
                for deg in spline_degrees:
                    col_name = f"{_spline_name(deg)}-spline_" "{:1.1f}".format(s)
                    col_names.append(col_name)
        else:
            raise ValueError("spline_smooths must be a list of float values")
        if gompertz is True:
            col_names.append("gompertz")
        return col_names

    def __column_name_from_args(
        self, poly_degree=None, spline_smooth=None, gompertz=None, spline_degree=3
    ) -> str:
        if isinstance(poly_degree, int):
            return f"poly_{poly_degree}"
        if spline_smooth is not None:
            spline_smooth = float(spline_smooth)  # if can't be converted, will throw
            return f"{_spline_name(spline_degree)}-spline_" "{:1.1f}".format(
                spline_smooth
            )
        if gompertz is True:
            return "gompertz"
        raise ValueError("Arguments to __column_name_from_args cannot all be None")


def __validate_args(**kwargs):
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
