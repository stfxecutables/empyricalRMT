"""
What we are really doing here is looking at the central eigenvalues that *appear
to be GOE based on the Nearest-Neighbour Spacing Distributions, after unfolding*.
By trimming system-specific eigenvalues related to overall non-random correlations,
we can examine the part of the system that *appears to be random*, from the perspective
of one of the RMT spectral observable metrics.

If the spectral observables are correlated with interesting info, great! It doesn't
matter that theory was violated in the process. The question then comes down to an
information-theoretic question: are we sure that the spectral observable machinery of
RMT hasn't just reduced the data in a trivial way and identified trivial differences
between states / modes of interest?

That is, since empirical RMT analyses are based solely on the sorted (unfolded)
eigenvalues of the correlations of a system,
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sbn

from numpy.polynomial.polynomial import polyfit, polyval
from pathlib import Path
from pyod.models.hbos import HBOS
from PyEMD import EMD
from scipy.interpolate import UnivariateSpline as USpline
from scipy.optimize import curve_fit
from warnings import warn


from ..rmt.construct import generateGOEMatrix
from ..rmt.detrend import emd_detrend
from ..rmt.eigenvalues import getEigs, stepFunctionVectorized
from ..rmt.exponentials import gompertz
from ..rmt.observables.spacings import computeSpacings
from ..rmt.plot import setup_plotting

# from utils import eprint
from ..utils import find_first, find_last, is_symmetric, mkdirp

EXPECTED_GOE_VARIANCE = 0.286
EXPECTED_GOE_MEAN = 1.000

DEFAULT_POLY_DEGREE = 9
DEFAULT_SPLINE_SMOOTH = 1.4
DEFAULT_SPLINE_DEGREE = 3

DEFAULT_POLY_DEGREES = [3, 4, 5, 6, 7, 8, 9, 10, 11]
DEFAULT_SPLINE_SMOOTHS = np.linspace(1, 2, num=11)
DEFAULT_SPLINE_DEGREES = [3]


# Morales et al. (2011) DOI: 10.1103/PhysRevE.84.016203
def test(
    smoother="poly", degree=DEFAULT_POLY_DEGREE, percent: int = None, title="Default"
):
    sbn.set(rc={"lines.linewidth": 0.9})

    M = np.corrcoef(generateGOEMatrix(400))  # get corr matrix
    if not is_symmetric(M):
        raise ValueError("Non-symmetric matrix generated")
    eigs = getEigs(M)  # already sorted ascending

    # TODO: update this
    if method == "spline":
        unfolded = spline(eigs, param, None, percent)
    else:
        unfolded = polynomial(eigs, param, None, percent)

    spacings = computeSpacings(unfolded, sort=False)
    s_av = np.average(spacings)
    s_i = spacings - s_av

    ns = np.zeros([len(unfolded)], dtype=int)
    delta_n = np.zeros([len(unfolded)])
    for n in range(len(unfolded)):
        delta_n[n] = np.sum(s_i[0:n])
        ns[n] = n

    data = {"n": ns, "δ_n": delta_n}
    df = pd.DataFrame(data)
    sbn.lineplot(x=ns, y=delta_n, data=df)

    # Plot expected (zero) progression
    _, right = plt.xlim()
    L = np.linspace(0.001, right, 1000)

    expected = np.empty([len(L)])
    expected.fill(np.average(delta_n))
    exp_delta = plt.plot(L, expected, label="Expected δ_n (Mean) Value")
    plt.setp(exp_delta, color="#0066FF")

    trend = EMD().emd(delta_n)[-1]
    emd_res = plt.plot(ns, trend, label="Empirical Mode Dist. Residual Trend for δ_n")
    plt.setp(emd_res, color="#FD8208")

    detrended = delta_n - trend
    delta_n_max = np.max(delta_n)  # want to push higher than this
    detrended_min = np.min(detrended)
    detrend_offset = delta_n_max + abs(detrended_min)

    detrend_plot = plt.plot(ns, detrended + detrend_offset, label="Detrended δ_n")
    plt.setp(detrend_plot, color="#08FD4F")

    detrend_zero = 0 * L
    detrend_zero_plot = plt.plot(
        L, detrend_zero + detrend_offset, label="Expected Detrended Mean"
    )
    plt.setp(detrend_zero_plot, color="#0066FF")

    detrend_mean = np.empty([len(L)])
    detrend_mean.fill(np.average(detrended))
    detrend_mean_plot = plt.plot(
        L, detrend_mean + detrend_offset, label="Actual Detrended Mean"
    )
    plt.setp(detrend_mean_plot, color="#222222")

    plt.xlabel("n")
    plt.ylabel("δ_n")
    plt.legend()
    plt.title(title)
    plt.show()


# TODO: Potentially rename this to like a "Modeler", "Fitter", or "(RMT_)Smoother",
# and have it return an "Unfolded" class which will not only make typing more
# consistent by always passing around and working with Unfolded objects, but
# also allow handy utility methods like "unfolded.spacings" to be implemented
class Unfolder:
    """Base class for storing eigenvalues, trimmed eigenvalues, and
    unfolded eigenvalues"""

    def __init__(self, eigs):
        """Construct an Unfolder.

        Parameters
        ----------
        eigs: array_like
            a list, numpy array, or other iterable of the computed eigenvalues
            of some matrix
        """
        if eigs is None:
            raise ValueError("`eigs` must be an array_like.")
        try:
            length = len(eigs)
            if length < 50:
                warn(
                    "You have less than 50 eigenvalues, and the assumptions of Random "
                    "Matrix Theory are almost certainly not justified. Any results "
                    "obtained should be interpreted with caution",
                    category=UserWarning,
                )
        except TypeError:
            raise ValueError(
                "The `eigs` passed to unfolded must be an object with a defined length via `len()`."
            )

        self.__raw_eigs = np.array(eigs)
        self.__sorted_eigs = np.sort(self.__raw_eigs)
        self.__trimmed_eigs = None
        self.__trimmed_indices = (None, None)
        self.__trimmed_steps = []
        self.__unfolded = None
        return

    @property
    def eigenvalues(self) -> np.array:
        """get the original (sorted) eigenvalues as a numpy array"""
        return self.__sorted_eigs

    @property
    def eigs(self) -> np.array:
        """get the original (sorted) eigenvalues as a numpy array (alternate)"""
        return self.__sorted_eigs

    @property
    def trimmed(self) -> np.array:
        """get the trimmed, sorted eigenvalues as a numpy array"""
        return self.__trimmed_eigs

    def trim(
        self, show_plot=False, save_plot: Path = None, outlier_tol=0.1, max_trim=0.5
    ):
        """compute the optimal trim regions iteratively via histogram-based outlier detection

        Parameters
        ----------
        show_plot: Boolean
            if True, show a (blocking) plot of the iterative trims
        save_plot: Path
            if a pathlib file Path is specified, ending in .png, the outlier plot will be
            saved to Path, without blocking
        outlier_tol: float
            A float between 0 and 1. Determines the tolerance paramater for HBOD
            histogram-based outlier detection
        max_trim: float
            A float between 0 and 1 of the maximum allowable proportion of eigenvalues
            that can be trimmed.
        """
        print("Trimming to central eigenvalues.")

        eigs = self.eigs
        _, steps = self.__fit(eigs, steps_only=True)
        self.__trimmed_steps = []
        self.__trimmed_steps = self.__collect_outliers(
            eigs, steps, tolerance=outlier_tol, max_trim=max_trim
        )
        if show_plot is True or save_plot is not None:
            self.__plot_outliers(show_plot, save_plot)

    def trim_unfold_best(
        self,
        poly_degrees=DEFAULT_POLY_DEGREES,
        spline_smooths=DEFAULT_SPLINE_SMOOTHS,
        spline_degrees=DEFAULT_SPLINE_DEGREES,
    ) -> np.array:
        """Exhaustively trim and unfold for various smoothers, and select the "best" overall trim
        percent and smoother according to GOE score.

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
        """
        _, _, best_unfoldeds, _ = self.trim_report_summary(show_plot=False)
        self.__unfolded = np.array(best_unfoldeds["best"])
        return np.copy(self.__unfolded)

    def trim_manual(self, start: int, end: int) -> np.array:
        """trim sorted eigenvalues to [start:end), e.g. [eigs[start], ..., eigs[end-1]], and save
        this in the Unfolder object"""
        self.__trimmed_eigs = self.eigs[start:end]
        return np.copy(self.__trimmed_eigs)

    def trim_report_summary(
        self,
        show_plot=True,
        save_plot: Path = None,
        poly_degrees=DEFAULT_POLY_DEGREES,
        spline_smooths=DEFAULT_SPLINE_SMOOTHS,
        spline_degrees=DEFAULT_SPLINE_DEGREES,
    ) -> (pd.DataFrame, dict, pd.DataFrame, list):
        """Performs unfolding and computes GOE fit scores for various possible default
        smoothers, and returns various "best" fits.

        Parameters
        ----------
        show_plot: boolean
            if True, shows a plot of the automated outlier detection results
        save_plot: Path
            if save_plot is a pathlib file Path, save the outlier detection plot to that
            location. Should be a .png, e.g. "save_plot = Path.home() / outlier_plot.png".
        poly_degrees: List[int]
            the polynomial degrees for which to compute fits. Default [3, 4, 5, 6, 7, 8, 9, 10, 11]
        spline_smooths: List[float]
            the smoothing factors passed into scipy.interpolate.UnivariateSpline fits.
            Default np.linspace(1, 2, num=11)
        spline_degrees: List[int]
            A list of ints determining the degrees of scipy.interpolate.UnivariateSpline
            fits. Default [3]

        Returns
        -------
        report: DataFrame
            A pandas DataFrame with various summary information about the different trims
            and smoothing fits
        best_smoothers: Dict
            A dict with keys "best", "second", "third", "0", "1", "2" and the GOE fit
            scores
        best_unfoldeds: DataFrame
            a DataFrame with column names identifying the fit method, and columns
            corresponding to the unfolded eigenvalues using those methods. The first
            column has the "best" unfolded values, the second column the second best, and
            etc, up to the third best
        consistent: List
            a list of the "generally" best overall smoothers, across various possible
            trimmings. I.e. returns the smoothers with the best mean and median GOE fit
            scores across all trimmings.

        """
        if len(self.__trimmed_steps) == 0:
            raise RuntimeError(
                "Eigenvalues have not been trimmed yet. Call Unfolder.trim() "
                "before attempting to generate a trim summary."
            )
        self.__plot_outliers(show_plot, save_plot)
        report, unfolds = (
            self.trim_report()
        )  # unfolds can be used to get best unfolded eigs
        scores = report.filter(regex=".*score.*").abs()

        # get column names so we don't have to deal with terrible Pandas return types
        score_cols = np.array(scores.columns.to_list())
        # gives column names of columns with lowest scores
        best_smoother_cols = list(scores.abs().min().sort_values()[:3].to_dict().keys())
        # indices of rows with best scores
        best_smoother_rows = report[best_smoother_cols].abs().idxmin().to_list()
        # best unfolded eigenvalues
        best_unfoldeds = unfolds[
            map(lambda s: s.replace("--score", ""), best_smoother_cols)
        ]

        # construct dict with trim amounts of best overall scoring smoothers
        best_smoothers = {}
        trim_cols = ["trim_percent", "trim_low", "trim_high"]
        for i, col in enumerate(best_smoother_cols):
            min_score_i = best_smoother_rows[i]
            cols = trim_cols + [
                col.replace("score", "mean_spacing"),
                col.replace("score", "var_spacing"),
                col,
            ]
            if i == 0:
                best_smoothers["best"] = report[cols].iloc[min_score_i, :]
            elif i == 1:
                best_smoothers["second"] = report[cols].iloc[min_score_i, :]
            elif i == 2:
                best_smoothers["third"] = report[cols].iloc[min_score_i, :]
            best_smoothers[i] = report[cols].iloc[min_score_i, :]

        median_scores = np.array(scores.median())
        mean_scores = np.array(scores.mean())

        # get most consistent 3 of each
        best_median_col_idx = np.argsort(median_scores)[:3]
        best_mean_col_idx = np.argsort(mean_scores)[:3]
        top_smoothers_median = set(score_cols[best_median_col_idx])
        top_smoothers_mean = set(score_cols[best_mean_col_idx])
        consistent = top_smoothers_mean.intersection(top_smoothers_median)
        consistent = list(map(lambda s: s.replace("--score", ""), consistent))

        return report, best_smoothers, best_unfoldeds, consistent

    def trim_report(
        self,
        poly_degrees=DEFAULT_POLY_DEGREES,
        spline_smooths=DEFAULT_SPLINE_SMOOTHS,
        spline_degrees=DEFAULT_SPLINE_DEGREES,
    ) -> (pd.DataFrame, pd.DataFrame):
        """Generate a dataframe showing the unfoldings that results from different
        trim percentages, and different choices of smoothing functions.

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

        Returns
        -------
        report: a pandas DataFrame of various statistics relating to the trimmed eigenvalues,
        and metrics of how GOE the resultant trimmed central eigenvalues look

        unfolded: a dataframe of all the different unfolded eigenvalues for each fitting /
        smoothing function
        """
        trims = self.__trimmed_steps
        eigs = self.eigs

        # trim_percents = [np.round(100*(1 - len(trim["eigs"]) / len(self.eigs)), 3) for trim in trims]
        col_names_base = self.__fit_all(
            dry_run=True,
            poly_degrees=poly_degrees,
            spline_smooths=spline_smooths,
            spline_degrees=spline_degrees,
        )
        height = len(trims)
        width = (
            len(col_names_base) * 3 + 3
        )  # entry for mean, var, score, plus trim_percent, trim_low, trim_high
        arr = np.empty([height, width], dtype=np.float32)
        for i, trim in enumerate(trims):
            trimmed = np.array(trim["eigs"])
            lower_trim_length = find_first(eigs, trimmed[0])
            upper_trim_length = len(eigs) - 1 - find_last(eigs, trimmed[-1])
            all_unfolds = self.__fit_all(trimmed)  # dataframe
            trim_percent = np.round(100 * (1 - len(trimmed) / len(self.eigs)), 3)
            lower_trim_percent = 100 * lower_trim_length / len(eigs)
            upper_trim_percent = 100 * upper_trim_length / len(eigs)
            arr[i, 0] = trim_percent
            arr[i, 1] = lower_trim_percent
            arr[i, 2] = upper_trim_percent

            for j, col in enumerate(
                all_unfolds
            ):  # get summary starts for each unfolding by smoother
                unfolded = np.array(all_unfolds[col])
                mean, var, score = self._evaluate_unfolding(unfolded)
                arr[
                    i, 3 * j + 3
                ] = mean  # arr[i, 0] is trim_percent, [i,1] is trim_min, etc
                arr[i, 3 * j + 4] = var
                arr[i, 3 * j + 5] = score

        col_names_final = ["trim_percent", "trim_low", "trim_high"]
        for name in col_names_base:
            col_names_final.append(f"{name}--mean_spacing")
            col_names_final.append(f"{name}--var_spacing")
            col_names_final.append(f"{name}--score")
        return pd.DataFrame(data=arr, columns=col_names_final), all_unfolds

    def unfold(
        self,
        smoother="poly",
        degree=9,
        spline_smooth="heuristic",
        trim=True,
        return_steps=False,
    ) -> np.array:
        unfolded, steps = self.__fit(
            self.eigs, smoother=smoother, degree=degree, spline_smooth=spline_smooth
        )
        self.__unfolded = unfolded
        if return_steps:
            return unfolded, steps
        return unfolded

    def __fit(
        self, eigs, smoother="poly", degree=None, spline_smooth=None, steps_only=False
    ) -> (np.array, np.array):
        """returns (unfolded, steps), the unfolded eigenvalues and the step
        function values

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
        eigs = np.sort(np.array(eigs))
        steps = stepFunctionVectorized(eigs, eigs)
        if steps_only:
            return eigs, steps
        self.__validate_args(
            smoother=smoother, degree=degree, spline_smooth=spline_smooth
        )

        if smoother == "poly":
            if degree is None:
                degree = DEFAULT_POLY_DEGREE
            poly_coef = polyfit(eigs, steps, degree)
            unfolded = np.sort(polyval(eigs, poly_coef))
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
                spline = USpline(eigs, steps, k=k, s=spline_smooth)
            return np.sort(spline(eigs)), steps

        if smoother == "gompertz":
            # use steps[end] as guess for the asymptote, a, of gompertz curve
            [a, b, c], cov = curve_fit(gompertz, eigs, steps, p0=(steps[-1], 1, 1))
            return np.sort(gompertz(eigs, a, b, c)), steps

    def __fit_all(
        self,
        eigs=None,
        poly_degrees=DEFAULT_POLY_DEGREES,
        spline_smooths=DEFAULT_SPLINE_SMOOTHS,
        spline_degrees=DEFAULT_SPLINE_DEGREES,
        dry_run=False,
    ) -> pd.DataFrame:
        """unfold eigenvalues for all possible smoothers

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

        Returns:
        --------
        DataFrame of unfolded eigenvalues for each set of fit parameters, e.g. where each
        column contains a name indicating the fitting parameters, with the values of that
        column being the (sorted) unfolded eigenvalues
        """
        if eigs is None and (dry_run is False or dry_run is None):
            raise ValueError(
                "If not doing a dry run, you must input eigenvalues to __fit"
            )
        spline_dict = {3: "cubic", 4: "quartic", 5: "quintic"}
        spline_name = (
            lambda i: spline_dict[i] if spline_dict.get(i) is not None else f"deg{i}"
        )

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

        for d in poly_degrees:
            col_name = f"poly_{d}"
            unfolded, _ = self.__fit(eigs, smoother="poly", degree=d)
            df[col_name] = unfolded
        for s in spline_smooths:
            for d in spline_degrees:
                col_name = f"{spline_name(d)}-spline_" "{:1.1f}".format(s)
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
        spline_dict = {3: "cubic", 4: "quartic", 5: "quintic"}
        spline_name = (
            lambda i: spline_dict[i] if spline_dict.get(i) is not None else f"deg{i}"
        )

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
                    col_name = f"{spline_name(deg)}-spline_" "{:1.1f}".format(s)
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
            spline_dict = {3: "cubic", 4: "quartic", 5: "quintic"}
            spline_name = (
                lambda i: spline_dict[i]
                if spline_dict.get(i) is not None
                else f"deg{i}"
            )
            spline_smooth = float(spline_smooth)  # if can't be converted, will throw
            return f"{spline_name(spline_degree)}-spline_" "{:1.1f}".format(
                spline_smooth
            )
        if gompertz is True:
            return "gompertz"
        raise ValueError("Arguments to __column_name_from_args cannot all be None")

    def __collect_outliers(self, eigs, steps, tolerance=0.1, max_trim=0.5):
        # zeroth iteration is just the full set of values, none considered outliers
        iter_results = [
            pd.DataFrame(
                {
                    "eigs": eigs,
                    "steps": steps,
                    "unfolded": self.__fit(eigs, degree=DEFAULT_POLY_DEGREE)[0],
                    "cluster": ["inlier" for _ in eigs],
                }
            )
        ]

        while (
            len(iter_results[-1]) / len(eigs)
        ) > max_trim:  # terminate if we have trimmed max_trim
            # because eigs are sorted, HBOS will always identify outliers at one of the
            # two ends of the eigenvalues, which is what we want
            df = iter_results[-1].copy(deep=True)
            df = df[df["cluster"] == "inlier"]
            hb = HBOS(tol=tolerance)
            is_outlier = np.array(
                hb.fit(df[["eigs", "steps"]]).labels_, dtype=bool
            )  # outliers get "1"

            # check we haven't removed middle values:
            if is_outlier[0]:
                start = find_first(is_outlier, False)
                for i in range(start, len(is_outlier)):
                    is_outlier[i] = False
            if is_outlier[-1]:
                stop = find_last(is_outlier, False)
                for i in range(stop):
                    is_outlier[i] = False
            if not is_outlier[0] and not is_outlier[-1]:  # force a break later
                is_outlier = np.zeros(is_outlier.shape, dtype=bool)

            df["cluster"] = ["outlier" if label else "inlier" for label in is_outlier]
            df["unfolded"], _ = self.__fit(
                np.array(df["eigs"]), degree=DEFAULT_POLY_DEGREE
            )
            iter_results.append(df)
            if np.alltrue(~is_outlier):
                break

        return iter_results

    def __plot_outliers(self, show_plot=True, save_plot=None):
        sbn.set_style("darkgrid")
        width = 5  # 5 plots
        height = np.ceil(len(self.__trimmed_steps) / width)
        for i, df in enumerate(self.__trimmed_steps):
            df = df.rename(index=str, columns={"eigs": "λ", "steps": "N(λ)"})
            trim_percent = np.round(
                100 * (1 - len(df["cluster"] == "inlier") / len(self.eigs)), 2
            )
            plt.subplot(height, width, i + 1)
            spacings = np.sort(np.array(df["unfolded"]))
            spacings = spacings[1:] - spacings[:-1]
            sbn.scatterplot(
                data=df,
                x="λ",
                y="N(λ)",
                hue="cluster",
                style="cluster",
                style_order=["inlier", "outlier"],
                linewidth=0,
                legend="brief",
                markers=[".", "X"],
                palette=["black", "red"],
                hue_order=["inlier", "outlier"],
            )
            title = "No trim" if i == 0 else "Trim {:.2f}%".format(trim_percent)
            info = "<s> {:.4f} var(s) {:.4f}".format(
                np.mean(spacings), np.var(spacings, ddof=1)
            )
            plt.title(f"{title}\n{info}")
        plt.subplots_adjust(wspace=0.8, hspace=0.8)
        plt.suptitle("Trim fits: Goal <s> == 1, var(s) == 0.286")
        if save_plot is not None:
            path = Path(save_plot)
            mkdirp(path.parent)
            fig = plt.gcf()
            fig.set_size_inches(width * 3, height * 3)
            plt.savefig(path, dpi=100)
            print(f"Saved {path.name} to {str(path.parent.absolute())}")
        if show_plot:
            fig = plt.gcf()
            fig.set_size_inches(width * 3, height * 3)
            plt.show()

    def __validate_args(self, **kwargs):
        """throw an error if args are in any way invalid"""
        smoother = kwargs.get("smoother")
        degree = kwargs.get("degree")
        spline_smooth = kwargs.get("spline_smooth")
        emd = kwargs.get("emd_detrend")  # TODO: implement
        method = kwargs.get("method")

        if smoother == "poly":
            if degree is None:
                warn(
                    f"No degree set for polynomial unfolding. Will default to polynomial of degree {DEFAULT_POLY_DEGREE}.",
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

    def _evaluate_unfolding(self, unfolded) -> [float, float, float]:
        """Calculate a naive unfolding score via comparison to the expected mean and
        variance of the level spacings of GOE matrices. Positive scores indicate
        there is too much variability in the unfolded eigenvalue spacings, negative
        scores indicate too little. Best score is zero.
        """
        spacings = unfolded[1:] - unfolded[:-1]
        mean, var = np.mean(spacings), np.var(spacings, ddof=1)
        # variance gets weight 1, i.e. mean is 0.05 times as important
        mean_weight = 0.05
        mean_norm = (mean - EXPECTED_GOE_MEAN) / EXPECTED_GOE_MEAN
        var_norm = (var - EXPECTED_GOE_VARIANCE) / EXPECTED_GOE_VARIANCE
        score = var_norm + mean_weight * mean_norm
        return mean, var, score
