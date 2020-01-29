import pandas as pd

from numpy import ndarray
from pandas import DataFrame
from pathlib import Path

from empyricalRMT.rmt._constants import (
    EXPECTED_GOE_MEAN,
    EXPECTED_GOE_VARIANCE,
    DEFAULT_POLY_DEGREE,
    DEFAULT_SPLINE_SMOOTH,
    DEFAULT_SPLINE_DEGREE,
    DEFAULT_POLY_DEGREES,
    DEFAULT_SPLINE_SMOOTHS,
    DEFAULT_SPLINE_DEGREES,
)
from empyricalRMT.rmt._eigvals import EigVals
from empyricalRMT.rmt.observables.step import stepFunctionVectorized
from empyricalRMT.rmt.unfold import Unfolded


class TrimReport:
    def __init__(self, trim_iters: [dict]):
        self._trim_iters = trim_iters


class Trimmed(EigVals):
    def __init__(self, eigenvalues: ndarray, trimmed: ndarray):
        super().__init__(eigenvalues)
        self._trim_iters = None
        self._trim_indices = None
        self._trim_report = None
        self._vals = trimmed

    @property
    def values(self) -> ndarray:
        return self._vals

    @property
    def vals(self) -> ndarray:
        return self._vals

    @property
    def trim_indices(self) -> (int, int):
        raise NotImplementedError
        return self._trim_indices

    @property
    def trim_report(self) -> DataFrame:
        raise NotImplementedError
        return self._trim_report

    @staticmethod
    def _from_iters(self, trim_iters: [DataFrame]) -> "Trimmed":
        eigs = trim_iters[0]["eigs"]
        vals = trim_report[-1]
        trimmed = Trimmed()

    def plot_trimmed(self):
        raise NotImplementedError

    def unfold(self) -> "Unfolded":
        raise NotImplementedError
        return


def _get_trim_iters(eigs, tolerance=0.1, max_trim=0.5, max_iters=5) -> [DataFrame]:
    """Iteratively perform histogram-based outlier detection until reaching
    either max_trim or max_iters, saving outliers identified at each step.

    Paramaters
    ----------
    tolerance: float
        tolerance level for HBOS
    max_trim: float
        Value in (0,1) representing the maximum allowable proportion of eigenvalues
        trimmed.
    max_iters: int
        Maximum number of iterations (times) to perform HBOS outlier detection.

    Returns
    -------
    trim_iterations: [DataFrame]
        A list of pandas DataFrames with structure:
        ```
        {
            "eigs": np.array,
            "steps": np.array,
            "unfolded": np.array,
            "cluster": ["inlier" | "outlier"]
        }
        ```
        such that trim_iterations[0] is the original values without trimming, and
        trim_iterations[i] is a DataFrame of the eigenvalues, step function values,
        unfolded values, and inlier/outlier labels at iteration `i`.
    """
    steps = stepFunctionVectorized(eigs, eigs)
    iter_results = [  # zeroth iteration is just the full set of values, none considered outliers
        pd.DataFrame(
            {
                "eigs": eigs,
                "steps": steps,
                "unfolded": self.__fit(eigs, degree=DEFAULT_POLY_DEGREE)[0],
                "cluster": ["inlier" for _ in eigs],
            }
        )
    ]
    # terminate if we have trimmed max_trim
    iters_run = 0
    while ((len(iter_results[-1]) / len(eigs)) > max_trim) and (iters_run < max_iters):
        iters_run += 1
        # because eigs are sorted, HBOS will usually identify outliers at one of the
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
        df["unfolded"], _ = self.__fit(np.array(df["eigs"]), degree=DEFAULT_POLY_DEGREE)
        iter_results.append(df)
        if np.alltrue(~is_outlier):
            break

    return iter_results


# TODO: make work with new layout
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


# TODO: make work with new layout
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
