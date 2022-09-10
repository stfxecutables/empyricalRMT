from __future__ import annotations

"""
Motivation for this module

Trimming is ultimately inextricably entangled with the smoothing process. We
trim because extreme eigenvalues act as leverage points for most fitting
procedures, but the extent to which a trimming is "bad" can only be determined
by evaluating whether a particular fit is still being unduly influenced by some
of the trimmed values. That is, trimming is *truly* actually a *supervised*
procedure.

Nevertheless, unsupervised outlier detection methods allow us to identify
contiguous sections of eigenvalues that are *very likely* to result in poorly
conditioned fits. So it makes sense to identify a set of likely trims first,
and then check the fits from there.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy import floating, ndarray
from pandas import DataFrame
from pyod.models.hbos import HBOS
from scipy.stats import trim_mean

# from empyricalRMT._eigvals import EigVals
import empyricalRMT._eigvals as _eigvals
from empyricalRMT._constants import (
    DEFAULT_POLY_DEGREE,
    DEFAULT_POLY_DEGREES,
    DEFAULT_SPLINE_DEGREES,
    DEFAULT_SPLINE_SMOOTH,
    DEFAULT_SPLINE_SMOOTHS,
    EXPECTED_GOE_MEAN,
    EXPECTED_GOE_VARIANCE,
)
from empyricalRMT._types import fArr
from empyricalRMT.plot import PlotMode, PlotResult, _plot_trim_iters
from empyricalRMT.smoother import Smoother, SmoothMethod
from empyricalRMT.unfold import Unfolded
from empyricalRMT.utils import find_first, find_last


class Trimmed(_eigvals.EigVals):
    """Class for holding already trimmed eigenvalues and giving access to convenience methods
    on those values."""

    def __init__(self, trimmed: ndarray):
        """Construct a Trimmed object.

        Parameters
        ----------
        trimmed: ndarray
            An ndarray which has had undesirable eigenvalues (small, large, or
            both) trimmed away.
        """

        super().__init__(trimmed)

    @property
    def values(self) -> ndarray:
        return self._vals

    @property
    def vals(self) -> ndarray:
        return self._vals

    def unfold(
        self,
        smoother: SmoothMethod = SmoothMethod.Polynomial,
        degree: int = DEFAULT_POLY_DEGREE,
        spline_smooth: float = DEFAULT_SPLINE_SMOOTH,
        detrend: bool = False,
    ) -> Unfolded:
        """
        Parameters
        ----------
        smoother: "poly" | "spline" | "gompertz" | lambda
            the type of smoothing function used to fit the step function

        degree: int
            the degree of the polynomial or spline

        spline_smooth: float
            the smoothing factors passed into scipy.interpolate.UnivariateSpline

        detrend: bool
            Whether or not to perform EMD detrending before returning the
            unfolded eigenvalues.


        Returns
        -------
        unfolded: Unfolded
            An Unfolded object containing the unfolded eigenvalues.
        """
        eigs = self.values
        unfolded, _, closure = Smoother(eigs).fit(
            smoother=smoother,
            degree=degree,
            spline_smooth=spline_smooth,
            detrend=detrend,
        )
        return Unfolded(originals=eigs, unfolded=unfolded, smoother=closure)

    def unfold_auto(
        self,
        poly_degrees: List[int] = DEFAULT_POLY_DEGREES,
        spline_smooths: List[float] = [],
        spline_degrees: List[int] = [],
        gompertz: bool = True,
        outlier_tol: float = 0.1,
    ) -> Unfolded:
        """Exhaustively compare mutliple trim regions and smoothers based on their "GOE score"
        and unfold the eigenvalues, using the trim region and smoothing parameters
        determined to be "most GOE" based on the exhaustive process.

        Exhaustively trim and unfold for various smoothers, and select the "best" overall trim
        percent and smoother according to GOE score.

        Parameters
        ----------
        poly_degrees: List[int]
            the polynomial degrees for which to compute fits. Default [3, 4, 5,
            6, 7, 8, 9, 10, 11]

        spline_smooths: List[float]
            the smoothing factors passed into scipy.interpolate.UnivariateSpline fits.
            Default np.linspace(1, 2, num=11)

        spline_degrees: List[int]
            A list of ints determining the degrees of scipy.interpolate.UnivariateSpline
            fits. Default [3]

        gompertz: bool
            Whether or not to use a gompertz curve as one of the smoothers.

        prioritize_smoother: bool
            Whether or not to select the optimal smoother before selecting the optimal
            trim region. See notes. Default: True.

        outlier_tol: float
            A float between 0 and 1, and which is passed as the tolerance parameter for
            [HBOS](https://pyod.readthedocs.io/en/latest/pyod.models.html#module-pyod.models.hbos)
            histogram-based outlier detection

        Returns
        -------
        unfolded: Unfolded
            An Unfolded object containing the unfolded eigenvalues.
        """

        trimmed = TrimReport(
            eigenvalues=self.values,
            max_trim=0.0,
            max_iters=0,
            poly_degrees=poly_degrees,
            spline_smooths=spline_smooths,
            spline_degrees=spline_degrees,
            gompertz=gompertz,
            outlier_tol=outlier_tol,
        )
        orig_trimmed, unfolded = trimmed._get_autounfold_vals()
        return Unfolded(orig_trimmed, unfolded)


class TrimReport:
    """A class for storing summaries of the various trim iterations."""

    def __init__(
        self,
        eigenvalues: ndarray,
        max_trim: float = 0.5,
        max_iters: int = 7,
        poly_degrees: List[int] = DEFAULT_POLY_DEGREES,
        spline_smooths: List[float] = [],
        spline_degrees: List[int] = [],
        gompertz: bool = True,
        detrend: bool = False,
        outlier_tol: float = 0.1,
        show_progress: bool = False,
    ):
        """Construct a TrimReport.

        Parameters
        ----------
        eigenvalues: ndarray
            The eigenvalues to trim and unfold.

        max_trim: float
            Float in (0, 1). The maximum allowable portion of eigenvalues to be trimmed.
            E.g. `max_trim=0.8` means to allow up to 80% of the original eigenvalues to
            be trimmed away.

        max_iters: int
            The maximum allowable number of iterations of outlier detection to run.
            Setting `max_iters=0` will not allow any trimming / outlier detection, and so
            will simply evaluate unfolding for different smoothers on the original raw
            eigenvalues. Typically, you would want this to be >= 4, to allow for trimming
            both some of the most extreme positive and negative eigenvalues.

        poly_degrees: List[int]
            the polynomial degrees for which to compute fits. Default [3, 4, 5,
            6, 7, 8, 9, 10, 11]

        spline_smooths: List[float]
            the smoothing factors passed into scipy.interpolate.UnivariateSpline fits.
            Default np.linspace(1, 2, num=11)

        spline_degrees: List[int]
            A list of ints determining the degrees of scipy.interpolate.UnivariateSpline
            fits. Default [3]

        gompertz: bool
            Whether or not to use a gompertz curve as one of the smoothers.

        detrend: bool
            Whether or not to perform EMD detrending before returning the
            unfolded eigenvalues.

        outlier_tol: float
            A float between 0 and 1, and which is passed as the tolerance parameter for
            [HBOS](https://pyod.readthedocs.io/en/latest/pyod.models.html#module-pyod.models.hbos)
            histogram-based outlier detection
        """
        eigenvalues = np.sort(eigenvalues)
        self._untrimmed: ndarray = eigenvalues
        self._all_unfolds: Optional[List[DataFrame]] = None

        self._trim_iters: List[TrimIter] = self.__get_trim_iters(
            tolerance=outlier_tol,
            max_trim=max_trim,
            max_iters=max_iters,
            poly_degrees=poly_degrees,
            spline_smooths=spline_smooths,
            spline_degrees=spline_degrees,
            gompertz=gompertz,
            detrend=detrend,
            show_progress=show_progress,
        )
        self._all_unfolds = list(map(lambda trim: trim.unfolds, self._trim_iters))  # type: ignore
        # set self._unfold_info, self._all_unfolds
        self._summary = self.__iters_to_dataframe(
            poly_degrees, spline_smooths, spline_degrees, gompertz
        )

    @property
    def trim_indices(self) -> List[Tuple[int, int]]:
        """Return a list of the various trim indices found by the iterative
        outlier detection procedure.

        Returns
        -------
        trim_idx: List[Tuple[int, int]]
            A list of indices (idx_min, idx_max), such that if the original
            eigenvalues are `eigs`, then the trims are `eigs[idx_min:idx_max]`
        """
        return list(map(lambda trim: trim.trim_indices, self._trim_iters))

    @property
    def untrimmed(self) -> ndarray:
        """Returns the class's copy of the original eigenvalues."""
        return self._untrimmed

    @property
    def summary(self) -> DataFrame:
        """Returns the summary DataFrame.

        Returns
        -------
        df: DataFrame
            The summary DataFrame. See Description below.

        Description
        -----------
        The summary DataFrame summarizes the results of the various unfoldings
        across different trims. Each row represents a particular trimming, with
        the first row containing all the original eigenvalues, and with each
        subsequent row containing a smaller range of eigenvalues than the
        previous.

        The first three columns ["trim_percent", "trim_low", "trim_high"]
        summarize the trim percents. E.g. a "trim_low" of 4.5 means that
        4.5% of the total eigenvalues were trimmed from the bottom (most
        negative) eigenvalues.

        The remaining columns contain a short identifier for the smoother (e.g.
        "poly_5" for a polynomial of degree 5) followed by a separator ('--'),
        and then a string indicating the values contained in the columns. Those
        strings are:

        - "mean_spacing": the mean spacing of the unfolded values
        - "var_spacing": the variance of the unfolded values
        - "msqe": the mean squared error of the fit to the step function
        - "score": the 'goe score' combining the above two values
        """

        return self._summary

    @property
    def unfoldings(self) -> List[DataFrame]:
        """Get the actual unfolded values for all trim iterations.

        Returns
        -------
        unfolded: List[DataFrame]
            A list of pandas DataFrame objects. Column names use the same
            identifier strings as used in the Unfolded.summary object.
        """
        if self._all_unfolds is None:
            raise RuntimeError("TrimReport inrrectly initialized.")
        return self._all_unfolds

    def use_trim_iteration(self, iteration: int) -> Trimmed:
        """Get a specific set of trimmed eigenvalues.

        Parameters
        ----------
        iteration: int
            The index of the desired iteration. Will coincide with the iteration
            listed in plot generated by `TrimReport.plot_trim_steps()`.

        Returns
        -------
        trimmed: Trimmed
            A new Trimmed object containing the trimmed (but *not* unfolded)
            eigenvalues.
        """
        from empyricalRMT.eigenvalues import Trimmed

        if iteration > (len(self._trim_iters) - 1):
            raise ValueError(f"Largest trim iteration is {len(self._trim_iters) - 1}")
        return Trimmed(self._trim_iters[iteration].eigs)

    def evaluate(self, criterion: Any, minimize_trim: bool = True) -> Any:
        """TODO: Implement various sensible criteria here.

        E.g:

        # Criteria

        - 'fit'
            - basically, choose the most flexible smoother and most agressive
              trim
        - 'goe'
            - choose the trim and unfolding that result in spacing mean and
              variance closest to that expected for GOE matrices
        - ''
        """
        raise NotImplementedError("Work in progress!")

    def best_overall(
        self,
    ) -> tuple[Dict[Union[str, int], str], list[DataFrame], list[Tuple[int, int]], list[str]]:
        """Computes GOE fit scores for the unfoldings performed, and returns various
        "best" fits.

        Returns
        -------
        best_smoothers: Dict
            A dict with keys "best", "second", "third", (or equivalently "0", "1", "2",
            respectively) and the GOE fit scores

        best_unfoldeds: List[DataFrame]
            A list of DataFrame elements with column names identifying the fit method, and
            columns corresponding to the unfolded eigenvalues using those methods. The
            first column of `best_unfoldeds[i]` has the "best" unfolded values, the second
            column the second best, and etc, up to the third best.

        best_trim_indices: List[Tuple[int, int]]
            Compares the GOE scores of each of the possible trim regions across all
            smoothers, and finds the three best trim regions with the best (lowest)
            score, averaging across smoothers. So e.g. if you fit polynomials of
            degree 4, 5, 6, and splines with smoothing parameters 1.0, 1.2, 1.4, then
            `best_trim_indices[0]` will contain values `(start, end)`, such that the
            original eigenvalues trimmed to [start, end) have the lowest average score
            (compared to other possible trim regions identified by the histogram-based
            outlier detection method) across those polynomial and spline fits. Likewise,
            `best_trim_indices[1]` contains the second best overall trim indices across
            smoothers, and `best_trim_indices[2]` contains the third best.

        consistent_smoothers: List[str]
            a list of names of the "generally" best overall smoothers, across various
            possible trimmings. I.e. returns the smoothers with the best mean and median
            GOE fit scores across all trimmings. Useful for deciding on a single smoothing
            method to use across a dataset. Consistent smoothers are *not* ordered.
        """
        report, all_unfolds = self.summary, self._all_unfolds
        if report is None or all_unfolds is None:
            raise RuntimeError("Eigenvalues have not yet been unfolded. This should be impossible.")
        scores = report.filter(regex=".*score.*").abs()

        # get column names so we don't have to deal with terrible Pandas return types
        score_cols = np.array(scores.columns.to_list())
        # gives column names of columns with lowest scores
        best_smoother_cols = list(scores.abs().min().sort_values()[:3].to_dict().keys())
        # indices of rows with best scores
        best_smoother_rows = report[best_smoother_cols].abs().idxmin().to_list()
        # best unfolded eigenvalues
        best_smoother_names = [s.replace("--score", "") for s in best_smoother_cols]
        best_unfoldeds = [unfold[best_smoother_names] for unfold in all_unfolds]

        # take the mean GOE score across smoothers for each trimming, find the row
        # with the lowest mean score, and call this the "best overall" trim
        sorted_row_mean_scores = report.filter(regex="score").abs().mean(axis=1).sort_values()
        best_three = list(sorted_row_mean_scores[:3].index)  # get indices of best three rows
        best_trim_indices = []
        for i, row_id in enumerate(best_three):
            best_trim_eigs = np.array(self._trim_iters[row_id].eigs)
            best_start, best_end = best_trim_eigs[0], best_trim_eigs[-1]
            best_indices = (
                list(self._untrimmed).index(best_start),
                list(self._untrimmed).index(best_end) + 1,
            )
            best_trim_indices.append(best_indices)

        # construct dict with trim amounts of best overall scoring smoothers
        best_smoothers: Dict[Union[str, int], str] = {}
        trim_cols = ["trim_percent", "trim_low", "trim_high"]
        for i, col in enumerate(best_smoother_cols):
            min_score_i = best_smoother_rows[i]
            cols = trim_cols + [
                col.replace("score", "mean_spacing"),
                col.replace("score", "var_spacing"),
                col,
            ]
            if i == 0:
                best_smoothers["best"] = report[cols].iloc[min_score_i, :].item()
            elif i == 1:
                best_smoothers["second"] = report[cols].iloc[min_score_i, :].item()
            elif i == 2:
                best_smoothers["third"] = report[cols].iloc[min_score_i, :].item()
            best_smoothers[i] = report[cols].iloc[min_score_i, :].item()

        # TODO: implement "best" trim

        median_scores = np.array(scores.median())
        mean_scores = np.array(scores.mean())

        # get most consistent 3 of each
        best_median_col_idx = np.argsort(median_scores)[:3]
        best_mean_col_idx = np.argsort(mean_scores)[:3]
        top_smoothers_median = set(score_cols[best_median_col_idx])
        top_smoothers_mean = set(score_cols[best_mean_col_idx])
        consistent = list(top_smoothers_mean.intersection(top_smoothers_median))
        consistent_smoothers = list(map(lambda s: str(s.replace("--score", "")), consistent))

        return best_smoothers, best_unfoldeds, best_trim_indices, consistent_smoothers

    def unfold_trimmed(self) -> Unfolded:
        raise NotImplementedError()

    def plot_trim_steps(
        self,
        title: str = "Trim fits",
        mode: PlotMode = PlotMode.Return,
        outfile: Optional[Path] = None,
        width: int = 4,
        log_info: bool = True,
    ) -> PlotResult:
        """Show which eigenvalues are trimmed at each iteration.

        Parameters
        ----------
        title: string
            The plot title string

        mode: "block" (default) | "noblock" | "save" | "return"
            If "block", call plot.plot() and display plot in a blocking fashion.
            If "noblock", attempt to generate plot in nonblocking fashion.
            If "save", save plot to pathlib Path specified in `outfile` argument
            If "return", return (fig, axes), the matplotlib figure and axes
            object for modification.

        outfile: Path
            If mode="save", save generated plot to Path specified in `outfile` argument.
            Intermediate directories will be created if needed.

        width: int
            Number of plots to show per row of figure.

        log_info: boolean
            If True, print additional information about each trimming to stdout.


        Returns
        -------
        (fig, axes): (Figure, Axes)
            The handles to the matplotlib objects, only if `mode` is "return".
        """
        if log_info:
            for trim in self._trim_iters:
                print(trim.summary()[0])
            print(self._trim_iters[0].summary()[1])  # print legend

        return _plot_trim_iters(
            self._trim_iters, width=width, title=title, mode=mode, outfile=outfile
        )

    def to_csv(self, *args: Any, **kwargs: Any) -> None:
        """A wrapper around pandas DataFrame.to_csv()"""
        self.summary.to_csv(*args, **kwargs)

    def _get_autounfold_vals(self) -> Tuple[ndarray, ndarray]:
        scores = self.summary.filter(regex="score").abs()
        mean_best = scores[scores < scores.quantile(0.9)].mean().sort_values()[:5].index.to_list()
        best_trim_scores = self.summary.filter(regex=mean_best[0])  # e.g. regex="poly_3--score"
        best_trim_id = int(best_trim_scores.abs().idxmin().item())
        best_unfolded = self.unfoldings[best_trim_id][mean_best[0].replace("--score", "")]
        orig_trimmed = self._trim_iters[best_trim_id].eigs
        return np.array(orig_trimmed), np.array(best_unfolded)

    def __get_trim_iters(
        self,
        tolerance: float = 0.1,
        max_trim: float = 0.5,
        max_iters: int = 7,
        show_progress: bool = False,
        **smoother_kwargs: Any,
    ) -> List[TrimIter]:
        """Helper function to iteratively perform histogram-based outlier detection
        until reaching either max_trim or max_iters, saving outliers identified at
        each step.

        Parameters
        ----------
        tolerance: float
            tolerance level for HBOS

        max_trim: float
            Value in (0,1) representing the maximum allowable proportion of eigenvalues
            that can be trimmed *away*. E.g. setting `max_trim` == 0.2 means we
            are allowed to trim *up to* 20% of the eigenvalues away, resulting
            in a set of trimmed eigenvalues with length 80% of the original eigenvalues.

        max_iters: int
            Maximum number of iterations (times) to perform HBOS outlier detection.


        Returns
        -------
        trim_iters: [DataFrame]
            A list of pandas DataFrames with structure:
            ```
            {
                "eigs": ndarray,
                "steps": ndarray,
                "unfolded": ndarray,
                "sqe": ndarray,
                "cluster": ["inlier" | "outlier"]
            }
            ```
            where:
                * `eigs` are the remaining eigs at this indexed trimming,
                * `steps` are the step-function values for `eigs`,
                * `unfolded` are the unfolded eigenvalues corresponding to the
                  smoother specified in the arguments,
                * `sqe` are the squared residuals of the unfolding smoother fit,
                * `cluster` indicates whether HBOS identifies a value as outlier

            and such that trim_iters[0] is the original values without trimming, and
            trim_iters[i] is a DataFrame of the eigenvalues, step function values,
            unfolded values, and inlier/outlier labels at iteration `i`.
        """
        eigs = np.copy(self._untrimmed)
        trim_iters = [TrimIter(eigs, eigs, tolerance, **smoother_kwargs)]
        for i in range(max_iters):
            if show_progress:
                print(f"Completed trim-unfold iteration: {i}.")
            trim = trim_iters[-1].next_iter()
            if trim.proportion_removed > max_trim:
                break
            trim_iters.append(trim)
            if trim.is_all_inliers():
                break
        return trim_iters

    def __iters_to_dataframe(
        self,
        poly_degrees: List[int] = DEFAULT_POLY_DEGREES,
        spline_smooths: List[float] = DEFAULT_SPLINE_SMOOTHS,
        spline_degrees: List[int] = DEFAULT_SPLINE_DEGREES,
        gompertz: bool = True,
    ) -> DataFrame:
        """Generate a dataframe showing the unfoldings that results from different
        trim percentages, and different choices of smoothing functions.

        Parameters
        ----------
        poly_degrees: List[int]
            the polynomial degrees for which to compute fits. Default [3, 4, 5,
            6, 7, 8, 9, 10, 11]

        spline_smooths: List[float]
            the smoothing factors passed into scipy.interpolate.UnivariateSpline fits.
            Default np.linspace(1, 2, num=11)

        spline_degrees: List[int]
            A list of ints determining the degrees of
            scipy.interpolate.UnivariateSpline fits. Default [3]

        Returns
        -------
        trim_report: DataFrame
            A pandas DataFrame with a row for each iteration of trimming, and
            columns with various summary statistics (mean spacing, variance of
            the spacings, msqe of the smoother fit, and GOE score) for each
            unfolding for that trim.

        """
        # save args for later
        self.__poly_degrees = poly_degrees
        self.__spline_smooths = spline_smooths
        self.__spline_degrees = spline_degrees
        trim_iters = self._trim_iters

        colnames = Smoother._get_smoother_names(
            poly_degrees=poly_degrees,
            spline_smooths=spline_smooths,
            spline_degrees=spline_degrees,
            gompertz=gompertz,
        )
        height = len(trim_iters)
        # entry for [mean, var, msqe, score] + [trim_percent, trim_low, trim_high]
        width = len(colnames) * 4 + 3

        # arr will be converted into the final DataFrame
        arr = np.zeros([height, width], dtype=np.float32)
        index = []
        for i, trim in enumerate(trim_iters):
            index.append(trim.id)
            # 3 columns of values per trim
            arr[i, 0] = trim.percent_removed
            arr[i, 1] = trim.lower_percent_removed
            arr[i, 2] = trim.upper_percent_removed

            # get summary stats for each unfolding by smoother
            for j, col in enumerate(trim.unfolds):
                unfolded = np.array(trim.unfolds[col])
                mean, var, score = self.__evaluate_unfolding(unfolded)
                # arr[i, 0] is trim_percent, [i,1] is lower_trim_percent, etc, up to
                # arr[i, 2], which has the upper_trim_percent. Thus ultimately 4
                # additional columns of values per smoother:
                arr[i, 4 * j + 3] = mean
                arr[i, 4 * j + 4] = var
                arr[i, 4 * j + 5] = trim.msqes[col]
                arr[i, 4 * j + 6] = score

        col_names_final = ["trim_percent", "trim_low", "trim_high"]
        # much match order added above
        for name in colnames:
            col_names_final.append(f"{name}--mean_spacing")
            col_names_final.append(f"{name}--var_spacing")
            col_names_final.append(f"{name}--msqe")
            col_names_final.append(f"{name}--score")
        trim_report = pd.DataFrame(data=arr, columns=col_names_final, index=index)
        return trim_report

    @staticmethod
    def __evaluate_unfolding(unfolded: fArr) -> Tuple[floating, floating, floating]:
        """Calculate a naive unfolding score via comparison to the expected mean and
        variance of the level spacings of GOE matrices. Positive scores indicate
        there is too much variability in the unfolded eigenvalue spacings, negative
        scores indicate too little. Best score is zero.
        """
        spacings = unfolded[1:] - unfolded[:-1]
        mean, var = np.mean(spacings), np.var(spacings, ddof=1)
        # variance gets weight 1, i.e. mean is 0.05 times as important
        mean_weight = 0.5
        mean_norm = (mean - EXPECTED_GOE_MEAN) / EXPECTED_GOE_MEAN
        var_norm = (var - EXPECTED_GOE_VARIANCE) / EXPECTED_GOE_VARIANCE
        score = var_norm + mean_weight * mean_norm
        return mean, var, score


class TrimIter:
    """Helper class for storing data and improving code readability of trimming
    process"""

    def __init__(self, origs: ndarray, eigs: ndarray, outlier_tol: float, **smoother_kwargs: Any):
        self.origs = origs
        self.eigs = eigs
        self.id = 0
        self.tol = outlier_tol
        self.kwargs = smoother_kwargs
        self.steps = np.arange(1, len(eigs) + 1)
        self.clusters = np.array(_get_outlier_labels(eigs, tol=outlier_tol))
        unfolds, spacings, msqes, smoothers = Smoother(eigs).fit_all(**smoother_kwargs)
        self.unfolds: DataFrame = unfolds
        self.spacings: DataFrame = spacings
        self.msqes: DataFrame = msqes
        self.smoothers: Dict[str, Callable] = smoothers

    @property
    def inlier_length(self) -> int:
        return int(np.count_nonzero(self.clusters == "inlier"))

    @property
    def outlier_length(self) -> int:
        return int(np.count_nonzero(self.clusters == "outlier"))

    @property
    def proportion_kept(self) -> float:
        return self.inlier_length / len(self.origs)

    @property
    def proportion_removed(self) -> float:
        return 1 - len(self.eigs) / len(self.origs)

    @property
    def percent_removed(self) -> float:
        return float(np.round(100.0 - 100.0 * len(self.eigs) / len(self.origs), 1))

    @property
    def lower_percent_removed(self) -> float:
        lower_trim_length = find_first(self.origs, self.eigs[0])
        return float(np.round(100 * lower_trim_length / len(self.origs), 1))

    @property
    def upper_percent_removed(self) -> float:
        upper_trim_length = len(self.origs) - 1 - find_last(self.origs, self.eigs[-1])
        return float(np.round(100 * upper_trim_length / len(self.origs), 1))

    @property
    def trim_indices(self) -> Tuple[int, int]:
        start = find_first(self.origs, self.eigs[0])
        end = find_last(self.origs, self.eigs[-1])
        return (start, end)

    @property
    def inliers(self) -> ndarray:
        return np.copy(self.eigs[self.clusters == "inlier"])  # type: ignore

    def is_all_inliers(self) -> bool:
        return bool(np.alltrue(self.clusters == "inlier"))  # type: ignore

    def next_iter(self) -> "TrimIter":
        trim = TrimIter(self.origs, self.inliers, self.tol, **self.kwargs)
        trim.id = self.id + 1
        return trim

    def summary(self) -> Tuple[str, str]:
        percent = self.percent_removed
        start, end = self.trim_indices
        mean = float(trim_mean(self.spacings.mean(), 0.2))
        var = float(trim_mean(self.spacings.var(ddof=1), 0.2))
        mmsqe = float(trim_mean(self.msqes, 0.2, axis=1))
        iter_info = "Iteration {:d}:".format(self.id)
        trim_info = "{:4.1f}% trimmed - Trim indices: ({:d},{:d})".format(percent, start, end)
        fit_info = "<s> = {:1.6f}, var(s) = {:04.5f}, <MSQE>: {:5.5f}.".format(mean, var, mmsqe)
        legend = (
            "\n<MSQE>: 20% trimmed mean MSQE across unfoldings.\n"
            "<s>: 20% trimmed mean of mean spacings across unfoldings.\n"
            "var(s): 20% trimmed mean of spacings variance across unfoldings.\n"
        )
        return f"{iter_info} {trim_info} - {fit_info}", legend


def _get_outlier_labels(eigs: ndarray, tol: float) -> List[str]:
    """Identify the outliers of eigs with HBOS."""
    hb = HBOS(tol=tol)
    steps = np.arange(0, len(eigs))
    X = np.vstack([eigs, steps]).T  # data array
    is_outlier = np.array(hb.fit(X).labels_, dtype=bool)  # outliers get "1"

    # because eigs are sorted, HBOS will *usually* identify outliers at one of
    # the two ends of the eigenvalues, which is what we want. But this is not
    # always the case, so we need to de-identify those values as outliers.
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

    return ["outlier" if label else "inlier" for label in is_outlier]
