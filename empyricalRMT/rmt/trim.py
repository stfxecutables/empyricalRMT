import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sbn

from numpy import ndarray
from pandas import DataFrame
from pathlib import Path
from pyod.models.hbos import HBOS
from typing import Dict, List, Optional, Tuple, Union

from empyricalRMT.rmt._constants import (
    EXPECTED_GOE_MEAN,
    EXPECTED_GOE_VARIANCE,
    DEFAULT_POLY_DEGREE,
    DEFAULT_POLY_DEGREES,
    DEFAULT_SPLINE_SMOOTH,
    DEFAULT_SPLINE_SMOOTHS,
    DEFAULT_SPLINE_DEGREES,
)
from empyricalRMT.rmt._eigvals import EigVals
from empyricalRMT.rmt.observables.step import _step_function_fast
from empyricalRMT.rmt.plot import _setup_plotting, PlotMode, PlotResult
from empyricalRMT.rmt.smoother import Smoother, SmoothMethod
from empyricalRMT.rmt.unfold import Unfolded
from empyricalRMT.utils import find_first, find_last, mkdirp


class TrimReport:
    def __init__(
        self,
        eigenvalues: ndarray,
        max_trim: float = 0.5,
        max_iters: int = 7,
        poly_degrees: List[int] = DEFAULT_POLY_DEGREES,
        spline_smooths: List[float] = [],
        spline_degrees: List[int] = [],
        gompertz: bool = True,
        outlier_tol: float = 0.1,
    ):
        eigenvalues = np.sort(eigenvalues)
        self._untrimmed: ndarray = eigenvalues
        self._unfold_info: Optional[DataFrame] = None
        self._all_unfolds: Optional[List[DataFrame]] = None

        self._trim_steps = self.__get_trim_iters(
            tolerance=outlier_tol, max_trim=max_trim, max_iters=max_iters
        )
        # set self._unfold_info, self._all_unfolds
        self._unfold_info, self._all_unfolds = self.__unfold_across_trims(
            poly_degrees, spline_smooths, spline_degrees, gompertz
        )

    @property
    def trim_indices(self) -> List[Tuple[int, int]]:
        trim_steps = self._trim_steps
        untrimmed = self._untrimmed

        indices = []
        for i, df in enumerate(trim_steps):
            if i == 0:
                indices.append((0, len(untrimmed)))
                continue
            eigs_list = list(untrimmed)
            start = eigs_list.index(list(df["eigs"])[0])
            end = eigs_list.index(list(df["eigs"])[-1])
            indices.append((start, end))
        return indices

    @property
    def untrimmed(self) -> ndarray:
        return self._untrimmed

    @property
    def unfold_info(self) -> DataFrame:
        return self._unfold_info

    @property
    def unfoldings(self) -> List[DataFrame]:
        if self._all_unfolds is None:
            raise RuntimeError("TrimReport inrrectly initialized.")
        return self._all_unfolds

    def compare_trim_unfolds(
        self,
        poly_degrees: List[int] = DEFAULT_POLY_DEGREES,
        spline_smooths: List[float] = DEFAULT_SPLINE_SMOOTHS,
        spline_degrees: List[int] = DEFAULT_SPLINE_DEGREES,
        gompertz: bool = True,
    ) -> DataFrame:
        """Computes unfoldings for the smoothing parameters specified in the
        arguments, across the multiple trim regions.

        Returns
        -------
        report: DataFrame
            A pandas DataFrame with various summary information about the different trims
            and smoothing fits
        """
        self._unfold_info, self._all_unfolds = self.__unfold_across_trims(
            poly_degrees, spline_smooths, spline_degrees, gompertz
        )

        return self._unfold_info

    def summarize_trim_unfoldings(
        self
    ) -> Tuple[Dict[Union[str, int], str], DataFrame, List[Tuple[int, int]], List[str]]:
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
        report, all_unfolds = self._unfold_info, self._all_unfolds
        if report is None or all_unfolds is None:
            raise RuntimeError(
                "Eigenvalues have not yet been unfolded. This should be impossible."
            )
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
        sorted_row_mean_scores = (
            report.filter(regex="score").abs().mean(axis=1).sort_values()
        )
        best_three = list(
            sorted_row_mean_scores[:3].index
        )  # get indices of best three rows
        best_trim_indices = []
        for i, row_id in enumerate(best_three):
            best_trim_eigs = np.array(self._trim_steps[row_id]["eigs"])
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
                best_smoothers["best"] = report[cols].iloc[min_score_i, :]
            elif i == 1:
                best_smoothers["second"] = report[cols].iloc[min_score_i, :]
            elif i == 2:
                best_smoothers["third"] = report[cols].iloc[min_score_i, :]
            best_smoothers[i] = report[cols].iloc[min_score_i, :]

        # TODO: implement "best" trim

        median_scores = np.array(scores.median())
        mean_scores = np.array(scores.mean())

        # get most consistent 3 of each
        best_median_col_idx = np.argsort(median_scores)[:3]
        best_mean_col_idx = np.argsort(mean_scores)[:3]
        top_smoothers_median = set(score_cols[best_median_col_idx])
        top_smoothers_mean = set(score_cols[best_mean_col_idx])
        consistent = list(top_smoothers_mean.intersection(top_smoothers_median))
        consistent_smoothers = list(
            map(lambda s: str(s.replace("--score", "")), consistent)
        )

        return best_smoothers, best_unfoldeds, best_trim_indices, consistent_smoothers

    def unfold_trimmed(self) -> Unfolded:
        raise NotImplementedError

    def plot_trim_steps(
        self,
        title: str = "Trim fits",
        mode: PlotMode = "block",
        outfile: Path = None,
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

        log_info: boolean
            If True, print additional information about each trimming to stdout.


        Returns
        -------
        (fig, axes): (Figure, Axes)
            The handles to the matplotlib objects, only if `mode` is "return".
        """
        trim_steps = self._trim_steps
        untrimmed = self._untrimmed

        if log_info:
            log = []
            for i, df in enumerate(trim_steps):
                if i == 0:
                    continue
                trim_percent = np.round(
                    100 * (1 - len(df["cluster"] == "inlier") / len(untrimmed)), 2
                )
                eigs_list = list(untrimmed)
                unfolded = df["unfolded"].to_numpy()
                spacings = unfolded[1:] - unfolded[:-1]
                info = "Iteration {:d}: {:4.1f}% trimmed. <s> = {:6.6f}, var(s) = {:04.5f} MSQE: {:5.5f}. Trim indices: ({:d},{:d})".format(  # noqa E501
                    i,
                    trim_percent,
                    np.mean(spacings),
                    np.var(spacings, ddof=1),
                    np.mean(df["sqe"]),
                    eigs_list.index(list(df["eigs"])[0]),
                    eigs_list.index(list(df["eigs"])[-1]),
                )
                log.append(info)
            print("\n".join(log))
            print(
                "MSQE, average spacing <s>, and spacings variance var(s)"
                f"calculated for polynomial degree {DEFAULT_POLY_DEGREE} unfolding."
            )

        _setup_plotting()

        width = 5  # 5 plots
        height = np.ceil(len(trim_steps) / width)
        # fig, axs = plt.subplots(height, width)
        for i, df in enumerate(trim_steps):
            df = df.rename(index=str, columns={"eigs": "λ", "steps": "N(λ)"})
            trim_percent = np.round(
                100 * (1 - len(df["cluster"] == "inlier") / len(untrimmed)), 2
            )
            plt.subplot(height, width, i + 1, label=f"plot_{i}")
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
                label=f"plot_{i}",
            )
            subtitle = "No trim" if i == 0 else "Trim {:.2f}%".format(trim_percent)
            info = "<s> {:.4f} var(s) {:.4f}".format(
                np.mean(spacings), np.var(spacings, ddof=1)
            )
            plt.title(f"{subtitle}\n{info}")
        plt.subplots_adjust(wspace=0.8, hspace=0.8)
        plt.suptitle(title)

        if mode == "save":
            if outfile is None:
                raise ValueError("Path not specified for `outfile`.")
            try:
                outfile = Path(outfile)
            except BaseException as e:
                raise ValueError("Cannot interpret outfile path.") from e
            mkdirp(outfile.parent)
            fig = plt.gcf()
            fig.set_size_inches(width * 3, height * 3)
            plt.savefig(outfile, dpi=100)
            print(f"Saved {outfile.name} to {str(outfile.parent.absolute())}")
        elif mode == "block" or mode == "noblock":
            fig = plt.gcf()
            fig.set_size_inches(width * 3, height * 3)
            plt.show(block=mode == "block")
        elif mode == "test":
            plt.show(block=False)
            plt.close()
        elif mode == "return":
            return plt.gca(), plt.gcf()
        else:
            raise ValueError("Invalid plotting mode.")
        return None

    def _get_autounfold_vals(self) -> Tuple[ndarray, ndarray]:
        scores = self.unfold_info.filter(regex="score").abs()
        mean_best = (
            scores[scores < scores.quantile(0.9)]
            .mean()
            .sort_values()[:5]
            .index.to_list()
        )
        best_trim_scores = self.unfold_info.filter(
            regex=mean_best[0]  # e.g. regex="poly_3--score"
        )
        best_trim_id = int(best_trim_scores.abs().idxmin())
        best_unfolded = self.unfoldings[best_trim_id][
            mean_best[0].replace("--score", "")
        ]
        orig_trimmed = self._trim_steps[best_trim_id]["eigs"]
        return np.array(orig_trimmed), np.array(best_unfolded)

    def __get_trim_iters(
        self, tolerance: float = 0.1, max_trim: float = 0.5, max_iters: int = 7
    ) -> List[DataFrame]:
        """Helper function to iteratively perform histogram-based outlier detection
        until reaching either max_trim or max_iters, saving outliers identified at
        each step.

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
        eigs = self._untrimmed
        steps = _step_function_fast(eigs, eigs)
        unfolded = Smoother(eigs).fit()[0]
        iter_results = [  # zeroth iteration is just the full set of values, none considered outliers
            pd.DataFrame(
                {
                    "eigs": eigs,
                    "steps": steps,
                    "unfolded": unfolded,
                    "sqe": (unfolded - steps) ** 2,
                    "cluster": ["inlier" for _ in eigs],
                }
            )
        ]
        # terminate if we have trimmed max_trim
        iters_run = 1
        while ((len(iter_results[-1]) / len(eigs)) > max_trim) and (
            iters_run < max_iters
        ):
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
            unfolded, steps, closure = Smoother(df["eigs"]).fit()
            df["unfolded"] = unfolded
            df["sqe"] = (unfolded - steps) ** 2

            iter_results.append(df)
            if np.alltrue(~is_outlier):
                break

        return iter_results

    def __unfold_across_trims(
        self,
        poly_degrees: List[int] = DEFAULT_POLY_DEGREES,
        spline_smooths: List[float] = DEFAULT_SPLINE_SMOOTHS,
        spline_degrees: List[int] = DEFAULT_SPLINE_DEGREES,
        gompertz: bool = True,
    ) -> Tuple[DataFrame, DataFrame]:
        """Generate a dataframe showing the unfoldings that results from different
        trim percentages, and different choices of smoothing functions. This should be run
        in the constructor.

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
        """
        # save args for later
        self.__poly_degrees = poly_degrees
        self.__spline_smooths = spline_smooths
        self.__spline_degrees = spline_degrees

        trims = self._trim_steps
        eigs = self._untrimmed

        # trim_percents = [np.round(100*(1 - len(trim["eigs"]) / len(self.eigs)), 3) for trim in trims]
        col_names_base = Smoother(eigs).fit_all(
            dry_run=True,
            poly_degrees=poly_degrees,
            spline_smooths=spline_smooths,
            spline_degrees=spline_degrees,
            gompertz=gompertz,
        )
        height = len(trims)
        width = (
            len(col_names_base) * 4 + 3
        )  # entry for [mean, var, msqe, score] + [trim_percent, trim_low, trim_high]

        # arr will be converted into the final DataFrame
        arr = np.zeros([height, width], dtype=np.float32)
        all_trim_unfolds = []
        for i, trim in enumerate(trims):
            trimmed = np.array(trim["eigs"])
            lower_trim_length = find_first(eigs, trimmed[0])
            upper_trim_length = len(eigs) - 1 - find_last(eigs, trimmed[-1])
            trim_unfolds, sqes, smoother_map = Smoother(trimmed).fit_all(
                poly_degrees, spline_smooths, spline_degrees, gompertz
            )
            all_trim_unfolds.append(trim_unfolds)
            msqes = sqes.mean()  # type: ignore
            trim_percent = np.round(100 * (1 - len(trimmed) / len(eigs)), 3)
            lower_trim_percent = 100 * lower_trim_length / len(eigs)
            upper_trim_percent = 100 * upper_trim_length / len(eigs)

            # 3 columns of values per trim
            arr[i, 0] = trim_percent
            arr[i, 1] = lower_trim_percent
            arr[i, 2] = upper_trim_percent

            for j, col in enumerate(
                trim_unfolds
            ):  # get summary starts for each unfolding by smoother
                unfolded = np.array(trim_unfolds[col])
                mean, var, score = self.__evaluate_unfolding(unfolded)
                # arr[i, 0] is trim_percent, [i,1] is lower_trim_percent, etc, up tp
                # arr[i, 2], which has the upper_trim_percent
                # 4 additional columns of values per smoother:
                arr[i, 4 * j + 3] = mean
                arr[i, 4 * j + 4] = var
                arr[i, 4 * j + 5] = msqes[col]
                arr[i, 4 * j + 6] = score

        col_names_final = ["trim_percent", "trim_low", "trim_high"]
        # much match order added above
        for name in col_names_base:
            col_names_final.append(f"{name}--mean_spacing")
            col_names_final.append(f"{name}--var_spacing")
            col_names_final.append(f"{name}--msqe")
            col_names_final.append(f"{name}--score")
        trim_report = pd.DataFrame(data=arr, columns=col_names_final)
        return trim_report, all_trim_unfolds

    @staticmethod
    def __evaluate_unfolding(unfolded: ndarray) -> Tuple[float, float, float]:
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


class Trimmed(EigVals):
    def __init__(self, trimmed: ndarray):
        super().__init__(trimmed)

    @property
    def values(self) -> ndarray:
        return self._vals

    @property
    def vals(self) -> ndarray:
        return self._vals

    def unfold(
        self,
        smoother: SmoothMethod = "poly",
        degree: int = DEFAULT_POLY_DEGREE,
        spline_smooth: float = DEFAULT_SPLINE_SMOOTH,
        emd_detrend: bool = False,
    ) -> Unfolded:
        """
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

        Returns
        -------
        unfolded: ndarray
            the unfolded eigenvalues

        steps: ndarray
            the step-function values
        """
        eigs = self.values
        unfolded, _, closure = Smoother(eigs).fit(
            smoother=smoother,
            degree=degree,
            spline_smooth=spline_smooth,
            emd_detrend=emd_detrend,
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
            A float between 0 and 1, and which is passed as the tolerance paramater for
            [HBOS](https://pyod.readthedocs.io/en/latest/pyod.models.html#module-pyod.models.hbos)
            histogram-based outlier detection
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
