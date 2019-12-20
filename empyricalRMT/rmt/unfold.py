import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sbn

from colorama import Fore, Style
from numba import jit
from numpy.polynomial.polynomial import polyfit, polyval
from pathlib import Path
from pyod.models.hbos import HBOS
from PyEMD import EMD
from scipy.interpolate import UnivariateSpline as USpline
from scipy.optimize import curve_fit, minimize_scalar
from scipy.stats import mode
from warnings import warn

from ..rmt.construct import generateGOEMatrix
from ..rmt.detrend import emd_detrend
from ..rmt.eigenvalues import getEigs, stepFunctionG, stepFunctionVectorized
from ..rmt.observables.spacings import computeSpacings
from ..rmt.observables.rigidity import slope as fullSlope
from ..rmt.plot import setup_plotting

# from utils import eprint
from ..utils import (
    eprint,
    find_first,
    find_last,
    is_symmetric,
    mkdirp,
    write_block,
    write_in_place,
)

RESET = Style.RESET_ALL
EXPECTED_GOE_VARIANCE = 0.286
EXPECTED_GOE_MEAN = 1.000


def spline(
    eigs: np.array, knots: int = 3, detrend=None, percent=None, plot=False
) -> np.array:
    steps = stepFunctionVectorized(eigs, eigs)
    spl = USpline(eigs, steps, s=len(eigs) ** 1.4)
    unfolded = np.sort(spl(eigs))

    if detrend is True:
        return emd_detrend(unfolded)
    if plot:
        plot_unfold_fit(eigs, eigs, unfolded, section="full", curve="Full 3-Spline")
    return unfolded


# TODO: rename -> naive_polynomial()
def polynomial(
    eigs: np.array,
    degree: int = 5,
    grid_size: int = 10000,
    detrend=None,
    percent="auto",
    plot=False,
) -> np.array:
    """Now assumes trimmed eigenvalues. Percent determines percent of eigenvalues used to fit"""

    # grid = generate_grid(eigs, grid_size)
    if percent == "auto":
        # fit_start, fit_end = find_fit_range(eigs)
        # trimmed = eigs[fit_start:fit_end]
        # grid = np.linspace(trimmed.min(), trimmed.max(), grid_size)
        # step_vals = stepFunctionVectorized(trimmed, grid)
        # poly_coef = polyfit(grid, step_vals, degree)
        step_vals = stepFunctionVectorized(eigs, eigs)
        poly_coef = polyfit(eigs, step_vals, degree)
        unfolded = np.sort(polyval(eigs, poly_coef))
        if plot:
            # plot_unfold_fit(grid, step_vals, poly_coef)
            plot_unfold_fit(eigs, step_vals, poly_coef)
        return unfolded

    start, end, mid_poly_coef = iter_polyfit(eigs, degree=degree)

    start_eigs = eigs[:start]
    start_steps = stepFunctionVectorized(eigs, start_eigs)
    # start_params = curve_fit(gompertz, start_eigs, start_steps, p0=[np.max(start_steps), 1, 1])[0]
    # start_sig_fit = gompertz(start_eigs, *start_params)
    # plot_unfold_fit(eigs, start_eigs, start_sig_fit, "start", "Gompertz")
    start_spl = USpline(start_eigs, start_steps, k=3)
    start_spl_fit = start_spl(start_eigs)
    plot_unfold_fit(eigs, start_eigs, start_spl_fit, "start", "3-Spline")

    mid_eigs = eigs[start:end]
    mid_steps = stepFunctionVectorized(eigs, mid_eigs)
    mid_poly_coef = polyfit(mid_eigs, mid_steps, degree)
    mid_poly_vals = polyval(mid_eigs, mid_poly_coef)
    plot_unfold_fit(eigs, mid_eigs, mid_poly_vals, "middle")

    end_eigs = eigs[end:]
    end_steps = stepFunctionVectorized(eigs, end_eigs)
    # end_poly_coef = polyfit(end_eigs, end_steps, 3)
    # end_poly_vals = polyval(end_eigs, end_poly_coef)
    # plot_unfold_fit(eigs, end_eigs, end_poly_vals, "end")
    # end_params = curve_fit(
    #     gompertz,
    #     end_eigs,
    #     end_steps,
    #     method="trf",
    #     p0=[np.max(end_steps), 1, 1],
    #     bounds=([0.999*np.max(end_steps), -np.inf, -np.inf], [np.max(end_steps), np.inf, np.inf])
    # )[0]
    # end_sig_fit = gompertz(end_eigs, *end_params)
    # plot_unfold_fit(eigs, end_eigs, end_sig_fit, "end", "Gompertz")
    end_spl = USpline(end_eigs, end_steps, k=3)
    end_spl_fit = end_spl(end_eigs)
    plot_unfold_fit(eigs, end_eigs, end_spl_fit, "end", "3-Spline")

    # step_vals = stepFunctionVectorized(eigs, grid)
    # poly_coef = polyfit(grid, step_vals, degree)
    # unfolded = np.hstack([start_sig_fit, mid_poly_vals, end_sig_fit])
    unfolded = np.hstack([start_spl_fit, mid_poly_vals, end_spl_fit])
    unfolded = np.sort(unfolded)
    if len(unfolded) != len(eigs):
        raise ValueError("Eigenvalues were not unfolded correctly")
    # unfolded = np.sort(polyval(eigs[start:end], poly_coef))

    if plot:
        # plot_unfold_fit(grid, step_vals, poly_coef)
        step_vals = stepFunctionVectorized(eigs, eigs)
        fitted = unfolded
        df = pd.DataFrame({"λ": eigs, "fitted": fitted, "N(λ)": step_vals})
        axes = sbn.lineplot(data=df, x="λ", y="N(λ)", label="N(λ)", color="black")
        fit = axes.scatter(eigs, fitted, label="Fitted / Unfolded Values")
        plt.setp(fit, color="#F01DB8")
        plt.title("Fit of total unfolding curve")
        plt.legend()
        plt.show()

    if detrend is True:
        return emd_detrend(unfolded)
    return unfolded


# Morales et al. (2011) DOI: 10.1103/PhysRevE.84.016203
def test(method="spline", param=9, percent: int = None, title="Default"):
    sbn.set(rc={"lines.linewidth": 0.9})

    M = np.corrcoef(generateGOEMatrix(400))  # get corr matrix
    if not is_symmetric(M):
        raise ValueError("Non-symmetric matrix generated")
    eigs = getEigs(M)  # already sorted ascending

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


def inspect_eigs(eigs, partitions):
    startpoints = [x for x in range(0, len(eigs), partitions)]
    startpoints.append(len(eigs))
    n_partitions = len(startpoints) - 1
    means = np.empty([n_partitions])
    for i in range(n_partitions):
        means[i] = np.mean(eigs[startpoints[i] : startpoints[i + 1]])
    print(means)
    return means


def iter_polyfit(eigs: np.array, degree: int = 11) -> np.array:
    poly_coef = None
    mape = 9e999
    mapes = []
    slopes = []
    step_vals = stepFunctionVectorized(eigs, eigs)
    last_fitted = None
    start, end = 0, len(eigs)
    # find_fit_range(eigs)
    while mape > 25:
        poly_coef = polyfit(eigs[start:end], step_vals[start:end], degree)
        poly_vals = polyval(eigs[start:end], poly_coef)
        mape = np.mean(np.abs(poly_vals - step_vals))
        mapes.append(mape)
        if len(mapes) < 5:
            local_mapes = np.array(mapes)
        else:
            local_mapes = np.array(mapes)[-10:]
        mapes_slope = slope(local_mapes)
        slopes.append(mapes_slope)

        percent_fit = np.round(100 * len(eigs[start:end]) / len(eigs), 2)
        if len(slopes) > 10:
            slopes_slope = slope(np.array(slopes)[-20:])
            if slopes_slope > 0:
                write_in_place(
                    f"Fitting {str(percent_fit).ljust(5)}% of eigenvalues. MAE",
                    mape,
                    Fore.YELLOW,
                )
            else:
                write_in_place(
                    f"Fitting {str(percent_fit).ljust(5)}% of eigenvalues. MAE",
                    mape,
                    Fore.GREEN,
                )
        right_max = np.abs(eigs[end - 1])
        left_max = np.abs(eigs[start])
        if right_max > left_max and percent_fit > 95:
            end -= 10
        elif right_max > left_max:
            end -= 5
        elif right_max < left_max and percent_fit > 95:
            start += 2
        else:
            start += 1

    return start, end, poly_coef


# TODO: Fit a spline to the outliers at each iteration
def manual_outlier_find(eigs, degree=5, tolerance=0.1) -> [np.array, np.array]:
    setup_plotting()
    steps = stepFunctionVectorized(eigs, eigs)
    hb = HBOS(tol=tolerance)
    X = np.vstack([eigs, steps]).T
    fits = []
    iterations = [X]
    total_removed = 0
    iter_count = 0
    while iter_count < 20:
        labs = hb.fit(X).labels_
        str_labels = ["Outlier" if label else "Inlier" for label in labs]
        mask = np.array(1 - labs, dtype=bool)  # boolean array for selecting inliers
        out_mask = np.array(labs, dtype=bool)  # boolean array for selecting outliers
        orig = X  # save for plotting context
        outs = X[out_mask, :]
        X = X[mask, :]  # boolean / mask indicing always copies anyway
        x, y = X[:, 0], X[:, 1]
        poly_fitted = polyval(orig[:, 0], polyfit(orig[:, 0], orig[:, 1], degree))
        lin_fit = polyval(orig[:, 0], polyfit(orig[:, 0], orig[:, 1], 1))
        fits.append(poly_fitted)
        residuals = poly_fitted - orig[:, 1]
        lin_res = lin_fit - orig[:, 1]
        mae = np.mean(np.abs(residuals))
        lin_mae = np.mean(np.abs(lin_res))
        mape = np.mean(np.abs(residuals / orig[:, 1]))
        lin_mape = np.mean(np.abs(lin_res / orig[:, 1]))
        mean_resid = np.mean(residuals)
        lin_mean_resid = np.mean(lin_res)
        med_resid = np.median(residuals)
        lin_med_resid = np.median(lin_res)
        resid_var = np.var(residuals)
        lin_resid_var = np.var(lin_res)

        # Print info to terminal
        total_removed += np.sum(out_mask)
        removed = np.round(100 * total_removed / len(eigs), 2)
        will_remove = np.round(np.sum(out_mask) / len(eigs), 2)
        outlier_color = Fore.WHITE
        if removed < 5:
            outlier_color = Fore.GREEN
        elif removed < 10:
            outlier_color = Fore.CYAN
        elif removed < 15:
            outlier_color = Fore.YELLOW
        else:
            outlier_color = Fore.RED
        remove_color = Fore.WHITE
        if will_remove < 5:
            remove_color = Fore.GREEN
        elif will_remove < 10:
            remove_color = Fore.CYAN
        elif will_remove < 15:
            remove_color = Fore.YELLOW
        else:
            remove_color = Fore.RED

        write_block(
            [
                f"Outliers removed so far:               {outlier_color}{removed}%{Fore.RESET}",
                f"Outliers identified this iteration:    {remove_color}{will_remove}%{Fore.RESET}",
                f"Mean Absolute Error:                   {mae}",
                f"Mean Absolute Percentage Error:        {mape}",
                f"Mean Residual:                         {mean_resid}",
                f"Median Residual:                       {med_resid}",
                f"Residual variance:                     {resid_var}",
                f"Linear Mean Absolute Error:            {lin_mae}",
                f"Linear Mean Absolute Percentage Error: {lin_mape}",
                f"Linear Mean Residual:                  {lin_mean_resid}",
                f"Linear Median Residual:                {lin_med_resid}",
                f"Linear Residual variance:              {lin_resid_var}",
                "",
                "Press `Enter` to continue removing outliers.",
                "Type `back` or `b` to go to the previous set of inliers.",
                "Type `ok` or `done` or `d` to accept the current fit.",
            ],
            border=None,
        )

        # df = pd.DataFrame({"λ": X[:, 0].T, "N(λ)": X[:, 1].T})
        # sbn.scatterplot(data=df, x="λ", y="N(λ)", edgecolors="none", linewidth=0)
        # plt.title(f"Inliers so far ({removed}% removed)")
        # plt.show()
        # df = pd.DataFrame({"λ": outs[:, 0].T, "N(λ)": outs[:, 1].T})
        # sbn.scatterplot(data=df, x="λ", y="N(λ)", color="red", linewidth=0)
        # plt.title(f"Removed outliers")
        # plt.show()
        # plot outliers in context
        df = pd.DataFrame(
            {"λ": orig[:, 0].T, "N(λ)": orig[:, 1].T, "Cluster": str_labels}
        )
        axes = sbn.scatterplot(
            data=df,
            x="λ",
            y="N(λ)",
            hue="Cluster",
            style="Cluster",
            style_order=["Inlier", "Outlier"],
            linewidth=0,
            legend="brief",
            markers=[".", "X"],
            palette=["black", "red"],
            hue_order=["Inlier", "Outlier"],
        )
        # remove ugly legend https://stackoverflow.com/a/51579663
        handles, labels = axes.get_legend_handles_labels()

        plt.title(
            f"Outlier Fit Effects (fitting {100 - removed}% of all original eigenvalues)"
        )
        poly = axes.plot(orig[:, 0], poly_fitted, label="Polynomial Fit")
        plt.setp(poly, color="#0066FF")
        lin = axes.plot(orig[:, 0], lin_fit, label="Linear Fit")
        plt.setp(lin, color="#08FD4F")
        plt.legend(loc="lower right")
        plt.show()

        loop_done = False
        go_back = False
        while not loop_done:
            command = input()
            if command in ["", "\n", "\r", "\r\n", "\n\r"]:
                iterations.append(X)  # move onto next iteration, save result
                loop_done = True
            elif command.find("back") > -1 or command == "b":
                iter_count -= 1
                iterations.pop(), fits.pop(), fits.pop()
                X = iterations.pop()
                total_removed -= np.sum(out_mask)
                loop_done = True
            elif command.find("ok") > -1 or command.find("done") > -1 or command == "d":
                return x, y
            else:
                write_block(
                    f"Outliers removed so far:               {outlier_color}{removed}%{Fore.RESET}",
                    f"Outliers identified this iteration:    {remove_color}{will_remove}%{Fore.RESET}",
                    f"Mean Absolute Error:                   {mae}",
                    f"Mean Absolute Percentage Error:        {mape}",
                    f"Mean Residual:                         {mean_resid}",
                    f"Median Residual:                       {med_resid}",
                    f"Residual variance:                     {resid_var}",
                    f"Linear Mean Absolute Error:            {lin_mae}",
                    f"Linear Mean Absolute Percentage Error: {lin_mape}",
                    f"Linear Mean Residual:                  {lin_mean_resid}",
                    f"Linear Median Residual:                {lin_med_resid}",
                    f"Linear Residual variance:              {lin_resid_var}",
                    "",
                    "Press `Enter` to continue removing outliers.",
                    "Type `back` or `b` to go to the previous set of inliers.",
                    "Type `ok` or `done` or `d` to accept the current fit.",
                    border=None,
                )
    return eigs, steps


def manual_fit(eigs: np.array, method="poly", degree: int = 11) -> np.array:
    poly_coef = None
    mape = 9e999
    mapes = []
    slopes = []
    last_fitted = None
    start, end = 0, len(eigs)
    step_vals = stepFunctionVectorized(eigs, eigs)
    # find_fit_range(eigs)
    while mape > 25:
        last_fitted = eigs[start:end]
        # step_vals = stepFunctionVectorized(eigs, eigs[start:end])
        poly_coef = polyfit(eigs[start:end], step_vals[start:end], degree)
        poly_vals = polyval(eigs[start:end], poly_coef)
        mape = np.mean(np.abs(poly_vals - step_vals))
        resid = pd.DataFrame(
            {
                "index": np.arange(0, len(poly_vals)),
                "residuals": poly_vals - step_vals[start:end],
            }
        )
        mapes.append(mape)
        if len(mapes) < 5:
            local_mapes = np.array(mapes)
        else:
            local_mapes = np.array(mapes)[-10:]
        mapes_slope = slope(local_mapes)
        slopes.append(mapes_slope)

        percent_fit = np.round(100 * len(eigs[start:end]) / len(eigs), 2)
        if len(slopes) > 10:
            slopes_slope = slope(np.array(slopes)[-20:])
            if slopes_slope > 0:
                write_in_place(
                    f"Fitting {str(percent_fit).ljust(5)}% of eigenvalues. MAE",
                    mape,
                    Fore.YELLOW,
                )
            else:
                write_in_place(
                    f"Fitting {str(percent_fit).ljust(5)}% of eigenvalues. MAE",
                    mape,
                    Fore.GREEN,
                )
        right_max = np.abs(eigs[end - 1])
        left_max = np.abs(eigs[start])
        if right_max > left_max and percent_fit > 95:
            end -= 10
        elif right_max > left_max:
            end -= 5
        elif right_max < left_max and percent_fit > 95:
            start += 2
        else:
            start += 1

    return start, end, poly_coef


def plot_unfold_fit(full_eigs, fit_eigs, fit_values, section: str, curve=None):
    step_vals = stepFunctionVectorized(full_eigs, fit_eigs)
    df = pd.DataFrame({"λ": fit_eigs, "fitted": fit_values, "N(λ)": step_vals})
    axes = sbn.scatterplot(data=df, x="λ", y="N(λ)", label="N(λ)", color="black")
    plot_range = np.abs((fit_eigs.max() - fit_eigs.min()) / 10)
    axes.set_xlim(left=fit_eigs.min() - plot_range, right=fit_eigs.max() + plot_range)
    fit = axes.scatter(fit_eigs, fit_values, label="Fitted / Unfolded Values")
    if section == "middle":
        plt.setp(fit, color="#FD8208")
    elif section == "start":
        plt.setp(fit, color="#0066FF")
    elif section == "end":
        plt.setp(fit, color="#FF0042")
    elif section in ["full", "all"]:
        plt.setp(fit, color="#F01DB8")
    if curve is not None:
        plt.title(f"Fit of {section} of unfolding polynomial ({curve})")
    else:
        plt.title(f"Fit of {section} of unfolding polynomial")
    plt.legend()
    plt.show()


def print_fit_info(x, y, fitted):
    mae = np.mean(np.abs(fitted - y))
    mean_resid = np.mean(fitted - y)
    var_resid = np.var(fitted - y)
    write_block(
        [
            "{:30}: {:.5e}".format("Mean Absolute Error", mae),
            "{:30}: {:>12}".format("Mean of residuals", mean_resid),
            "{:30}: {:>12}".format("Residual variance", var_resid),
            "{:30}: {:>12}".format(),
        ]
    )


@jit(nopython=True, fastmath=True)
def slope(arr: np.array) -> np.float64:
    x = np.arange(0, len(arr))
    return fullSlope(x, arr)


@jit(nopython=True, fastmath=True)
def derivative(x: np.array, y=np.array) -> np.array:
    res = np.empty(x.shape, dtype=np.float64)
    # for i = 1 (i.e. y[1], we compute (y[0] - y[2]) / 2*spacing)
    # ...
    # for i = L - 2, we compute (y[L-3] - y[L-1]) / 2*spacing
    # i.e. (y[0:L-2] - y[2:]) / 2*spacing
    L = len(x)
    res[1:-1] = (y[2:] - y[0 : L - 2]) / (x[2:] - x[0 : L - 2])
    res[0] = (y[1] - y[0]) / (x[1] - x[0])
    res[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
    return res


@jit(nopython=True, fastmath=True)
def gompertz(t, a, b, c):
    return a * np.exp(-b * np.exp(-c * t))


@jit(nopython=True, fastmath=True)
def inverse_gompertz(t, a, b, c):
    return np.log(b / np.log(1 / t)) / c


def generate_grid(eigs, grid_size) -> np.array:
    if grid_size < len(eigs):
        eprint("Grid less dense than eigenvalues. Choosing `grid_size` = 10*len(eigs).")
        grid_size = 10 * len(eigs)

    start = minimize_scalar(
        lambda x: stepFunctionG(eigs, x),
        method="golden",
        tol=1e-30,
        options={"xtol": 1e-30},
    ).x
    end = minimize_scalar(
        lambda x: -stepFunctionG(eigs, x),
        method="golden",
        tol=1e-30,
        options={"xtol": 1e-30},
    ).x
    grid = np.linspace(start, end, len(eigs))
    steps = np.array(stepFunctionVectorized(eigs, grid), dtype=np.int)
    while (mode(steps).count[0]) > 1:
        end -= 5e-6
        grid = np.linspace(start, end, len(eigs))
        steps = np.array(stepFunctionVectorized(eigs, grid), dtype=np.int)
        print(mode(steps).count[0])
    while len(steps[steps == 0]) > 1:
        start += 5e-6
        grid = np.linspace(start, end, len(eigs))
        steps = np.array(stepFunctionVectorized(eigs, grid), dtype=np.int)
        print(len(steps[steps == 0]))
    return np.linspace(start, end, grid_size)

    # maxend = int(stepFunctionG(eigs, end))
    # while int(stepFunctionG(eigs, end)) == maxend:
    #     end -= 5e-6
    # print(f"Minimizing x: {start}, Maximizing x: {end}")


class UnfoldOptions:
    """Basically you pass in a dict like:
    {
        "smooth_function": "poly" | "spline" | "gompertz" | lambda | None,
        "poly_degree": int | "auto" | None,
        "spline_degree": int | None,
        "spline_smooth": float | "heuristic" | None,
        "emd_detrend": boolean | None,
        "method": "auto" | "manual" | None,
    }
    """

    def __init__(
        self,
        smooth_function="poly",
        poly_degree=8,
        spline_degree=None,
        spline_smooth=None,
        emd_detrend=False,
        method=None,
        options=None,
    ):
        """ if `options` is not None, only look at `options` argument. If options is
        None, ignore `options` argument.
        """
        if options is not None:
            self.options = self.__validate_dict(options)
            return

        options = self.__validate_dict(self.__default(self))
        options["smooth_function"] = smooth_function
        options["poly_degree"] = poly_degree
        options["spline_degree"] = spline_degree
        options["spline_smooth"] = spline_smooth
        options["emd_detrend"] = emd_detrend
        options["method"] = method
        self.options = self.__validate_dict(options)

    @staticmethod
    def __default(self) -> dict:
        default = {
            "smooth_function": "poly",
            "poly_degree": 8,
            "spline_degree": 3,
            "spline_smooth": None,
            "emd_detrend": False,
            "method": None,
        }
        return default

    @property
    def smoother(self):
        return self.options["smooth_function"]

    @property
    def degree(self):
        return self.options["poly_degree"]

    @property
    def spline_degree(self):
        return self.options["spline_degree"]

    @property
    def spline_smooth(self):
        return self.options["spline_smooth"]

    @property
    def emd(self):
        return self.options["emd_detrend"]

    @property
    def method(self):
        return self.options["method"]

    def __validate_dict(self, options: dict):
        func = options.get("smooth_function")
        degree = options.get("poly_degree")
        spline_degree = options.get("spline_degree")
        spline_smooth = options.get("spline_smooth")
        emd = options.get("emd_detrend")
        method = options.get("method")

        if func == "poly":
            if degree is None:
                degree = self.__default["poly_degree"]
                warn(
                    f"No degree set for polynomial unfolding. Will default to polynomial of degree {degree}.",
                    category=UserWarning,
                )
            if not isinstance(degree, int):
                raise ValueError("Polynomial degree must be of type `int`")
            if degree < 3:
                raise ValueError("Unfolding polynomial must have minimum degree 3.")
        elif func == "spline":
            if spline_degree is None:
                spline_degree = 3
            if not isinstance(spline_degree, int):
                raise ValueError("Degree of spline must be an int <= 5")
            if spline_degree > 5:
                raise ValueError("Degree of spline must be an int <= 5")
            if spline_smooth is not None and spline_smooth != "heuristic":
                spline_smooth = float(spline_smooth)

        if emd is None:
            emd = False
        elif isinstance(emd, bool):
            pass
        else:
            raise ValueError("UnfoldOption `emd` can only be either True or False.")

        if method is None or method == "auto" or method == "manual":
            pass
        else:
            raise ValueError(
                "UnfoldOption `method` must be one of 'auto', 'manual', or 'None'"
            )

        return {
            "smooth_function": func,
            "poly_degree": degree,
            "spline_degree": spline_degree,
            "spline_smooth": spline_smooth,
            "emd_detrend": emd,
            "method": method,
        }


class Unfolder:
    """Base class for storing eigenvalues, trimmed eigenvalues, and
    unfolded eigenvalues"""

    def __init__(self, eigs, options: UnfoldOptions = UnfoldOptions()):
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

        if not isinstance(options, UnfoldOptions):
            raise ValueError("`options` argument must be of type `UnfoldOptions")

        self.__unfold_options = UnfoldOptions(options=options.options)
        self.__raw_eigs = np.array(eigs)
        self.__sorted_eigs = np.sort(self.__raw_eigs)
        self.__trimmed_eigs = None
        self.__trimmed_indices = (None, None)
        self.__trimmed_steps = []
        return

    @property
    def eigenvalues(self) -> np.array:
        """get the original (sorted) eigenvalues as a numpy array"""
        return self.__sorted_eigs

    @property
    def eigs(self) -> np.array:
        """get the original (sorted) eigenvalues as a numpy array (alternate)"""
        return self.__sorted_eigs

    def trim(self, method="auto", smoother="polynomial", outlier_tol=0.1):
        """compute the optimal trim region and fit statistics"""
        print("Trimming to central eigenvalues.")
        method = self.__unfold_options.method

        eigs = self.eigs
        unfolded, steps = self.__fit(eigs)

        if method == "auto":
            self.__trimmed_steps = self.__collect_outliers(eigs, steps)

    def trim_report_summary(self, show_plot=True, save_plot: Path = None):
        if len(self.__trimmed_steps) == 0:
            raise RuntimeError(
                "Eigenvalues have not been trimmed yet. Call Unfolder.trim() "
                "before attempting to generate a trim summary."
            )
        self.__plot_outliers(show_plot, save_plot)
        report = self.trim_report()
        scores = report.filter(regex=".*score.*").abs()

        # get column names so we don't have to deal with terrible Pandas return types
        score_cols = np.array(scores.columns.to_list())
        # gives column names of columns with lowest scores
        best_smoother_cols = list(scores.abs().min().sort_values()[:3].to_dict().keys())
        # indices of rows with best scores
        best_smoother_rows = report[best_smoother_cols].abs().idxmin().to_list()

        # construct dict with trim amounts of best overall scoring smoothers
        best_smoothers = {}
        trim_cols = ["trim_percent", "trim_low", "trim_high"]
        for i, col in enumerate(best_smoother_cols):
            min_score_i = best_smoother_rows[i]
            cols = trim_cols + [col.replace("score", "mean_spacing"), col.replace("score", "var_spacing"), col]
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

        return report, best_smoothers, consistent

    def trim_report(self):
        """Generate a dataframe showing the unfoldings that results from different
        trim percentages, and different choices of smoothing functions.
        """
        trims = self.__trimmed_steps
        eigs = self.eigs

        # trim_percents = [np.round(100*(1 - len(trim["eigs"]) / len(self.eigs)), 3) for trim in trims]
        col_names_base = self.__fit_all(dry_run=True)
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
        return pd.DataFrame(data=arr, columns=col_names_final)

    def unfold(self):
        pass

    def __fit(
        self,
        eigs,
        method=None,
        poly_degree=None,
        spline_smooth=None,
        spline_degree=None,
    ) -> (np.array, np.array):
        eigs = np.array(eigs)
        if method is None:
            smoother = self.__unfold_options.smoother
        else:
            smoother = method
        steps = stepFunctionVectorized(eigs, eigs)
        if smoother == "poly":
            if poly_degree is None:
                degree = self.__unfold_options.degree
            else:
                degree = poly_degree
            poly_coef = polyfit(eigs, steps, degree)
            unfolded = np.sort(polyval(eigs, poly_coef))
            return unfolded, steps
        if smoother == "spline":
            if spline_degree is None:
                k = self.__unfold_options.spline_degree
            else:
                try:
                    k = int(spline_degree)
                except BaseException as e:
                    print(ValueError("Cannot convert spline degree to int."))
                    raise e
            smoothing = self.__unfold_options.spline_smooth
            if smoothing == "heuristic":
                spline = USpline(eigs, steps, k=int(k), s=len(eigs) ** 1.4)
            elif spline_smooth is not None:
                if not isinstance(spline_smooth, float):
                    raise ValueError("Spline smoothing factor must be a float")
                spline = USpline(eigs, steps, k=int(k), s=len(eigs) ** spline_smooth)
            else:
                spline = USpline(eigs, steps, k=int(k), s=smoothing)
            return np.sort(spline(eigs)), steps
        if smoother == "gompertz":
            # use steps[end] as guess for the asymptote, a, of gompertz curve
            [a, b, c], cov = curve_fit(gompertz, eigs, steps, p0=(steps[-1], 1, 1))
            return np.sort(gompertz(eigs, a, b, c)), steps

    def __fit_all(
        self,
        eigs=None,
        poly_degrees=[3, 4, 5, 6, 7, 8, 9, 10, 11],
        spline_smooths=np.linspace(1, 2, num=11),
        spline_degrees=[3],
        dry_run=False,
    ) -> pd.DataFrame:
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
        col_names = []
        if dry_run:  # early return strings of colums names
            for d in poly_degrees:
                col_names.append(f"poly_{d}")
            for s in spline_smooths:
                for deg in spline_degrees:
                    col_name = f"{spline_name(deg)}-spline_" "{:1.1f}".format(s)
                    col_names.append(col_name)
            col_names.append("gompertz")
            return col_names

        for d in poly_degrees:
            col_name = f"poly_{d}"
            unfolded, _ = self.__fit(eigs, method="poly", poly_degree=d)
            df[col_name] = unfolded
        for s in spline_smooths:
            for deg in spline_degrees:
                col_name = f"{spline_name(deg)}-spline_" "{:1.1f}".format(s)
                if dry_run:
                    col_names.append(col_name)
                    break
                unfolded, _ = self.__fit(
                    eigs, method="spline", spline_smooth=s, spline_degree=deg
                )
                df[col_name] = unfolded
        if dry_run:
            col_names.append("gompertz")
        else:
            df["gompertz"], _ = self.__fit(eigs, method="gompertz")
            return df

    def __collect_outliers(self, eigs, steps, tolerance=0.1):
        iter_results = [
            pd.DataFrame(
                {
                    "eigs": eigs,
                    "steps": steps,
                    "unfolded": self.__fit(eigs)[0],
                    "cluster": ["inlier" for _ in eigs],
                }
            )
        ]

        while (
            len(iter_results[-1]) / len(eigs)
        ) > 0.5:  # terminate if we have trimmed half
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
            df["unfolded"], _ = self.__fit(np.array(df["eigs"]))
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

    def _evaluate_unfolding(self, unfolded) -> [float, float, float]:
        """Calculate a naive unfolding score via comparison to the expected mean and
        variance of the level spacings of GOE matrices. Positive scores indicate
        there is too much variability in the unfolded eigenvalue spacings, negative
        scores indicate too little. Best score is zero.
        """
        spacings = unfolded[1:] - unfolded[:-1]
        mean, var = np.mean(spacings), np.var(spacings, ddof=1)
        mean_weight = (
            0.05
        )  # variance gets weight 1, i.e. mean is 0.05 times as important
        mean_norm = (mean - EXPECTED_GOE_MEAN) / EXPECTED_GOE_MEAN
        var_norm = (var - EXPECTED_GOE_VARIANCE) / EXPECTED_GOE_VARIANCE
        score = var_norm + mean_weight * mean_norm
        return mean, var, score
