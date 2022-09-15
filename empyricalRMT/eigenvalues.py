from __future__ import annotations

from time import strftime
from typing import Any, List, Optional, Tuple, Type, Union
from warnings import warn

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from numpy import ndarray
from numpy.typing import ArrayLike
from scipy.integrate import quad
from typing_extensions import Literal

from empyricalRMT._constants import DEFAULT_POLY_DEGREE, DEFAULT_POLY_DEGREES, DEFAULT_SPLINE_SMOOTH
from empyricalRMT._eigvals import EigVals
from empyricalRMT._types import MatrixKind, fArr
from empyricalRMT.construct import _generate_GOE_matrix, _generate_GOE_tridiagonal
from empyricalRMT.correlater import correlate_fast
from empyricalRMT.detrend import emd_detrend
from empyricalRMT.signalproc.detrend import DetrendMethod
from empyricalRMT.signalproc.detrend import detrend as _detrend
from empyricalRMT.smoother import Smoother, SmoothMethod
from empyricalRMT.trim import Trimmed, TrimReport
from empyricalRMT.unfold import Unfolded


class Eigenvalues(EigVals):
    """Basic class providing access to various items of interest in RMT."""

    __WARNED_SMALL = False

    def __init__(
        self,
        values: ArrayLike,
        kind: Optional[MatrixKind] = None,
        detrending: Optional[DetrendMethod] = None,
    ):
        """Construct an Eigenvalues object.

        Parameters
        ----------
        values: ArrayLike
            If 0D or 1D, then `values` are assumed to be eigenvalues. If 2D, then
            eigs will be computed via np.linalg.eigvalsh or np.linalg.eigvals, depending
            on if the matrix is symmetric Hermititian or not.
        """
        super().__init__(values)
        if len(self._vals) < 50 and not self.__class__.__WARNED_SMALL:
            warn(
                "You have less than 50 eigenvalues, and the assumptions of Random "
                "Matrix Theory are almost certainly not justified. Any results "
                "obtained should be interpreted with caution",
                category=UserWarning,
            )
            self.__class__.__WARNED_SMALL = True  # don't warn more than once per execution

        self.kind: Optional[MatrixKind] = kind
        self.deterending: Optional[DetrendMethod] = detrending
        self._series_T: Optional[int] = None
        self._series_N: Optional[int] = None
        # get some Marchenko-Pastur endpoints
        self._marchenko: Optional[Tuple[float, float]] = None
        self._marchenko_shifted: Optional[Tuple[float, float]] = None

    @staticmethod
    def generate(
        matsize: int,
        kind: MatrixKind = MatrixKind.GOE,
        seed: Optional[int] = None,
        log_time: bool = False,
        use_tridiagonal: bool = True,
    ) -> Eigenvalues:
        """Generate a random square matrix and return the computed eigenvalues.

        Parameters
        ----------
        matsize: int
            The size (e.g. width or height) of the square matrix that will be
            generated.

        kind: "goe" | "gue" | "poisson" | "gde" | "uniform"
            From which ensemble to sample the matrix:

               - "goe" is a matrix from the Gaussian Orthogonal Ensemble
               - "gue" is a matrix from the Gaussian Unitary Ensemble
               - "poisson" or "gde" is a "Gaussian Diagonal Ensemble", a
                 diagonal matrix with all entries being samples from a standard
                 normal distribution.
               - "uniform" is currently unimplemented

        seed: int
            Seed value for `np.random.seed`. Ensures reproducible results.

        log_time: bool
            Whether or not to log to stdout the start and end times for
            computing eigenvalues.

        use_tridiagonal: bool
            For `kind` "goe" only. Generate a tridiagonal matrix with identical
            eigenvalue distributions to a GOE matrix instead of a full GOE
            matrix. *Dramatically* speeds up computation of eigenvalues, and is
            recommended for generating matrices of approximately size N >= 2000.
        """
        kind = MatrixKind.validate(kind)
        if kind == MatrixKind.Poisson:
            np.random.seed(seed)
            # eigenvalues of diagonal are just the entries
            return Eigenvalues(np.sort(np.random.standard_normal(matsize)), kind=kind)
        elif kind == MatrixKind.Uniform:
            raise NotImplementedError("Uniform random matrices not implemeted.")
        elif kind == MatrixKind.GUE:
            size = [matsize, matsize]
            if seed is not None:
                np.random.seed(seed)
            A = np.random.standard_normal(size) + 1j * np.random.standard_normal(size)
            M: fArr = (A + A.conjugate().T) / 2  # type: ignore
        elif kind == MatrixKind.GOE:
            if matsize > 500 and use_tridiagonal:
                M = _generate_GOE_tridiagonal(size=matsize, seed=seed)
            else:
                M = _generate_GOE_matrix(matsize, seed=seed)
        else:
            kinds = [e.value for e in MatrixKind]  # type: ignore[unreachable]
            raise ValueError(f"`kind` must be one of {kinds}")

        if log_time:
            print(f"\n{strftime('%H:%M:%S (%b%d)')} -- computing eigenvalues...")
        eigs: fArr = np.linalg.eigvalsh(M)
        if log_time:
            print(f"{strftime('%H:%M:%S (%b%d)')} -- computed eigenvalues.")
        return Eigenvalues(eigs, kind=kind)

    @classmethod
    def from_correlations(
        cls: Type[Eigenvalues],
        data: ndarray,
        atol: float = float(1e3 * np.finfo(np.float64).eps),
        lower: bool = True,
    ) -> Eigenvalues:
        """Use positive semi-definiteness to identify likely zero-valued eigenvalues
        due to floating point imprecision.

        Parameters
        ----------
        data: ndarray
            Either a 2-dimensional symmetric correlation matrix, or the
            1-dimensional computed eigenvalues from such a matrix.

        atol: float
            Absolute tolerance. Eigenvalues with absolute value less than _atol_
            will be considered equal to zero.

        lower: bool
            If _lower_ is True (default), use only the lower triangle to compute
            the eigenvalues. Otherwise, use the upper triangle.


        Returns
        -------
        eigenvalues: Eigenvalues
            The Eigenvalues object, with values close to zero pre-trimmed away.
        """
        if len(data.shape) > 2:
            raise ValueError("`data` must be either flat, 1-dimensional, or 2-dimensional.")
        if len(data.shape) == 2 and np.min(data.shape) > 1:
            eigs = np.linalg.eigvalsh(data, "L" if lower else "U")
        eigs = data.reshape(-1)  # equivalent to ravel but less likely to copy
        return cls(eigs[np.abs(eigs) < atol])

    # TODO: see if we can represent the full eigenvalue problem and solve that,
    # instead of going through the covariance matrix
    @classmethod
    def from_time_series(
        cls: Type[Eigenvalues],
        data: ndarray,
        covariance: bool = True,
        trim_zeros: bool = True,
        zeros: Union[float, Literal["negative"]] = "negative",
        time_axis: int = 1,
        use_sparse: bool = False,
        **sp_args: Any,
    ) -> Eigenvalues:
        """Use Marchenko-Pastur and positive semi-definiteness to identify likely noise
        values and zero-valued eigenvalues due to floating point imprecision

        Parameters
        ----------
        data: ndarray
            A 2-dimensional matrix of time-series data.

        covariance: bool
            If True (default) compute the eigenvalues of the covariance matrix.
            If False, use the correlation matrix.

        trim_zeros: bool
            If True (default) only return eigenvalues greater than `zeros`
            (e.g. remove values that are likely unstable or actually zero due
            to floating point precision limitations).

        zeros: float
            If a float, The smallest acceptable value for an eigenvalue not to
            be considered zero.
            If "negative", trim invalid negative eigenvalues (e.g. because
            coviarance and correlation matrices are positive semi-definite)
            If "heuristic", trim away eigenvalues likely to be unstable:
                - if computed eigenvaleus are `eigs`, and if `emin = eigs.min()`,
                  `emin < 0`, then trim to `eigs[eigs > 100 * np.abs(emin)]
                - if emin >= 0, trim to `eigs[eigs > 0]`

        time_axis: int
            If 0, assumes the data.shape == (n, T), where n is the number of
            features / variables, and T is the length (number of points) in each
            time series.

        use_sparse: bool
            Convert the interim correlation matrix to a sparse triangular
            matrix, and use `scipy.sparse.linalg.eigsh` to solve for the
            eigenvalues. This currently does not save memory (since we still
            compute an interim dense covariance matrix) but gives more control
            over what eigenvalues are returned.

        sp_args:
            Keyword arguments to pass to scipy.sparse.linalg.eigsh.


        Returns
        -------
        eigenvalues: Eigenvalues
            The Eigenvalues object, with extra time-series relevant data:
            - Eigenvalues.marcenko_endpoints: (float, float)
        """
        if len(data.shape) != 2:
            raise ValueError("Input `data` array must have dimension of 2.")
        if time_axis not in [0, 1]:
            raise ValueError("Invalid `time_axis`. Must be either 0 or 1.")
        if time_axis == 0:
            data = data.T

        N, T = data.shape
        M = None
        eigs: fArr
        if N <= T:  # no benefit from intermediate transposition
            M = np.cov(data, ddof=1) if covariance else correlate_fast(data, ddof=1)
            if use_sparse:
                M = scipy.sparse.tril(M)
                if sp_args.get("return_eigenvectors") is True:
                    raise ValueError(
                        "This function is intended only as a helper "
                        "to extract eigenvalues from time-series."
                    )

                eigs = np.array(scipy.sparse.linalg.eigsh(M, **sp_args))
            else:
                eigs = np.linalg.eigvalsh(M)
        else:
            eigs = _eigs_via_transpose(data, covariance=covariance)

        if trim_zeros:
            if zeros == "heuristic":
                e_min = eigs.min()  # type: ignore[unreachable]
                minval = 0
                if e_min <= 0:
                    minval = -100 * e_min
                eigs = eigs[eigs > minval]
            elif zeros == "negative":
                eigs = eigs[eigs > 0]
            else:
                try:
                    zeros = float(zeros)
                except ValueError as e:
                    raise ValueError(
                        "`zeros` must be a either a float, 'heuristic' or 'negative'"
                    ) from e
                eigs = eigs[eigs > zeros]

        eigenvalues = cls(eigs)
        N, T = data.shape
        eigenvalues._series_T = T
        eigenvalues._series_N = N
        # get some Marchenko-Pastur endpoints
        shift = 1 - eigs.max() / N
        r = np.sqrt(N / T)
        eigenvalues._marchenko = ((1 - r) ** 2, (1 + r) ** 2)
        eigenvalues._marchenko_shifted = (shift * (1 + r) ** 2, shift * (1 - r) ** 2)
        return eigenvalues  # type: ignore

    @property
    def values(self) -> ndarray:
        """Return the stored eigenvalues."""
        return self._vals

    @property
    def vals(self) -> ndarray:
        """Return the stored eigenvalues. Alternate for Eigenvalues.values"""
        return self._vals

    @property
    def eigenvalues(self) -> ndarray:
        """Return the stored eigenvalues. Alternate for Eigenvalues.values"""
        return self._vals

    @property
    def eigs(self) -> ndarray:
        """Return the stored eigenvalues. Alternate for Eigenvalues.values"""
        return self._vals

    def detrend(self, method: DetrendMethod) -> Eigenvalues:
        if self.deterending is not None:
            raise ValueError("Eigenvalues have already been detrended")
        detrended = _detrend(self.vals, method)
        print(method, np.mean(np.diff(detrended)))
        return Eigenvalues(detrended, detrending=method)

    def trim_report(
        self,
        max_trim: float = 0.5,
        max_iters: int = 7,
        poly_degrees: List[int] = DEFAULT_POLY_DEGREES,
        spline_smooths: List[float] = [],
        spline_degrees: List[int] = [],
        gompertz: bool = True,
        detrend: bool = False,
        outlier_tol: float = 0.1,
        show_progress: bool = False,
    ) -> TrimReport:
        """Compute multiple trim regions iteratively via histogram-based outlier
        detection, perform unfolding for each trim region, and summarize the
        resultant spacings and trimmings.


        Parameters
        ----------
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


        Returns
        -------
        trimmed: TrimReport
            A TrimReport object, which contains various information and functions
            for evaluating the different possible trim regions.
        """
        print("Trimming to central eigenvalues.")

        eigs = self.vals
        return TrimReport(
            eigenvalues=eigs,
            max_trim=max_trim,
            max_iters=max_iters,
            poly_degrees=poly_degrees,
            spline_smooths=spline_smooths,
            spline_degrees=spline_degrees,
            gompertz=gompertz,
            detrend=detrend,
            outlier_tol=outlier_tol,
            show_progress=show_progress,
        )

    def get_best_trimmed(
        self,
        smoother: SmoothMethod = SmoothMethod.Polynomial,
        degree: int = DEFAULT_POLY_DEGREE,
        spline_smooth: float = DEFAULT_SPLINE_SMOOTH,
        max_iters: int = 7,
        max_trim: float = 0.5,
        detrend: bool = False,
        outlier_tol: float = 0.1,
    ) -> Trimmed:
        """For the given smoother and smmothing and trim options, compute
        a up to `max_iters` different trim regions, and select the region
        which has an unfolding that is most GOE-like in terms of its local
        spacings.


        Parameters
        ----------
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

        smoother: "poly" | "spline" | "gompertz" | lambda
            the type of smoothing function used to fit the step function

        degree: int
            the degree of the polynomial or spline

        spline_smooth: float
            the smoothing factors passed into scipy.interpolate.UnivariateSpline

        outlier_tol: float
            A float between 0 and 1, and which is passed as the tolerance parameter for
            [HBOS](https://pyod.readthedocs.io/en/latest/pyod.models.html#module-pyod.models.hbos)
            histogram-based outlier detection


        Returns
        -------
        best_indices: Tuple[int, int]
            The indices (start, end) such that eigenvalues[start:end] is the trimmed
            region that is "most GOE" in terms of its nearest-neighbour level spacings.
        """
        report = None
        if smoother == "poly":
            report = TrimReport(
                eigenvalues=self.vals,
                max_trim=max_trim,
                max_iters=max_iters,
                poly_degrees=[degree],
                spline_smooths=[],
                spline_degrees=[],
                gompertz=False,
                detrend=detrend,
                outlier_tol=outlier_tol,
            )
        elif smoother == "spline":
            report = TrimReport(
                eigenvalues=self.vals,
                max_trim=max_trim,
                max_iters=max_iters,
                poly_degrees=[],
                spline_smooths=[spline_smooth],
                spline_degrees=[degree],
                gompertz=False,
                detrend=detrend,
                outlier_tol=outlier_tol,
            )
        elif smoother == "gompertz":
            report = TrimReport(
                eigenvalues=self.vals,
                max_trim=max_trim,
                max_iters=max_iters,
                poly_degrees=[],
                spline_smooths=[],
                spline_degrees=[],
                gompertz=True,
                detrend=detrend,
                outlier_tol=outlier_tol,
            )
        else:
            raise ValueError("Unknown smoother.")

        _, _, best_indices, _ = report.best_overall()
        start, end = best_indices[0][0], best_indices[0][1]
        return self.trim_manually(start, end)

    def trim_marchenko_pastur(
        self,
        series_length: Optional[int] = None,
        n_series: Optional[int] = None,
        largest: bool = True,
        use_shifted: bool = True,
    ) -> Tuple[Trimmed, Tuple[float, float]]:
        """Trim to noise eigenvalues under assumption that eigenvalues come from
        correlation matrix.

        Parameters
        ----------
        series_length: int
            The length of the time series (e.g. number of time points per
            series). If None, check self for saved values and use those.

        n_series: int
            The number of time series of length `series_length`. If None,
            check self for saved values and use those.

        largest: bool
            If False, return the central (e.g. noise) eigenvalues. If True,
            return only the largest eigenvalues as determined by the cutpoints.

        use_shifted: bool
            If True, use the shifted distribution (see references below) which
            accounts for common nonstationary trends. Else, use classic
            Marcenko-Pastur cutpoints.

        Returns
        -------
        trimmed: Trimmed
            A Trimmed object containing the eigenvalues trimmed according to

        trim_vals: Tuple[float, float]
            The (trim_min, trim_max) cutpoints.


        References
        ----------
        Almog, A., Buijink, M. R., Roethler, O., Michel, S., Meijer, J. H.,
        Rohling, J. H. T., & Garlaschelli, D. (2019). Uncovering functional
        signature in neural systems via random matrix theory. PLOS Computational
        Biology, 15(5), e1006934. doi:10.1371/journal.pcbi.1006934
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6513117/#sec008title,

        'A second and related improvement takes into account the effects of
        common (nonstationary) trends for a system with N cells, and in
        particular the largest eigenvalue eig_max. We realize that the effects
        of noise are inseparably coupled to those of the global trend [51], as
        the presence of the latter modifies and left-shifts the density of
        eigenvalues that we would otherwise observe in presence of noise only.
        So we do not simply superimpose the two effects as in [15]; on the
        contrary, we calculate the modification of the random bulk exactly,
        given the system’s empirical eig_max. In particular, we calculate the
        shifted value of an original Wishart matrix [15] to find

        eig_+- = (1 − eig_max/N)(1 +- 1/(sqrt(Q)))**2

        where Q = T/N is the ratio between the number of time steps in the data
        T and the number of cells N. Fig 2 shows both the modified and
        unmodified spectral densities. It also shows that taking the left-shift
        of the random bulk into account is very important, as it unveils
        informative empirical eigenvalues that would otherwise be classified as
        consistent with the random spectrum and hence discarded.'

        See also
        https://en.wikipedia.org/w/index.php?title=Marchenko%E2%80%93Pastur_distribution&oldid=939377392
        for a simple example of how the unshifted form of this trimming is done.
        """

        N = self._series_N if n_series is None else n_series
        T = self._series_T if series_length is None else series_length
        if N is None or T is None:
            raise ValueError(
                "Cannot determine either time series length or number of time series "
                "corresponding to eigenvalues. Consider extracting your eigenvalues "
                "from your data matrix via `Eigenvalues.from_time_series` or be sure "
                "to pass correct values to `n_series` and `series_length`."
            )
        eig_max = self.vals.max()
        if use_shifted:
            shift = 1 - (eig_max / N)
            trim_min = shift * (1 - np.sqrt(N / T)) ** 2
            trim_max = shift * (1 + np.sqrt(N / T)) ** 2
        else:
            trim_min, trim_max = (1 - np.sqrt(N / T)) ** 2, (1 + np.sqrt(N / T)) ** 2

        trims = (trim_min, trim_max)

        if largest:
            return Trimmed(self.vals[self.vals > trim_max]), trims
        else:
            return (
                Trimmed(self.vals[(self.vals > trim_min) & (self.vals < trim_max)]),
                trims,
            )

    def trim_manually(self, start: int, end: int) -> Trimmed:
        """trim sorted eigenvalues to [start:end), e.g. [eigs[start], ..., eigs[end-1]]"""
        trimmed_eigs = self.vals[start:end]
        return Trimmed(trimmed_eigs)

    def trim_interactively(self) -> None:
        raise NotImplementedError("This feature will be in a later release.")

    def trim_unfold_auto(
        self,
        max_trim: float = 0.5,
        max_iters: int = 7,
        poly_degrees: List[int] = DEFAULT_POLY_DEGREES,
        spline_smooths: List[float] = [],
        spline_degrees: List[int] = [],
        gompertz: bool = True,
        prioritize_smoother: bool = True,
        outlier_tol: float = 0.1,
    ) -> Unfolded:
        """Exhaustively compare mutliple trim regions and smoothers based on their "GOE score"
        and unfold the eigenvalues, using the trim region and smoothing parameters
        determined to be "most GOE" based on the exhaustive process.

        Exhaustively trim and unfold for various smoothers, and select the "best" overall trim
        percent and smoother according to GOE score.


        Parameters
        ----------
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

        prioritize_smoother: bool
            Whether or not to select the optimal smoother before selecting the optimal
            trim region. See notes. Default: True.

        outlier_tol: float
            A float between 0 and 1, and which is passed as the tolerance parameter for
            [HBOS](https://pyod.readthedocs.io/en/latest/pyod.models.html#module-pyod.models.hbos)
            histogram-based outlier detection


        Notes
        -----
        Summary of the automatic trim-unfold process:

        1. Compute multiple "natural" trim regions via histogram-based outlier detection,
           halting when trimming would reach `max_trim` and/or `max_iters`. Visually,
           histogram-based outlier detection on the sorted eigenvalues will tend to find
           regions where there is a sudden change in the spacings between adjacent
           eigenvalues.

        2. For each trim region, fit all possible smoothers (i.e., smoother family +
           smoother parameters) specified in the arguments, and generate a set of unfolded
           eigenvalues for each.

        3. For each set of unfolded eigenvalues, compute the *GOE score*. The GOE score
           indexes how much the mean and variance of the spacings of the unfolded values
           differ from the expected spacing variance and mean for the unfolding of a GOE
           matrix.

        4. Assume that the choice of smoother should determine the optimal trim region,
           and not the converse. That is, since the combination of smoothers and trim
           regions yields a grid of scores:
              - for each trim region, there is a GOE score per smoother
              - for each smoother, there is a GOE score per trim region

           then we might:
                a. first choose the best trim, on average, across smoothers, and then, for
                   that trim, choose the smoother that results in the best GOE score, OR
                b. first choose the best smoother, on average, across trims, and then, for
                   that smoother, choose the trim that results in the best GOE score

            Choosing (a) might make sense, but in practice, the more eigenvalues that
            are trimmed, the more clustered or "smooth" the remaining values. That is,
            the more you trim, the more you can alter the nearest-neighbors' spacing distribution
            simply by varying the flexibility of your smoother. Since the GOE score is
            calculated based on the NNSD, this means can achieve more variable spacings by
            increasing the smoother flexibility, and vice-versa. Presumably, with increased
            flexibility, we also increase the number level variance, and decrease the spectral
            rigidity. In short, it is not clear exactly *what* we are doing if we make a
            particular trimming look most locally-GOE. It is also poorly motivated, since
            the reason we trim in the first place is to remove anchor points that have
            strong effects on the smoothing procedure.

            However, choosing (b) amounts to something like "choose the best approximation
            of the functional form of the eigenvalues, regardless of scale / outliers, and
            then account for bad fits due to outliers". Here, the danger is that outliers
            prior to trimming will badly effect flexible smoothers, making a naive summary
            statistic, like the average score across trims, bad for determining what
            smoother is overall. So instead, we use a trimmed mean.

        5. Assume that the best smoother is the one which results in the most GOE-like
           spacing distribution across all trims and all smoothers
        """
        trimmed = TrimReport(
            self.values,
            max_trim=max_trim,
            max_iters=max_iters,
            poly_degrees=poly_degrees,
            spline_smooths=spline_smooths,
            spline_degrees=spline_degrees,
            gompertz=gompertz,
            outlier_tol=outlier_tol,
        )
        orig_trimmed, unfolded = trimmed._get_autounfold_vals()
        percent = np.round(100 * len(orig_trimmed) / len(self.values), 1)
        print(f"Trimmed to {percent}% of original eigenvalues.")
        return Unfolded(orig_trimmed, unfolded)

    def unfold(
        self,
        smoother: Optional[SmoothMethod] = None,
        degree: int = DEFAULT_POLY_DEGREE,
        spline_smooth: float = DEFAULT_SPLINE_SMOOTH,
        detrend: bool = False,
    ) -> Unfolded:
        """Unfold the eigenvalues with the specified smoothers.

        Parameters
        ----------
        eigs: ndarray
            sorted eigenvalues

        smoother: "poly" | "spline" | "gompertz" | "goe" | lambda | None
            The type of smoothing function used to fit the step function.
            - "poly": perform polynomial unfolding.
            - "spline": use fit a univarate spline.
            - "gompertz": fit a Gompertz exponential curve.
            - "goe": perform a "smooth" unfolding via the semicircle law
            - lambda: not implemented.
            - None (default):
              - if constructed from Eigenvalues.generate(..., kind="goe"),
              then uses the analytic unfolding, smoother="goe"
              - if constructed from Eigenvalues.generate(..., kind="poisson"),
              then uses a polynomial of degree 19

        degree: int
            The degree of the polynomial or spline.

        spline_smooth: float
            The smoothing factors passed into scipy.interpolate.UnivariateSpline

        emd_detrend: bool
            Whether to apply a final Empirical Mode Decomposition detrending
            (Morales et al.) before returning the final unfolded values.


        Returns
        -------
        unfolded: ndarray
            the unfolded eigenvalues

        steps: ndarray
            the step-function values
        """
        if smoother is None:
            if self.kind is MatrixKind.GOE:
                print(
                    "Using default Wigner surmise for (analytic) unfolding of GOE "
                    "eigenvalues. To silence this message, specify `smoother` "
                    "argument to Eigenvalues.unfold()."
                )
                return self.unfold_goe()
            elif self.kind in [MatrixKind.GDE, MatrixKind.Poisson]:
                print(
                    "Using default (analytic, exponential) unfolding for GDE / "
                    "Poisson eigenvalues. To silence this message, specify `smoother` "
                    "argument to Eigenvalues.unfold()."
                )
                return self.unfold_poisson()
            else:
                print(
                    "No smoother specified for unfolding. Using polynomial of degree "
                    f"{DEFAULT_POLY_DEGREE}. To silence this message, specify `smoother` "
                    "argument to Eigenvalues.unfold()."
                )
                smoother = SmoothMethod.Polynomial
        smoother = SmoothMethod.validate(smoother)
        if smoother is SmoothMethod.GOE:
            return self.unfold_goe()  # type: ignore[unreachable]
        if smoother is SmoothMethod.Poisson:
            return self.unfold_poisson()

        eigs = self.eigs
        unfolded, _, closure = Smoother(eigs).fit(
            smoother=smoother,
            degree=degree,
            spline_smooth=spline_smooth,
            detrend=detrend,
            return_callable=True,
        )
        if detrend:
            unfolded = emd_detrend(unfolded)
        return Unfolded(
            originals=eigs,
            unfolded=np.sort(unfolded),
            smoother=closure,  # type: ignore
        )

    def unfold_goe(self) -> Unfolded:
        """Unfold via Wigner's semicircle law."""

        eigs = self.eigenvalues
        N = len(eigs)
        end = np.sqrt(2 * N)

        def __R1(x: float) -> np.float64:
            """The level density R_1(x), as per p.152, Eq. 7.2.33 of Mehta (2004)"""
            if np.abs(x) < end:
                return np.float64((1 / np.pi) * np.sqrt(2 * N - x * x))
            return np.float64(0.0)

        MAX = np.float64(quad(__R1, -end, end)[0])

        def smooth_goe(x: float) -> np.float64:
            if x > end:
                return MAX
            return np.float64(quad(__R1, -end, x)[0])

        unfolded = np.sort(np.vectorize(smooth_goe)(eigs))
        return Unfolded(originals=eigs, unfolded=unfolded)

    def unfold_poisson(self) -> Unfolded:
        """
        See https://iopscience.iop.org/article/10.1088/1742-6596/492/1/012011/pdf
        pg.2, middle paragraph, which notes:

            In the traditional approach to the unfolding procedure (see e.g.
            [12]), a mapping of the sequence of actual energy levels
            {E(i): i = 1...N}, into dimensionless levels e(i), where

                E(i) -> e(i) =def= N[E(i)] = int_{-inf}^{E(i)} p(E')dE'

            (1) realizes this process. Here, N[E] is a smooth
            approximation to the accumulated or integrated density function
            (IDOS)

                N[E(i)] = sum_{i=1}^N [H * (E − E(i))],

            where H is the Heaviside step function. N[E(i)] counts the exact
            number of levels i up to excitation energy E(i), and increases by
            one unit as the energy E passes a (non-degenerate) eigenvalue E(i)
            [6, 7, 13]. The unfolding procedure is straightforward if a
            theoretical prediction for the average level density p(E) is
            available, e.g. the Weyl formula in the case of quantum billiards
            [9], the semi-circular distribution in the case of GOE, and the
            normal distribution in the Poisson case [3]. However, an analytical
            form for p(E) is usually only valid in the asymptotic limit for
            spectra with a very large number of levels N -> inf [8] and often is
            not known [7]. In those cases, a local unfolding can be applied,
            which can only be used to study short-range correlations [14].

        Another way to look at it is that the limiting
        """
        eigs = self.eigenvalues
        N = len(eigs)

        def __R1(x: float) -> np.float64:
            return np.float64(np.exp(-np.abs(x)))  # correct!

        def smooth_poisson(x: float) -> np.float64:
            return N * np.float64(quad(__R1, -np.inf, np.abs(x))[0])

        # we are in the case https://math.stackexchange.com/a/115132
        # but where mu and lambda are both 1
        unfolded = np.sort(np.vectorize(smooth_poisson)(eigs))  # correct!
        return Unfolded(originals=eigs, unfolded=unfolded)


def _eigs_via_transpose(
    M: fArr, covariance: bool = True, use_sparse: bool = False, **sp_args: Any
) -> fArr:
    """Use transposes to rapidly compute eigenvalues of covariance and
    correlation matrices.

    Parameters
    ----------
    M: ndarray
        A time-series array with shape (N, T) such that N > T. (If N <= T, no
        benefits are gained from transposed intermediates, so the eigenvalues
        are simply computed in the normal fashion).

    Returns
    -------
    eigs: ndarray
        The computed eigenvalues.

    Notes
    -----
    # Covariance Matrices

    if:

        (n, p) = X.shape, n > p, AND
        norm(X) = X - np.mean(X, axis=1, keepdims=True),
        Z = norm(X)

    then:

        np.cov(X)  == np.matmul(norm(X), norm(X).T) / (p - 1)
                   == r * Z * Z.T,   if   r = 1/(1-p)

    Now, it then follows that:

        eigs(np.cov(X))  ==  eigs(r Z * Z.T)
                         == r * eigs(Z * Z.T)

    But the nonzero eigenvalues of Z*Z.T are exactly those of Z.T*Z. Thus:

        eigs_nz(np.cov(X))  == r * eigs(Z.T * Z)

    which can be computed extremely quickly if p << n

    # Correlation Matrices

    Keeping X, n, p, r the same as above, and letting C = corr(X), and

        stand(X) = (
                   (X - np.mean(X, axis=1, keepdims=True)) /
                   np.std(X, axis=1, ddof=1, keepdims=True)
                )
                 = norm(X) / np.std(X, axis=1, ddof=1, keepdims=True)

    we have:

        corr(X) == np.cov(stand(X))

    but then since norm(stand(X)) = stand(X), if we let Y = stand(X), Z = norm(Y)

        eigs(corr(X)) == eigs(np.cov(stand(X)))
                      == eigs(np.cov(Y))

    Now, as above, it then follows that:

        eigs_nz(corr(X)) == r * eigs(Z.T*Z)
                         == r * eigs(Y.T*Y)

    """
    N, T = M.shape
    if N <= T:
        raise ValueError("Array is not of correct shape to benefit from transposed intermediates.")
    r = 1 / (T - 1)
    Z = M - np.mean(M, axis=1, keepdims=True)
    if not covariance:
        Z /= np.std(M, axis=1, ddof=1, keepdims=True)
    M = np.matmul(Z.T, r * Z)

    if use_sparse:
        M_sparse = scipy.sparse.tril(M)
        if sp_args.get("return_eigenvectors") is True:
            raise ValueError(
                "This function is intended only as a helper "
                "to extract eigenvalues from time-series."
            )

        eigs: fArr = np.array(scipy.sparse.linalg.eigsh(M_sparse, **sp_args))
    else:
        eigs = np.linalg.eigvalsh(M)

    return eigs
