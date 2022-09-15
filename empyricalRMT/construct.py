import time
from typing import Optional, Tuple
from warnings import warn

import numpy as np
from numpy import ndarray
from scipy.integrate import quad
from scipy.linalg import eigvalsh_tridiagonal
from scipy.sparse import diags

from empyricalRMT._types import MatrixKind, fArr
from empyricalRMT.correlater import correlate_fast
from empyricalRMT.unfold import Unfolded


def goe_unfolded(matsize: int, log: bool = False) -> Unfolded:
    """Generate GOE eigenvalues and perform a smooth / analytic unfolding
    using the expected limiting distribution, e.g. Wigner's semicricle law.

    Parameters
    ----------
    matsize: int
        The size (e.g. width) of the square GOE matrix to generate.
    log: bool
        Whether or not to log when starting and completing solving the
        eigenvalue problem. Useful for large matrices on slow machines.

    Returns
    -------
    unfolded: Unfolded
        The Unfolded object containing the resultant original and unfolded
        eigenvalues.
    """

    N = matsize
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

    M = _generate_GOE_tridiagonal(matsize)
    print("Computing GOE eigenvalues...")
    eigs = np.linalg.eigvalsh(M)
    print("Done!")
    print(f"Unfolding GOE eigenvalues (N == {N})...")
    unfolded_eigs = np.sort(np.vectorize(smooth_goe)(eigs))
    print("Done!")
    return Unfolded(originals=eigs, unfolded=unfolded_eigs)


def correlated_eigs(
    percent: float = 25,
    shape: Tuple[int, int] = (1000, 500),
    noise: float = 0.1,
    log: bool = True,
) -> tuple[fArr, fArr, fArr]:
    """[WIP]. Generate a correlated system for examinatino with, e.g.
    Marcenko-Pastur."""
    A = np.random.standard_normal(shape)
    correlated = np.random.permutation(A.shape[0] - 1) + 1  # don't select first row
    last = int(np.floor((percent / 100) * A.shape[0]))
    corr_indices = correlated[:last]
    # introduce correlation in A
    ch, unif, norm = np.random.choice, np.random.uniform, np.random.normal
    for i in corr_indices:
        A[i, :] = ch([-1, 1]) * unif(1, 2) * (A[0, :] + norm(0, noise, size=A.shape[1]))
    M = correlate_fast(A)
    if log:
        print(f"\n{time.strftime('%H:%M:%S (%b%d)')} -- computing eigenvalues...")
    eigs: fArr = np.linalg.eigvalsh(M)
    if log:
        print(f"{time.strftime('%H:%M:%S (%b%d)')} -- computed eigenvalues.")
    n, t = shape
    eig_min, eig_max = (1 - np.sqrt(n / t)) ** 2, (1 + np.sqrt(n / t)) ** 2
    print(f"Eigenvalues in ({eig_min},{eig_max}) are likely noise-related.")
    return eigs, A, M


# TODO, WIP This is really only valid for testing e.g. Tracy-Widom and the
# semi-circle law.
def tracy_widom_eigs(
    n_eigs: int = 1000,
    sub_matsize: int = 100,
    kind: MatrixKind = MatrixKind.GOE,
    use_tridiagonal: bool = False,
    return_normalized: bool = False,
) -> ndarray:
    """Quickly generate extreme eigenvalues.

    Parameters
    ----------
    n_eigs: int
        the desired size of the square matrix, i.e. number of eigenvalues

    sub_matsize: int
        the size of the smaller matrices to use to speed eigenvalue calculation

    kind: "goe" | "gue" | "poisson"

    use_tridiagonal: bool
        Whether or not to force GOE eigenvalues to be always generated from
        tridiagonal matrices, regardless of submatrix size.

    return_normalized: bool
        If True, normalize eigenvalues based on n_eigs

    Returns
    -------
    max_eigs: the largest eigenvalues generated

    References
    ----------
    Edelman, A., Sutton, B. D., & Wang, Y. (2014). Random matrix theory,
    numerical computation and applications. Modern Aspects of Random Matrix
    Theory, 72, 53, [pg 3., Algorithm 1]
    """

    """
    max_eigs = np.empty([n_eigs])
    for i in range(n_eigs):
        sub_eigs = Eigenvalues.generate(matsize=sub_matsize, use_tridiagonal=use_tridiagonal)
        max_eigs[i] = sub_eigs.max()
    if return_normalized:
        max_eigs *= float(n_eigs) ** (1.0 / 6.0)
        max_eigs -= 2.0 * np.sqrt(n_eigs)
    return max_eigs
    """
    raise NotImplementedError()


# TODO, WIP
def time_series_eigs(n: int = 1000, t: int = 200, dist: str = "normal", log: bool = True) -> fArr:
    """Generate a correlation matrix for testing Marcenko-Pastur, other spectral observables."""
    if dist == "normal":
        M_time = np.random.standard_normal([n, t])

    if log:
        print(f"\n{time.strftime('%H:%M:%S (%b%d)')} -- computing correlations...")
    M = correlate_fast(M_time)  # type:ignore
    if log:
        print(f"\n{time.strftime('%H:%M:%S (%b%d)')} -- computing eigenvalues...")
    eigs: fArr = np.linalg.eigvalsh(M)
    if log:
        print(f"\n{time.strftime('%H:%M:%S (%b%d)')} -- computed eigenvalues...")
    return eigs


def generate_uniform(matsize: int = 1000, lower: float = 0, upper: float = 1) -> ndarray:
    raise NotImplementedError


def _generate_GOE_matrix(size: int = 100, seed: Optional[int] = None) -> fArr:
    if seed is not None:
        np.random.seed(seed)
    M = np.random.standard_normal([size, size])
    return (M + M.T) / np.sqrt(2)  # type: ignore


def _generate_GOE_tridiagonal(size: int = 100, seed: Optional[int] = None) -> fArr:
    """See: Edelman, A., Sutton, B. D., & Wang, Y. (2014).
    Random matrix theory, numerical computation and applications.
    Modern Aspects of Random Matrix Theory, 72, 53.
    """
    if seed is not None:
        np.random.seed(seed)
    chi_range = size - 1 - np.arange(size - 1)
    chi = np.sqrt(np.random.chisquare(chi_range))
    diagonals = [
        np.random.normal(0, np.sqrt(2), size) / np.sqrt(2),
        chi / np.sqrt(2),
        chi / np.sqrt(2),
    ]
    M = diags(diagonals, [0, -1, 1])  # type: ignore
    return M.toarray()  # type: ignore


def _generate_GOE_tridiagonal_direct(
    size: int = 100, seed: Optional[int] = None, dowarn: bool = True
) -> fArr:
    """See: Edelman, A., Sutton, B. D., & Wang, Y. (2014).
    Random matrix theory, numerical computation and applications.
    Modern Aspects of Random Matrix Theory, 72, 53.
    """
    if dowarn:
        warn(
            "While this method is fast, and uses the least memory, it appears that"
            "`eigvalsh_tridiagonal` is considerably less precise, and will result"
            "in significant deviations from the expected values for the long range"
            "spectral observables (e.g. spectral rigidity, level number variance)."
        )
    if seed is not None:
        np.random.seed(seed)
    size = size + 2
    chi_range = size - 1 - np.arange(size - 1)
    chi = np.sqrt(np.random.chisquare(chi_range))
    diagonal = np.random.normal(0, np.sqrt(2), size) / np.sqrt(2)
    eigs: fArr = np.array(
        eigvalsh_tridiagonal(
            diagonal,
            chi,
            # select="a",
            check_finite=False,
            select="i",
            select_range=(1, size - 2),
            lapack_driver="stebz",
            tol=float(4.0 * np.finfo(np.float64).eps),
        )
    )
    return eigs


def _generate_poisson(size: int = 100, seed: Optional[int] = None) -> fArr:
    if seed is not None:
        np.random.seed(seed)
    return np.diag(np.random.standard_normal(size))  # type: ignore
