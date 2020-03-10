import numpy as np
import time

from numpy import ndarray

from scipy.integrate import quad
from scipy.linalg import eigvalsh_tridiagonal
from scipy.sparse import diags
from typing import Tuple, Union
from typing_extensions import Literal
from warnings import warn

from empyricalRMT.correlater import correlate_fast
from empyricalRMT.unfold import Unfolded


MatrixKind = Union[
    Literal["goe"], Literal["gue"], Literal["uniform"], Literal["poisson"]
]


def generate_eigs(
    matsize: int,
    kind: MatrixKind = "goe",
    seed: int = None,
    log: bool = False,
    use_tridiagonal: bool = True,
) -> ndarray:
    """Generate a random matrix as specified by arguments, and compute and return
    the eigenvalues.

    Parameters
    ----------
    matsize: int
        The size (e.g. width or height) of the square matrix that will be
        generated.

    kind: "goe" | "gue" | "poisson" | "uniform"
        From which ensemble to sample the matrix:
        - "goe" is a matrix from the Gaussian Orthogonal Ensemble
        - "gue" is a matrix from the Gaussian Unitary Ensemble
        - "poisson" is a "Gaussian Diagonal Ensemble", a diagonal matrix with
          all entries being samples from a standard normal distribution.
        - "uniform" is currently unimplemented

    seed: int
        Seed value for `np.random.seed`. Ensures reproducible results.

    log: bool
        Whether or not to log start and end times for computing eigenvalues.

    use_tridiagonal: bool
        For `kind` "gue" only. Generate a tridiagonal matrix with identical
        eigenvalue distributions to a GOE matrix instead of a full GOE matrix.
        *Dramatically* speeds up computation of eigenvalues, and is strongly
        recommended for generating matrices of approximately size N >= 2000.
    """
    if kind == "poisson":
        np.random.seed(seed)
        # eigenvalues of diagonal are just the entries
        return np.sort(np.random.standard_normal(matsize))
    elif kind == "uniform":
        raise Exception("UNIMPLEMENTED!")
    elif kind == "gue":
        size = [matsize, matsize]
        if seed is not None:
            np.random.seed(seed)
        A = np.random.standard_normal(size) + 1j * np.random.standard_normal(size)
        M = (A + A.conjugate().T) / 2
    elif kind == "goe":
        if matsize > 500 and use_tridiagonal:
            M = _generate_GOE_tridiagonal(size=matsize, seed=seed)
        else:
            M = _generate_GOE_matrix(matsize, seed=seed)
    else:
        kinds = ["goe", "gue", "poisson", "uniform"]
        raise ValueError(f"`kind` must be one of {kinds}")

    if log:
        print(f"\n{time.strftime('%H:%M:%S (%b%d)')} -- computing eigenvalues...")
    eigs = np.linalg.eigvalsh(M)
    if log:
        print(f"{time.strftime('%H:%M:%S (%b%d)')} -- computed eigenvalues.")
    return eigs


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
        """The level density R_1(x), as per p.152, Eq. 7.2.33 of Mehta (2004) """
        if np.abs(x) < end:
            return np.float64((1 / np.pi) * np.sqrt(2 * N - x * x))
        return 0.0

    MAX = quad(__R1, -end, end)[0]

    def smooth_goe(x: float) -> np.float64:
        if x > end:
            return MAX
        return quad(__R1, -end, x)[0]

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
    return_mats: bool = False,
) -> Union[ndarray, Tuple[ndarray, ndarray, ndarray]]:
    """[WIP]. Generate a correlated system for examinatino with, e.g.
    Marcenko-Pastur. """
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
    eigs = np.linalg.eigvalsh(M)
    if log:
        print(f"{time.strftime('%H:%M:%S (%b%d)')} -- computed eigenvalues.")
    n, t = shape
    eig_min, eig_max = (1 - np.sqrt(n / t)) ** 2, (1 + np.sqrt(n / t)) ** 2
    print(f"Eigenvalues in ({eig_min},{eig_max}) are likely noise-related.")

    if return_mats:
        return eigs, A, M
    return eigs


# TODO, WIP This is really only valid for testing e.g. Tracy-Widom and the
# semi-circle law.
def tracy_widom_eigs(
    n_eigs: int = 1000,
    sub_matsize: int = 100,
    kind: MatrixKind = "goe",
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
    max_eigs = np.empty([n_eigs])
    for i in range(n_eigs):
        sub_eigs = generate_eigs(matsize=sub_matsize, use_tridiagonal=use_tridiagonal)
        max_eigs[i] = sub_eigs.max()
    if return_normalized:
        max_eigs *= float(n_eigs) ** (1.0 / 6.0)
        max_eigs -= 2.0 * np.sqrt(n_eigs)
    return max_eigs


# TODO, WIP
def time_series_eigs(
    n: int = 1000, t: int = 200, dist: str = "normal", log: bool = True
) -> ndarray:
    """Generate a correlation matrix for testing Marcenko-Pastur, other spectral observables."""
    if dist == "normal":
        M_time = np.random.standard_normal([n, t])

    if log:
        print(f"\n{time.strftime('%H:%M:%S (%b%d)')} -- computing correlations...")
    M = correlate_fast(M_time)
    if log:
        print(f"\n{time.strftime('%H:%M:%S (%b%d)')} -- computing eigenvalues...")
    eigs = np.linalg.eigvalsh(M)
    if log:
        print(f"\n{time.strftime('%H:%M:%S (%b%d)')} -- computed eigenvalues...")
    return eigs


def generate_uniform(
    matsize: int = 1000, lower: float = 0, upper: float = 1
) -> ndarray:
    raise NotImplementedError


def _generate_GOE_matrix(size: int = 100, seed: int = None) -> ndarray:
    if seed is not None:
        np.random.seed(seed)
    M = np.random.standard_normal([size, size])
    return (M + M.T) / np.sqrt(2)


def _generate_GOE_tridiagonal(size: int = 100, seed: int = None) -> ndarray:
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
    M = diags(diagonals, [0, -1, 1])
    return M.toarray()


def _generate_GOE_tridiagonal_direct(
    size: int = 100, seed: int = None, dowarn: bool = True
) -> ndarray:
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
    eigs = eigvalsh_tridiagonal(
        diagonal,
        chi,
        # select="a",
        check_finite=False,
        select="i",
        select_range=(1, size - 2),
        lapack_driver="stebz",
        tol=4 * np.finfo(np.float64).eps,
    )
    return eigs


def _generate_poisson(size: int = 100, seed: int = None) -> ndarray:
    if seed is not None:
        np.random.seed(seed)
    return np.diag(np.random.standard_normal(size))
