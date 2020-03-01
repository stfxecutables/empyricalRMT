import numpy as np
import time

from numpy import ndarray

from scipy.integrate import quad
from scipy.linalg import eigvalsh_tridiagonal
from scipy.sparse import diags
from typing import Union
from typing_extensions import Literal
from warnings import warn

from empyricalRMT.rmt.unfold import Unfolded


MatrixKind = Union[
    Literal["goe"], Literal["gue"], Literal["uniform"], Literal["poisson"]
]


def generate_eigs(
    matsize: int,
    mean: float = 0,
    sd: float = 1,
    kind: MatrixKind = "goe",
    seed: int = None,
    log: bool = False,
) -> ndarray:
    """Generate a random matrix as specified by arguments, and compute and return
    the eigenvalues.

    Parameters
    ----------
    matsize: int
        The width (or height) of a the square matrix that will be generated.
    mean: float
        If `kind` is "goe", the mean of the normal distribution used to generate
        the normally-distributed values.
    sd: float
        If `kind` is "goe", the s.d. of the normal distribution used to generate
        the normally-distributed values.
    kind: "goe" | "gue" | "poisson" | "uniform"
        The kind of matrix to generate.
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
        if matsize > 500:
            M = _generate_GOE_tridiagonal(size=matsize, seed=seed)
        else:
            M = _generate_GOE_matrix(matsize, mean, sd, seed=seed)
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
    """Perform a smooth / analytic unfolding using the expected limiting distribution,
    e.g. Wigner's semicricle law.

    Parameters
    ----------
    matsize: int
        The size (e.g. width) of the square GOE matrix.
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


def fast_poisson_eigs(matsize: int = 1000, sub_matsize: int = 100) -> ndarray:
    """Use independence and fact that eigvalsh is bottleneck to more quickly generate
    eigenvalues. E.g. if matsize == 1024, sub_matsize == 100, generate 10 (100x100)
    matrices and one (24x24) matrix, can concatenate the resultant eigenvalues.
    Parameters
    ----------
    matsize: int
        the desired size of the square matrix, or number of eigenvalues
    sub_matsize: int
        the size of the smaller matrices to use to speed eigenvalue calculation
    """
    if matsize < 100:
        return generate_eigs(matsize, kind="goe")
    if sub_matsize >= matsize:
        raise ValueError("Submatrices must be smaller in order to speed calculation.")
    n_submats = int(matsize / sub_matsize)
    last_matsize = matsize % sub_matsize
    eigs_submats = np.empty([n_submats, sub_matsize])
    for i in range(n_submats):
        M = _generate_GOE_matrix(size=sub_matsize)
        sub_eigs = np.linalg.eigvalsh(M)
        eigs_submats[i, :] = sub_eigs
    eigs_remain = np.linalg.eigvalsh(_generate_GOE_matrix(size=last_matsize))
    eigs = list(eigs_submats.flatten()) + list(eigs_remain)
    eigs = np.sort(eigs)
    return eigs


def generate_uniform(
    matsize: int = 1000, lower: float = 0, upper: float = 1
) -> ndarray:
    raise NotImplementedError


def _almost_identity(size: int = 100) -> ndarray:
    E = np.random.standard_normal(size * size).reshape(size, size)
    E = (E + E.T) / np.sqrt(2)
    M = np.ma.identity(size)
    return M + E


def _random_1vector(size: int = 100) -> ndarray:
    vals = np.random.standard_normal([size, 1])
    return vals * vals.T


def _generate_GOE_matrix(
    size: int = 100, mean: float = 0, sd: float = 1, seed: int = None
) -> ndarray:
    if seed is not None:
        np.random.seed(seed)
    if mean != 0 or sd != 1:
        M = np.random.normal(mean, sd, [size, size])
    else:
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
