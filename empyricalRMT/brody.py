import numpy as np

from numpy import ndarray
from scipy.optimize import minimize_scalar
from scipy.special import gamma


def brody_dist(s: ndarray, beta: float) -> ndarray:
    """See Eq. 8 of
    Dettmann, C. P., Georgiou, O., & Knight, G. (2017).
    Spectral statistics of random geometric graphs.
    EPL (Europhysics Letters), 118(1), 18003.
    """
    b1 = beta + 1
    alpha = gamma((beta + 2) / b1) ** b1
    return b1 * alpha * s ** beta * np.exp(-alpha * s ** b1)


def log_brody(s: ndarray, beta: float) -> ndarray:
    """Just a helper re-written to prevent overflows"""
    b1 = beta + 1.0
    alpha = gamma((beta + 2.0) / b1) ** b1
    return np.log(b1 * alpha) + beta * np.log(s) - alpha * s ** b1


def fit_brody(s: ndarray) -> float:
    """Return the maximum likelihood estimate for beta.

    Paramaters
    ----------
    s: ndarray
        The array of spacings.

    Returns
    -------
    beta: float
        The MLE estimate for beta.
    """
    # use negative log-likelihood because we want to minimize
    log_like = lambda beta: -np.sum(log_brody(s, beta))
    opt_result = minimize_scalar(log_like, bounds=(1e-10, 1.0), method="Bounded")
    if not opt_result.success:
        raise RuntimeError("Optimizer failed to find optimal Brody fit.")
    return float(opt_result.x)
