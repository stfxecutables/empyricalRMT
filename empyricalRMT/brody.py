import numpy as np

from numpy import ndarray
from scipy.optimize import minimize_scalar
from scipy.special import gamma
from scipy.stats.mstats import gmean
from statsmodels.distributions.empirical_distribution import ECDF


def brody_dist(s: ndarray, beta: float) -> ndarray:
    """See Eq. 8 of
    Dettmann, C. P., Georgiou, O., & Knight, G. (2017).
    Spectral statistics of random geometric graphs.
    EPL (Europhysics Letters), 118(1), 18003.
    """
    b1 = beta + 1
    alpha = gamma((beta + 2) / b1) ** b1
    return b1 * alpha * s ** beta * np.exp(-alpha * s ** b1)


def brody_cdf(s: ndarray, beta: float) -> ndarray:
    """Return the cumulative distribution function of the Brody distribution for beta."""
    b1 = beta + 1
    alpha = gamma((beta + 2) / b1) ** b1
    return 1 - np.exp(-alpha * s ** b1)


def log_brody(s: ndarray, beta: float) -> ndarray:
    """Just a helper re-written to prevent overflows and filter negative spacings"""
    b1 = beta + 1.0
    alpha = gamma((beta + 2.0) / b1) ** b1
    s = s[s > 0.0]
    # the lines below are separate for better logging of underflow issues
    t1 = np.log(b1 * alpha)
    t2 = beta * np.log(s)
    t3 = alpha * s ** b1
    return np.sum([t1, t2, t3])


def fit_brody(s: ndarray, method: str = "spacing") -> float:
    """Get an estimate for the beta parameter of the Brody distribution

    Paramaters
    ----------
    s: ndarray
        The array of spacings.

    Returns
    -------
    beta: float
        The MLE estimate for beta.
    """
    method = method.lower()
    if method == "spacing" or method == "spacings":
        return fit_brody_max_spacing(s)
    if method == "mle":
        return fit_brody_mle(s)
    raise ValueError("`method` must be one of 'spacing' or 'mle'.")


def fit_brody_mle(s: ndarray) -> float:
    """Return the maximum likelihood estimate for beta.

    Paramaters
    ----------
    s: ndarray
        The array of spacings.

    Returns
    -------
    beta: float
        The MLE estimate for beta.

    Notes
    -----
    Try using https://en.wikipedia.org/wiki/Maximum_spacing_estimation
    instead
    """
    # use negative log-likelihood because we want to minimize
    # log_like = lambda beta: -np.sum(log_brody(s, beta))
    log_like = lambda beta: -np.sum(brody_dist(s, beta))
    opt_result = minimize_scalar(
        log_like, bounds=(1e-5, 1.0 - 1e-5), method="Bounded", tol=1e-10
    )
    if not opt_result.success:
        raise RuntimeError("Optimizer failed to find optimal Brody fit.")
    return float(opt_result.x)


def fit_brody_max_spacing(s: ndarray) -> float:
    """Return the maximum likelihood estimate for beta.

    Paramaters
    ----------
    s: ndarray
        The array of spacings.

    Returns
    -------
    beta: float
        The maximum spacings estimate for beta.

    Notes
    -----
    Try using https://en.wikipedia.org/wiki/Maximum_spacing_estimation
    instead
    """

    n = len(s) - 1

    def alpha(beta: float) -> np.float64:
        return gamma((beta + 2) / (beta + 1)) ** (beta + 1)

    def _positive_diffs(s: ndarray, beta: float) -> np.float64:
        s = np.sort(s)
        brody_cdf = 1.0 - np.exp(-alpha(beta) * (s ** (beta + 1)))
        diffs = np.diff(brody_cdf)
        diffs = diffs[diffs > 0]  # necessary to prevent over/underflows
        return diffs

    # use negative log-likelihood because we want to minimize
    # log_like = lambda beta: -np.sum(log_brody(s, beta))

    # s = np.sort(s)
    # brody_cdf = lambda beta: 1.0 - np.exp(-alpha(beta) * (s ** (beta + 1)))
    # diffs = lambda beta: np.diff(brody_cdf(beta))

    log_spacings = lambda beta: np.log(_positive_diffs(s, beta))
    S_n = lambda beta: -np.sum(log_spacings(beta)) / (n + 1)
    opt_result = minimize_scalar(
        S_n, bounds=(1e-5, 1.0 - 1e-5), method="Bounded", tol=1e-10
    )
    if not opt_result.success:
        raise RuntimeError("Optimizer failed to find optimal Brody fit.")
    return float(opt_result.x)


def brody_fit_evaluate(s: ndarray, method: str = "spacing") -> dict:
    beta = fit_brody(s, method)
    ecdf = ECDF(s)
    ecdf_x = ecdf.x[1:]  # ECDF always makes first x value -inf if `side`=="left"
    ecdf_y = ecdf.y[1:]
    bcdf = brody_cdf(ecdf_x, beta)
    mad = np.mean(np.abs(ecdf_y - bcdf))
    msqd = np.mean((ecdf_y - bcdf) ** 2)
    return {
        "beta": beta,
        "mad": mad,
        "msqd": msqd,
        "spacings": ecdf_x,
        "ecdf": ecdf_y,
        "brody_cdf": bcdf,
    }
