import numpy as np
from PyEMD import EMD

from empyricalRMT._types import fArr


# detrended unfolding via the Empirical Mode Decomposition and first
# intrinsic mode function
# TODO: re-integrate this option into updated API
def emd_detrend(unfolded: fArr) -> fArr:
    """'Detrend' the unfolded eigenvalues via Empirical Mode Decomposition.

    Parameters
    ----------
    unfolded: ndarray
        The unfolded eigenvalues.

    Returns
    -------
    detrended: ndarray
        The 'detrended' values (i.e. with last IMF residue removed).

    References
    ----------
    Morales, I. O., Landa, E., Stránský, P., & Frank, A. (2011). Improved
    unfolding by detrending of statistical fluctuations in quantum spectra.
    Physical Review E, 84(1). doi:10.1103/physreve.84.016203
    """
    spacings = np.diff(unfolded)
    s_av = np.average(spacings)
    s_i = spacings - s_av

    ns = np.zeros([len(unfolded)], dtype=int)
    delta_n = np.zeros([len(unfolded)])
    for n in range(len(unfolded)):
        delta_n[n] = np.sum(s_i[0:n])
        ns[n] = n

    # last member of IMF basis is the trend
    trend = EMD().emd(delta_n)[-1]
    detrended_delta = delta_n - trend

    # see Morales (2011) DOI: 10.1103/PhysRevE.84.016203, Eq. 15
    unfolded_detrend = np.empty([len(unfolded)])
    for i in range(len(unfolded_detrend)):
        if i == 0:
            unfolded_detrend[i] = unfolded[i]
            continue
        unfolded_detrend[i] = detrended_delta[i - 1] + unfolded[0] + (i - 1) * s_av

    return unfolded_detrend
