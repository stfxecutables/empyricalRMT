import numpy as np

from PyEMD import EMD

from empyricalRMT.rmt.observables.spacings import computeSpacings

# detrended unfolding via the Empirical Mode Decomposition and first
# intrinsic mode function
def emd_detrend(unfolded: np.array) -> np.array:
    spacings = computeSpacings(unfolded, trim=False)
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
