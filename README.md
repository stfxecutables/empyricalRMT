A python library for Random Matrix Theory eigenvalue unfolding and
computation and plotting of spectral observables.

This libary is still undergoing major re-writes and development, and should
be considered in pre-alpha at this point. Feel free to post issues or ask
questions relating to the code, but keep in mind any bugs and/or issues are
very likely to change in the near future.

# Installation

## From Source
```bash
git clone https://github.com/stfxecutables/empyricalRMT
cd empyricalRMT
pip install -e .
```

## From pip
```bash
pip install empyricalRMT
```

# Basic Usage
```python
import empyricalRMT.observables.unfold as unfold
import empyricalRMT.rmt.plot as plot

from empyricalRMT.observables.levelvariance import sigmaSquared
from empyricalRMT.observables.rigidty import spectralRigidity
from empyricalRMT.rmt.construct import generateGOEMatrix, newEigs

# generate a new matrix from the Gaussian Orthogonal Ensemble and extract
# its eigenvalues
eigs = newEigs(matsize=1000, kind="goe")
# eigs = newEigs(matsize=1000, kind="gue")
# eigs = newEigs(matsize=1000, kind="poisson")

# plot the eigenvalue distribution to confirm Wigner's Semicircle law
plot.rawEigDist(eigs, bins=100, title="Wigner Semicircle", block=True)

# plot the eigenvalue step function
plot.stepFunction(eigs, trim=False, block=True)

# plot the normalized spacings to inspect the Nearest Neighbors Spacing
# Distribution (NNSD) and compare it to those predicted by classical RMT
plot.spacings(eigs=eigs, kind="goe")

# unfold the spectrum with a polynomial fit
unfolded = unfold.polynomial(eigs, degree=11)

# Compute the spectral rigidity of the resultant unfolding
L, delta3 = spectralRigidity(unfolded, eigs, c_iters=200, L_grid_size=100, min_L=0.5, max_L=25)
# Plot the resultant values
df = pd.DataFrame({"L": L, "∆3(L)": delta3})
plot.spectralRigidity(unfolded, df, title="GOE Matrix Spectral Rigidity", mode="block")

# Compute the number level variance
L, sigma_sq = sigmaSquared(eigs, unfolded, c_iters=1000, L_grid_size=100, min_L=0.5, max_L=20)
# Plot the resultant values
df = pd.DataFrame({"L": L_grid, "∑²(L)": sigma_sq})
plot.levelNumberVariance(unfolded, df, title=f"{kind.upper()} Matrix", mode="block")

```

# Documentation
Be sure to read the documentation comments in the [source code](https://github.com/stfxecutables/empyricalRMT/tree/master/empyricalRMT).