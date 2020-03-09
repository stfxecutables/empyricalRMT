# Table of Contents
- [Table of Contents](#table-of-contents)
- [Introduction](#introduction)
- [Installation](#installation)
  - [Using `venv` (recommended)](#using-venv-recommended)
  - [From Source](#from-source)
  - [From pip](#from-pip)
  - [Notes on Windows](#notes-on-windows)
- [Basic Usage](#basic-usage)
- [Documentation](#documentation)
- [Development](#development)
  - [Installation](#installation-1)
  - [Testing](#testing)

# Introduction

A python library for Random Matrix Theory eigenvalue unfolding and
computation and plotting of spectral observables.

This libary is still undergoing major re-writes and development, and should
be considered in pre-alpha at this point. Feel free to post issues or ask
questions relating to the code, but keep in mind any bugs and/or issues are
very likely to change in the near future.

# Installation

As always, using a virtual environment is recommended for proper use.
However, you _should_ be okay doing a global `pip install empyricalRMT`
to try out the library.

## Using `venv` (recommended)

Navigate to the project that you wish to use empyricalRMT in.

```bash
cd MyProject
```

Create and active the virtual environment. Replace "env" with whatever name
you prefer.

```bash
python -m venv env && source env/bin/activate
```

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

Note that this will install the library "globally" if you haven't activated
a virtual environment of some kind.

## Notes on Windows

This _should_ still all work on Windows, although you may have to follow
[modified instructions for setting up the *venv*](https://docs.python.org/3/library/venv.html).

If you run into issues you think might be Windows-related, please do report
them, but keep in mind I currently can only test on Windows via virtual machine :(

# Basic Usage

```python
import empyricalRMT.observables.unfold as unfold
import empyricalRMT.plot as plot

from empyricalRMT.observables.levelvariance import level_number_variance
from empyricalRMT.observables.rigidty import spectralRigidity
from empyricalRMT.construct import generateGOEMatrix, generate_eigs

# generate a new matrix from the Gaussian Orthogonal Ensemble and extract
# its eigenvalues
eigs = generate_eigs(matsize=1000, kind="goe")
# eigs = generate_eigs(matsize=1000, kind="gue")
# eigs = generate_eigs(matsize=1000, kind="poisson")

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
L, sigma_sq = level_number_variance(eigs, unfolded, c_iters=1000, L_grid_size=100, min_L=0.5, max_L=20)
# Plot the resultant values
df = pd.DataFrame({"L": L_grid, "∑²(L)": sigma_sq})
plot.levelNumberVariance(unfolded, df, title=f"{kind.upper()} Matrix", mode="block")

```

# Documentation

Be sure to read the documentation comments in the [source code](https://github.com/stfxecutables/empyricalRMT/tree/master/empyricalRMT).

# Development

## Installation

Assuming you want your `venv` virtual environment to be named "env":

```bash
git clone https://github.com/stfxecutables/empyricalRMT
cd empyricalRMT
python -m venv env
source env/bin/activate
python -m pip install -r requirements-dev.txt
```

## Testing

To run all tests, run:

```bash
python -m pytest -v
```

There are a number of pytest _marks_ labelling different testing aspects.
Brief descriptions can be found in `pytest.ini`. However, likely most useful
will be running:

```bash
python -m pytest -v -m fast
```

which runs all tests that _shouldn't_ take too long to execute.
