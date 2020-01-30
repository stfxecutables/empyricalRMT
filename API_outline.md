# Folder Structure

To be decided.


# Actual "Objects" in RMT

## Helper Class

```python
class EigVals:
    .__construct_vals           -> np.array
    .original_values            -> np.array
    .original_eigs              -> np.array
    .vals                       -> np.array
    .values                     -> np.array
    .steps                      -> np.array
    .spacings                   -> np.array
    .step_function(x: np.array) -> np.array
    .plot_sorted()              -> Axes
    .plot_distribution()        -> Axes
    .plot_steps()               -> Axes
    .plot_spacings()            -> Axes
```

## Eigenvalues

```python
class Eigenvalues(EigVals):
    .trim()                     -> Trimmed
    .trim_manually()            -> Trimmed
    .trim_interactively()       -> Trimmed
    .trim_unfold()              -> Unfolded
    .unfold(smoother: Smoother) -> Unfolded
```

## Trimmed

Doesn't make sense to keep trimmming, trimming should be an operation that is
only done once, and on the original eigenvalues. We can enforce this by removing
trimming methods on Trimmed objects.

```python
class Trimmed(EigVals):
    .compute_trims                -> pd.DataFrame
    .trim_best_indices          -> (int, int)
    ._trim_steps                -> [dict]
    ._trim_best_index           -> int  # location in compute_trims list of best overall trim
    .plot_trimmed()             -> Axes
    .unfold()                   -> Unfolded
```

However, trimming is not, in general, a simple, one-off endeavor. Typically, it is not
clear just *how* much must be trimmed before we have "useful" eigenvalues. That is, we
are always going to be comparing multiple possible trimmings, and so we want some way
to carefully track and compare these different trimming results.

In general, we will only be concerned with:

1. How many extreme eigenvalues to trim
2. whether there are dramatic differences across smoothers for each unfolding

Ideally, we want a "smoother free" trimming. However, this is fundamentally impossible.

We trim only because large spacings in the (sorted) eigenvalues cause "bad" unfoldings.
And we decide we have a "bad" trimming when the smoother used to perform the unfolding on
that trimming results in approximations of the smooth part (as opposed to the fluctuating
part) that don't even yield the expected empirical observables for samples we *know* come
from specific ensembles.

Or, alternately, we know that most choices of smoothers will overfit extreme values
(extreme values act as anchor points). We trim not because we are removing "aberrant parts
of the system" but because we want in general to fit only the "majority of the system".
Smoothers that *aren't locally sensitive to extreme tails* (e.g. piecewise splines) should
not, in theory, have these issues. And we do see this in simulations.

In short, the "trimming" procedure is identical to the general problem of "outlier
detection" in data analysis. Just as there, there is no real *general* way to justify
calling a particular data point an "outlier", there is no general way to identify a "good
trim" here in RMT. Whether or not a datapoint is an "outlier" depends entirely on the
goals of one's statistical analysis.

Since we are here dealing with RMT, we really only have a few analytic goals. The "optimal
trim percent" depends on those analytic goals. If our analytic goals are:

1. To assess the "percent chaoticity" of a ***total*** system
    * then we want to find the trimming that results in a set of eigenvalues that looks
      most GOE
    * for the long-range spectral observables of such a system, we might then think of
      further analysis after such a trimming as an analysis of the "non-central"
      (non-principal, remainder) comopnents of the system
        - e.g. the handful of principal components are interesting, but the massive
          remainder of non-principal components is in some ways of comparable "size"
          to the few principal components, so RMT gives us a way to identify and analyze
          those elements of the system

That is,
the only way to determine whether a particular trimming is "good" is to ask if those
trimmed values look "mostly smooth". And the only way to assess if the trimmed values are
"smooth" is to fit a smoother and see that the information loss is low.

Now, from a practical standpoint of developing a reliable trimming and unfolding
procedure, there is a siple logical difficulty. For any system, we know that:

```
if GOE system, then ==> expected GOE curves
```

However, we do NOT know that:

```
if expected GOE curves, then ==> GOE system
```

Thus, any trimming and unfolding procedure must result in the expected curves *only* when
a system really is GOE. If the procedures result in such curves

```python
class TrimReport:
    ._trim_iters                -> [DataFrame]
    ._best_iters                -> int



## Smoother

```python
class Smoother():
    .fit(eigs: Eigenvalues, smoother="poly"|"spline"|"gompertz"|"emd", emd_detrend: boolean) -> Unfolded:
    .fit_all(eigs: Eigenvalues, *args)
```

## Unfolded

```python
class Unfolded(RawEigs):
    .trimmed                     -> Trimmed
    .plot_smoother_fit()         -> Axes
    .compare(ensemble: Ensemble) -> pd.DataFrame report
    .nnsd()                      -> pd.DataFrame | np.array
    .level_variance()            -> pd.DataFrame | np.array
    .spectral_rigidty()          -> pd.DataFrame | np.array
```

## Ensemble

```python
class Ensemble:
    .generate_eigs(unfold: boolean) -> Eigenvalues | Unfolded
    .nnsd()                         -> np.array
    .spectral_rigidity()            -> np.array
    .level_variance()               -> np.array

class GOE(Ensemble):
class GUE(Ensemble):
class GSE(Ensemble):
class GDE(Ensemble):
class m_GOE(Ensemble): # m GOE blocks
```
