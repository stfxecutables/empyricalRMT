# Notes on Trimming

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

In short, the "trimming" procedure struggles with much of the same issues that arise
in outlier detection. Just as there, there is no real *general* way to justify
calling a particular data point an "outlier", there is no general way to identify a "good
trim" here in RMT. Whether or not a datapoint is an "outlier" depends entirely on the
goals of one's statistical analysis.

Since we are here dealing with RMT, we really only have a few analytic goals. The "optimal
trim percent" depends on those analytic goals. If our analytic goals are:

- to examine the "chaotic" / "noisy" parts of a system:
    * if we have correlation matrices, then we can use Marcenko-Pastur, or the time-series
      modification in Almog et al. (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6513117/#sec008title)
- to compare spectral observables for a consistent smoother across multiple matrices:
    * then we don't care so much about the exact trim portion, we are trimming simply to
      ensure we have decent fit, and so we can use goodness of fit metrics to determine
      a sufficiently good trim
    * here, we should proably just make sure trimming is consistent (identical) across
      matrices being compared
- to examine how *much* or what *portion* of a system looks noisy/chaotic/GOE:
    * then we want to find the trimming that results in a set of eigenvalues that looks
      most GOE (under a particular unfolding) across the most spectral observables

## Affirming the Consequent

If developing a reliable trimming and unfolding procedure, there is a simple logical
difficulty. For any system, we know that:

```
system is GOE ==> expected GOE curves / observables
```

However, we do NOT know that:

```
expected GOE curves ==> GOE system
```

Thus, any trimming and unfolding procedure must result in the expected curves *only* when
a system really is GOE. If the procedures result in such curves for non-GUE systems, the
trimming / unfolding procedure is uninformative.

This is an issue for polynomial unfolding procedures, since testing shows that these can
yield GOE-like Nearest-Neighbour Spacing Distributions for, e.g., some orthogonal uniform
matrices.