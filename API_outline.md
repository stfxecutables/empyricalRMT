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
    .trim_manual()              -> Trimmed
    .trim_unfold()              -> Unfolded
    .unfold(smoother: Smoother) -> Unfolded
```

## Trimmed

```python
class Trimmed(EigVals):
    .trim_report()              -> pd.DataFrame
    .trim_indices()             -> (int, int)
    .plot_trimmed()             -> Axes
    .unfold()                   -> Unfolded
```

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
