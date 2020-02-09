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
    .eigs                      -> ndarray
    .eigenvalues               -> ndarray
    .trim_report()              -> TrimReport
    .get_best_trimmed()         -> Trimmed
    .trim_marcenko_pastur()     -> Trimmed
    .trim_manually()            -> Trimmed
    .trim_interactively()       -> NotImplementedError
    .trim_unfold()              -> Unfolded
    .trim_unfold_auto()         -> TrimReport, Unfolded
    .unfold(smoother: Smoother) -> Unfolded
    .unfold_auto()              -> TrimReport, Unfolded
```

## Trimmed

```python
class TrimReport:
    .untrimmed                   -> ndarray
    .unfold_info                 -> DataFrame
    .unfoldings                  -> DataFrame
    .plot_trim_steps()           -> (Figure, Axes)
    .compare_trim_unfolds()      -> DataFrame
    .summarize_trim_unfoldings() -> DataFrame
    .unfold()                    -> Unfolded
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
