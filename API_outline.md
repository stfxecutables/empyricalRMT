# Actual "Objects" in RMT

## Eigenvalues

```python
class Eigenvalues:
    .plot()                     -> matplotlib.axes
    .steps()                    -> np.array
    .step(x: np.float)          -> np.float
    .trim()                     -> Trimmed
    .trim_manual()              -> Trimmed
    .trim_unfold()              -> Unfolded
    .unfold(smoother: Smoother) -> Unfolded
```

## Trimmed

```python
class Trimmed:
    .eigs()         -> np.array
    .eigenvalues()  -> np.array
    .trim_report()  -> pd.DataFrame
    .trim_indices() -> (int, int)
    .unfold()       -> Unfolded
```

## Smoother

```python
class Smoother():
    .fit(eigs: Eigenvalues, method="poly"|"spline"|"gompertz"|"emd", emd_detrend: boolean) -> Unfolded:
    .fit_all()
```

## Unfolded

```python
class Unfolded:
    .original_eigs               -> Eigenvalues
    .trimmed                     -> Trimmed
    .spacings()                  -> np.array
    .plot()                      -> matplotlib.axes
    .compare(ensemble: Ensemble) -> pd.DataFrame report
    .nnsd()                      -> pd.DataFrame | np.array
    .level_variance()            -> pd.DataFrame | np.array
    .spectral_rigidty()          -> pd.DataFrame | np.array
```

## Ensemble

```python
class Ensemble:
    .GOE
    .GUE
    .GSE
    .GDE / Poisson
    .m-GOE (GOE with m blocks)
        .generate_eigs(unfold: boolean) -> Eigenvalues | Unfolded
        .nnsd()                         -> np.array
        .spectral_rigidity()            -> np.array
        .level_variance()               -> np.array
```
