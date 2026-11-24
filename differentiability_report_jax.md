# What taurex4 adds to TauREx3

Nothing in TauREx3 changes — numpy model, chemistry, contributions, priors and
output all stay. The addition is `taurex.differentiability`: the transmission
model re-expressed in `jax.numpy`, compiled and differentiable, and a new
optimizer that fits it by gradient. Input files, `[Fitting]`, priors, bounds and
the HDF5 layout are untouched.

|               | TauREx3                                                                                           | taurex4                                                                   |
| ------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| forward model | numpy, eager                                                                                      | JAX, traced and JIT-compiled, CPU or GPU                                  |
| optimizers    | `multinest`/`pymultinest`, `nestle`, `polychord`/`pypolychord`, `dypolychord`/`dynamic-polychord` | the same four, plus `laplace` (`jax`, `jax_laplace`, `differentiable`)     |
| posterior     | nested sampling, thousands of likelihood evaluations                                              | mode and Hessian, then `sampler = laplace`, `nuts`, `importance`, `mclmc` or `smc` |
| evidence      | yes                                                                                               | `smc` only: adaptive tempered sequential Monte Carlo returns `log Z`       |

## Package and packaging

`src/taurex/differentiability/` holds `physics.py` (profile, path matrix, opacity
interpolation, smoothing), `model.py` (`Atmosphere`), `optimizer.py`
(`LaplaceOptimizer`) and `__init__.py`. `pyproject.toml` gains the
`differentiable` extra (`jax`, `optax`, `blackjax`) and the entry point that
resolves `optimizer = laplace`.

```console
$ pip install -e . --no-deps
$ pip install -e .[differentiable]
```

## What the new optimizer adds

`optax.lbfgs` drives the model to the mode in an unconstrained coordinate mapped
through a sigmoid and each parameter's prior inverse CDF, so the priors and
bounds of the input file hold by construction; the likelihood convention is
TauREx3's. `jax.hessian` at the mode gives the covariance. The `sampler` option
picks what happens next, all of it off that one fit:

- `laplace` (default): `num_samples` (2000) draws from that Gaussian give the
  medians, the errors and the profile bands; `covariance` is the Hessian's.
- `nuts`: blackjax NUTS from the mode, whitened by the Laplace covariance. Exact,
  equally weighted draws, one trajectory of gradients each.
- `importance`: Student-t proposals from the Laplace covariance reweighted by the
  exact density; one forward pass each and no gradients. The proposals are kept
  with their weights, and the effective sample size and the Pareto tail index
  `k-hat` say how far the posterior is from the Gaussian; above `k-hat = 0.7`
  a few proposals decide the answer and the run says so.
- `mclmc`: blackjax microcanonical Langevin, one gradient and one sample per step,
  no accept test, integrator error `O(step^2)`, at least two fitted parameters.
- `smc`: adaptive tempered sequential Monte Carlo. It draws a defensive
  mixture of a Student-t around the mode (Laplace covariance) and the prior,
  corrects the initial weights by the exact prior, and only tempers and runs
  fixed-length HMC for the part the proposal misses. When the proposal carries
  the posterior the run spends no gradients at all; the sum of the stage
  normalising constants is the Bayesian evidence. The zero-gradient shortcut is
  taken only when the proposal weights pass the `k-hat <= 0.7` check as well as
  the ESS check; a heavy-tailed proposal is tempered instead. `smc_start = prior`
  is the slower, multi-modal fallback.

The samplers report the sample covariance; `smc` reports the weighted one. All
five go through the usual `get_solution`, `get_samples` and `get_weights`, so
output and post-processing are unchanged. `smc` additionally fills
`log_evidence` and `log_evidence_error` and writes them under `GlobalStats` in
the output file, under the keys the nested samplers use. Extra diagnostics:
`n_starts`, a fitted `jitter`, `fisher_information()`, and the timing and
evaluation counters.

## The supported subset

- temperature `Isothermal`, `NPoint`; chemistry `TaurexChemistry` with
  `ConstantGas`;
- contributions `Absorption`, `Rayleigh`, `CIA`, `FlatMie`, `LeeMie`;
- opacities: cross-section (`xsec`) tables only, not `ktables`;
- binning `FluxBinner`, `NativeBinner`.

Anything else raises `NotImplementedError` naming the class, rather than
silently falling back to a different implementation.

## Changes to the shared numpy path

The TauREx3 `#120` memory fixes are carried here: `_last_wngrid`, a warm-up on
the observed grid with `trim_memory()`, `store_contributions` reusing that grid,
and `h5py` datasets streamed into shared memory in 32 MB chunks. Consequence:
`Output/Spectra/native_*` and the per-contribution `native_*` arrays are written
on the clipped grid of the last evaluation; `binned_*` is unaffected.

## Cost and parallelism

One 6-core CPU, 80 layers, 4395 native points, 488 bins: forward 3.7 ms against
59.7 ms for numpy, gradient 24 ms, compile 6.2-7.0 s per `compute_fit` (half of a
fit; `JAX_COMPILATION_CACHE_DIR` persists it), retrieval 13-16 s, peak RSS
0.81 GB against 3.45 GB, native spectrum within `2.8e-16` relative of numpy.
Under MPI the mode fit runs on the root rank and is broadcast, because the
quasi-Newton chain is sequential; `smc` also shards its particle axis over the
ranks when `SMC_MPI=1`, while every other sampler runs in one JAX process that
XLA spreads over the node's cores. `poetry.lock` has no `jax`, `optax` or
`blackjax`: run `poetry lock` first.

## Which sampler

`smc` is the cheapest route to a posterior that is not a Gaussian. One tempering
stage costs one HMC trajectory per particle: `smc_hmc_steps` leapfrogs (two by
default) plus the gradient that initialises it, against tens to hundreds of
gradients per `nuts` draw, and a proposal that already carries the posterior
ends the run in one forward pass per particle with no gradient. It is also the
only sampler that returns `log Z` and the only one that can use more than one
MPI rank. Keep `nuts` as the reference on a reduced case, and use `mclmc` only
when the gradient is the bottleneck and its `O(step^2)` bias is acceptable.

Two parameterisation aids sit under the samplers. `mode = auto` picks a log
prior when a parameter's bounds are both positive and span more than two
decades, and a linear prior over more than two decades is reported as a warning
because the sigmoid-to-prior transform is then ill-conditioned. A short NUTS
pilot (`pilot_warmup`, `pilot_samples`) can replace the local Laplace
covariance with a blend of the pilot and Laplace metrics before the SMC or NUTS
run; `optimizer.pilot_parameters` reports the blend weight and the trace ratio.
The pilot is a warm-cache or quality choice: it adds a NUTS compile.
