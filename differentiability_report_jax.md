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
| posterior     | nested sampling, thousands of likelihood evaluations                                              | mode and Hessian, then `laplace`, `nuts` or `smc`                          |
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

- `laplace` (default): `num_samples` draws from that Gaussian give the medians,
  the errors and the profile bands; `covariance` is the Hessian's. The Hessian is
  inverted after its eigenvalue spectrum is clipped at `eigenvalue_floor` of the
  largest, which keeps the matrix positive definite while a well conditioned but
  correlated Hessian keeps its exact inverse. Retrieval parameters are strongly
  correlated, so a diagonal ridge strong enough to force positive definiteness
  (the Gershgorin bound) would add the off-diagonal row to every diagonal entry
  and shrink every error bar by an order of magnitude; that is what an earlier
  revision did, and it is why its Laplace posteriors looked too narrow.
- `nuts`: blackjax NUTS from the mode, whitened by the Laplace covariance. Exact,
  equally weighted draws, one trajectory of gradients each. Its cost is
  `num_samples * leapfrogs_per_draw` gradients, so the Laplace covariance is what
  keeps the trajectories short. The run logs the leapfrogs per draw: tens is a
  healthy whitening, and a value near the `max_num_doublings` ceiling means the
  metric is wrong, not that the posterior is hard.
- `smc`: adaptive tempered sequential Monte Carlo. It draws a defensive
  mixture of a Student-t around the mode (Laplace covariance) and the prior,
  corrects the initial weights by the exact prior, and only tempers and runs
  fixed-length HMC for the part the proposal misses. When the proposal carries
  the posterior the run spends no gradients at all; the sum of the stage
  normalising constants is the Bayesian evidence. The zero-gradient shortcut is
  taken only when the proposal weights pass the `k-hat <= 0.7` check as well as
  the ESS check; a heavy-tailed proposal is tempered instead. The posterior is
  the weighted particle population, so `num_particles` is also its resolution:
  the class default of 128 leaves about 75 effective draws, and 512 or more are
  needed for a posterior comparable to a nested run. `smc_start = prior` is the
  slower, multi-modal fallback; `smc_fit = lbfgs` leaves the inverse Hessian that
  the proposal and the metric start from.

The samplers report the sample covariance; `smc` reports the weighted one. All
three go through the usual `get_solution`, `get_samples` and `get_weights`, so
output and post-processing are unchanged. `smc` additionally fills
`log_evidence` and `log_evidence_error` and writes them under `GlobalStats` in
the output file, under the keys the nested samplers use. Extra diagnostics:
`n_starts`, a fitted `jitter`, `fisher_information()`, and the timing and
evaluation counters.

## Flat directions

A retrieval posterior can be nearly flat over a large part of the prior. Two
different things look like that, and they need different treatment.

- *Far from the posterior* the likelihood is almost constant and the gradient
  carries no direction. This is a volume problem, not a gradient problem: the
  posterior occupies a tiny fraction of the prior volume (`e^-H` for an
  information gain `H`, which is of order 100 nats in a high signal-to-noise
  transmission spectrum). It is why a tempering ladder from the prior is
  hopeless and why nested sampling, which shrinks the prior volume, stays
  competitive. The answer is not to follow the gradient across the flat region
  but to start at the mode and propose from the covariance; `smc_start =
  laplace` and `smc_fit = lbfgs` do that.
- *Inside the posterior*, some coordinates carry no data curvature at all. Their
  posterior is the prior, but the clipped Laplace covariance can still be far
  narrower than the prior there, which stretches the whitened target and turns
  a flat direction into trajectories of thousands of leapfrogs, or an SMC
  proposal whose weights collapse to one particle. The optimiser therefore
  computes the data curvature as the Hessian of the negative log posterior
  minus its prior-and-Jacobian part, and for a coordinate whose data error bar
  exceeds `prior_dominated_sigma` (default 3) prior standard deviations it uses
  the prior's own variance in `u` (the logistic `pi^2/3`) for the whitening and
  the SMC proposal. The target is unchanged, the effective dimension is logged,
  and `prior_dominated_sigma = None` turns the test off.

The run logs the file the optimiser was imported from, the covariance spectrum,
the NUTS leapfrogs per draw and the SMC stages, so a cluster run can be checked
without guessing which build is installed.

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
quasi-Newton chain is sequential. `submit.sh` runs the JAX samplers as one task
on one node, with XLA spreading the compiled programs over its cores;
`_run_smc` can still shard the particle axis over MPI ranks when the optimizer is
launched under `mpirun` directly, but that layout compiles a model per rank and
adds a collective to every likelihood sweep, so it only pays off across several
nodes. `poetry.lock` has no `jax`, `optax` or `blackjax`: run `poetry lock`
first.

## Which sampler

`laplace` is the cheap baseline and the source of the mode, the Hessian and the
initial metric every other sampler uses; keep it for the MAP and for a first
look. Its error bars are only as good as the Gaussian, but with the clipped
inverse Hessian their width is right for a correlated, well conditioned fit.

`nuts` is the reference posterior: exact, equally weighted, and with the Laplace
whitening it costs tens of gradients per draw. Its wall clock is
`num_samples * leapfrogs_per_draw`, so the two knobs that matter are the metric
(check the logged leapfrogs; if they sit at the `max_num_doublings` ceiling,
raise `num_warmup`) and `num_samples` (1000 draws is ample for an 11-parameter
corner plot).

`smc` is the cheapest route to a posterior that is not a Gaussian, and the only
sampler that returns `log Z`. A stage costs
`num_mcmc_steps * num_particles * (smc_hmc_steps + 1)` gradients, plus one
likelihood sweep to choose the tempering increment and one to reweight, so its
cost is `stages * particles` and its posterior resolution is the particle count.
128 particles is cheap but coarse; 512-1024 is the range for a final run. Start
it from the mode (`smc_start = laplace`, `smc_fit = lbfgs`) so the proposal does
the work; a prior ladder pays many more stages for the same posterior.

These three are the whole set: `laplace`, `nuts` and `smc`. The `importance`
sampler that earlier revisions exposed is the zero-gradient branch of `smc`
without its fallback, and `mclmc` was only worth its `O(step^2)` bias when one
gradient was much cheaper than a NUTS trajectory; both are gone rather than
left as weaker duplicates.

`submit.sh` encodes that: `./submit.sh 1 128 50 jax smc` runs 512 particles from
the mode with an 8-leapfrog HMC kernel, and `... jax nuts` runs 1000 draws over
16 chains with a 200-step warm-up. Both run as one task on one node and share
the compiled programs through `JAX_COMPILATION_CACHE_DIR`, so the second run of
a configuration skips the minutes of compilation.

Two parameterisation aids sit under the samplers. `mode = auto` picks a log
prior when a parameter's bounds are both positive and span more than two
decades, and a linear prior over more than two decades is reported as a warning
because the sigmoid-to-prior transform is then ill-conditioned. A short NUTS
pilot (`pilot_warmup`, `pilot_samples`) can replace the local Laplace
covariance with a blend of the pilot and Laplace metrics before the SMC or NUTS
run; `optimizer.pilot_parameters` reports the blend weight and the trace ratio.
The pilot is a warm-cache or quality choice: it adds a NUTS compile.
