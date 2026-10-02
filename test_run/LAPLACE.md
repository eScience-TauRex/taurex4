# The Laplace optimizer

`taurex.differentiability.optimizer.LaplaceOptimizer`

A retrieval that replaces nested sampling with a gradient-based fit followed by a
Gaussian (Laplace) approximation of the posterior.

---

## 1. Why it exists

A normal TauREx retrieval uses a sampling optimizer (MultiNest, NestedSampler, ...).
Those explore the whole posterior, but they have no gradient to follow, so they
need thousands of forward-model evaluations. A differentiable atmosphere does have
a gradient, which means the same two answers can be obtained much more cheaply:

| Output          | Sampling optimizer             | Laplace optimizer                             |
| --------------- | ------------------------------ | --------------------------------------------- |
| Best-fit values | mode of the samples            | maximisation of the posterior (MAP)           |
| Error bars      | spread of the samples          | width of a Gaussian at the MAP                |
| Cost            | thousands of model evaluations | tens to hundreds of evaluations + one Hessian |
| Posterior shape | any                            | assumed Gaussian                              |

That last row is the trade-off: the Laplace approximation is only as good as the
assumption that the posterior is roughly a single Gaussian blob.

---

## 2. The pieces

### `PriorTransform` — cube to parameter map

One instance per fitting parameter, built from the parameter's taurex `Prior`.

- **Uniform / LogUniform** (`prior._low_bounds`, `prior._up_bounds`) — linear map
  from the cube coordinate to `[low, high]`.
- **Gaussian / LogGaussian** (`prior._loc`, `prior._scale`) — the inverse normal
  CDF via `jax.scipy.special.erfinv`, clipped just inside the cube edges because
  the inverse CDF diverges there (the clip sits at ~5.5σ, so no usable prior is
  affected).

It provides:

- `forward(cube)` — cube coordinate in `(0, 1)` → parameter value
- `inverse(theta)` — parameter value → cube coordinate (uses `scipy.special.ndtr`)
- `log_prior(theta)` — log prior density in parameter space. A flat prior returns
  zero: it shifts the posterior but neither its mode nor its curvature.

Any other prior type raises `NotImplementedError`, since a gradient cannot be
taken through it.

### `Atmosphere` — the forward model

`taurex.differentiability.model.Atmosphere` wraps the taurex object graph in JAX.
Relevant entry points:

- `spectrum(theta)` — model spectrum for a parameter vector
- `chi_squared(theta)` — `sum(((data - spectrum) / error)**2)`
- `log_likelihood(theta)` — `-log_normalisation - 0.5 * chi_squared(theta)`,
  matching the convention of the other taurex optimizers

### `LaplaceOptimizer` — the driver

Subclasses `taurex.optimizer.optimizer.Optimizer`, so it is loaded by the input
file driver exactly like any other optimizer.

---

## 3. How a fit runs

### Step 1 — build the problem

```python
atmosphere = Atmosphere(self._model, self._observed, fit_params=fit_params, dtype=jnp.float64)
transforms = [PriorTransform(p.fit_prior, p.name, dtype) for p in fit_params]
```

The starting point `_starting_point` inverts the parameter values from the input
file through each prior's `inverse`, clips into `[1e-6, 1 - 1e-6]`, and applies
`logit` to give the unconstrained start `u0`.

### Step 2 — reparameterise into an unconstrained space

The optimizer always works on a vector `u` with no constraints:

```
cube  = sigmoid(u)          # in (0, 1)
theta = PriorTransform.forward(cube)   # real parameter, inside the prior support
```

`parameters_of(u)` does this for every parameter and stacks along the last axis,
so it also works for a batch of vectors (used later for the samples).

Why bother? Two reasons:

1. **Bounds and priors are respected exactly** — the mapping is built from the
   prior itself, so the optimizer cannot step outside the allowed region and no
   penalty terms are needed.
2. **Conditioning** — retrieval parameters can span many orders of magnitude. In
   `u` space the sigmoid saturation keeps everything of order one, which is what
   makes the Hessian numerically tractable later.

### Step 3 — the objective

```python
def negative_log_posterior(u):
    theta = parameters_of(u)
    log_prior = sum(t.log_prior(theta[i]) for i, t in enumerate(transforms))
    return -atmosphere.log_likelihood(theta) - log_prior
```

Note the prior is written as a density **over `theta`**, not over `u`. That keeps
the mode and the covariance in the same space as the reported parameters. With
flat priors the objective reduces to the chi-squared minimum.

### Step 4 — L-BFGS to the maximum a posteriori point

The whole fit is one compiled program:

```python
solver = optax.lbfgs(
    learning_rate=1.0,
    memory_size=self.memory_size,
    scale_init_precond=self.scale_init_precond,
    linesearch=optax.scale_by_zoom_linesearch(
        max_linesearch_steps=self.max_lbfgs_iterations
    ),
)
value_and_grad = optax.value_and_grad_from_state(negative_log_posterior)
```

- `jax.lax.while_loop` contains the step, the line search and the stopping test,
  so the cost is one device round trip per step rather than a Python callback.
- Stopping: gradient infinity-norm below `gradient_tolerance`, or
  `max_iterations` steps. The initial gradient norm is `inf` so at least one step
  is always taken.
- `optax.lbfgs` rather than `jax.scipy.optimize.minimize`: the latter aborts the
  entire optimization when a line search fails the Wolfe conditions, and retrieval
  objectives spanning many orders of magnitude fail them easily. optax keeps the
  best step it found and carries on.
- `scale_init_precond=True` scales the initial inverse-Hessian estimate by the
  reciprocal gradient magnitude, capping the first step at a unit Euclidean ball.
  This is what makes the fit converge from raw input-file values with no tuning.
- `value_and_grad_from_state` reuses the value and gradient at the last accepted
  point, so a line-search trial costs a forward pass rather than forward + backward.

Everything is lowered and compiled up front, including the Hessian, the transform
(both for one vector and for the sample batch), the Jacobian and the likelihood.
That one-off cost is measured as `compile_time`, and `fit_time` is measured around
`jax.block_until_ready(solve(u))` so the wall clock reflects execution, not JAX's
asynchronous dispatch.

If the fit stops at the iteration limit with the gradient still above tolerance it
emits a warning — the difference between "this is the mode" and "this is as far as
the fit got".

### Step 5 — the Laplace approximation

`_laplace_covariance(hessian_of, u_map)`:

```python
hessian = hessian_of(u_map)
hessian = 0.5 * (hessian + hessian.T)          # symmetrise
largest = float(jnp.max(jnp.abs(hessian)))
scaled  = hessian / largest                    # bound the matrix
diagonal    = jnp.diagonal(scaled)
off_diagonal = jnp.sum(jnp.abs(scaled), axis=1) - jnp.abs(diagonal)
shift = off_diagonal + jnp.clip(-diagonal, 0.0, None) + self.eigenvalue_floor
covariance = jnp.linalg.inv(scaled + jnp.diag(shift))
```

Points worth explaining to a user:

- The Hessian is taken in **`u` space**, where the posterior is well conditioned.
- `jax.hessian` uses forward-over-reverse: one reverse pass per parameter, instead
  of the reverse-over-reverse nesting a Jacobian of a gradient would cost.
- A retrieval Hessian spans many orders of magnitude across its eigenvalues, and
  its numerically null directions are what break a plain inversion. Dividing by
  the largest entry bounds the matrix; the ridge of `eigenvalue_floor` (default
  `1e-3`) caps the variance of directions the data does not constrain, so those
  parameters come back **prior dominated** rather than undefined. Constrained
  directions sit orders of magnitude above the ridge and are unaffected.
- The shift is chosen so the scaled matrix is strictly diagonally dominant, which
  by Gershgorin's theorem makes it positive definite regardless of numerical noise.
- A flat or non-finite Hessian is not an error: it means the data constrains
  nothing, and every sample collapses onto the optimum.
- `variance_scale = 1 / largest` is carried separately so the covariance never has
  to be formed in the badly scaled space.

### Step 6 — samples, medians and covariance in parameter space

`_draw_samples`:

```python
factor = jnp.linalg.cholesky(covariance)                       # in scaled space
normal = jax.random.normal(key, (num_samples, ndim))
offset = normal @ factor.T * sqrt(variance_scale)
theta  = transform_of(u_map[None, :] + offset)                 # one traced call
```

- The Cholesky factor is taken in the **scaled** space for conditioning and applied
  at the end, avoiding the ill-conditioned covariance of `u`.
- Draws are made in `u` space and pushed through the priors, so every sample lies
  inside its prior's support however wide the Gaussian is.
- `seed` makes a retrieval reproducible. All draws go through the transform in a
  single traced call; a Python loop over two thousand draws would be pure dispatch
  overhead.
- `_median` is the median of these samples.

`_parameter_covariance` carries the covariance back with the Jacobian of the
`u → theta` map:

```
Sigma_theta = J @ (Sigma_u) @ J.T
```

### Step 7 — what is stored

| Attribute                                                    | Meaning                        |
| ------------------------------------------------------------ | ------------------------------ |
| `_map`                                                       | parameter values at the MAP    |
| `_median`                                                    | median of the Laplace samples  |
| `_samples`                                                   | `(num_samples, nparams)` draws |
| `_covariance`                                                | covariance in parameter space  |
| `_log_posterior`                                             | log posterior at the MAP       |
| `_log_likelihood`                                            | log likelihood at the MAP      |
| `iterations`, `function_evaluations`, `gradient_evaluations` | fit bookkeeping                |
| `compile_time`, `fit_time`, `laplace_time`                   | timing breakdown               |

---

## 4. Interface

Implements the standard optimizer interface:

- `get_solution()` yields `(0, self._map, self._median, [])`
- `get_samples(solution_id)` returns `_samples`
- `get_weights(solution_id)` returns uniform weights, because the Laplace draws are
  already posterior samples (no thinning needed — see `sigma_fraction`, which can
  therefore be much smaller than for a chain-based optimizer)
- `covariance`, `log_posterior`, `log_likelihood_map` properties

Input file keywords (`input_keywords`): `jax`, `jax_laplace`, `laplace`,
`differentiable`.

Unknown keyword arguments are ignored, so an input file written for a sampling
optimizer still loads.

---

## 5. Options

| Option                 | Default     | Effect                                                                                                                                          |
| ---------------------- | ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `num_samples`          | 2000        | number of posterior draws                                                                                                                       |
| `max_iterations`       | 200         | maximum quasi-Newton steps                                                                                                                      |
| `max_lbfgs_iterations` | 40          | line-search evaluations per step                                                                                                                |
| `memory_size`          | 10          | L-BFGS curvature pairs; 0 gives steepest descent with a line search                                                                             |
| `scale_init_precond`   | True        | scale the first step to a unit ball; makes fitting from input-file values work without tuning                                                   |
| `gradient_tolerance`   | 1e-10       | convergence threshold on the gradient norm                                                                                                      |
| `eigenvalue_floor`     | 1e-3        | ridge on the scaled Hessian before inversion                                                                                                    |
| `seed`                 | 0           | seed for the posterior draws                                                                                                                    |
| `device`               | JAX default | e.g. `cpu`; by default the GPU when one is visible                                                                                              |
| `sigma_fraction`       | 0.02        | fraction of samples reused for profile uncertainties; the dominant cost, because that step goes through the numpy model on the full native grid |

---

## 6. Limitations

- Assumes a single, roughly Gaussian posterior. Multi-modal or banana-shaped
  posteriors are reported as one blob with misleading error bars.
- The result depends on the starting point, unlike nested sampling.
- If the Hessian is dominated by a poorly determined direction, the ridge makes
  that parameter prior dominated — correct behaviour, but it should be visible in
  the reported uncertainties.
- Priors other than Uniform, LogUniform, Gaussian and LogGaussian are not
  supported (no differentiable inverse CDF).
- A flat objective is handled gracefully but yields no information.
