"""Retrieval that uses the gradient of the forward model.

Nested sampling needs thousands of likelihood evaluations because it has no
gradient to follow, while the differentiable atmosphere does have one, so the
posterior is located with a quasi-Newton optimiser and then described by a
Laplace approximation around the maximum a posteriori point.

1. the posterior is maximised over an unconstrained vector ``u`` that is mapped
   onto the fitting parameters through their own priors, so the bounds and
   priors from the input file are respected exactly;
2. the Hessian of the negative log posterior at the optimum gives a Gaussian
   covariance that is then transformed back to parameter space, from which
   posterior samples, medians and error bars follow.

What changes with JAX is how the arithmetic is scheduled. Nothing is executed
eagerly: the objective and its gradient are traced once, the whole optimiser
loop is compiled into a single program, and the Hessian is taken with
forward-over-reverse mode, which costs one reverse pass per parameter instead of
the reverse pass over a reverse pass that differentiating a gradient would cost.
A fit is therefore a handful of XLA programs with no python callback between
them.

The quasi-Newton step itself is :func:`optax.lbfgs`, an L-BFGS with a
strong-Wolfe zoom line search, which is the same algorithm the torch
implementation used. It is used rather than :func:`jax.scipy.optimize.minimize`
because the latter aborts the whole optimisation when a line search fails to
satisfy the Wolfe conditions, and a retrieval objective that spans many orders
of magnitude between its parameters fails those conditions easily; optax keeps
the best step it found and carries on.
"""

import time
import typing as t

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import optax

from taurex.core.priors import Gaussian
from taurex.core.priors import Prior
from taurex.core.priors import Uniform
from taurex.model import ForwardModel
from taurex.optimizer.optimizer import Optimizer
from taurex.spectrum import BaseSpectrum

from .model import Atmosphere


Array = jax.Array


class PriorTransform:
    """Cube to parameter map built from a taurex prior.

    Mirrors :meth:`taurex.optimizer.optimizer.Optimizer.prior_transform`, which
    evaluates the inverse cumulative distribution of the prior, but in JAX so
    that a gradient can flow from the parameters back to the cube.

    Parameters
    ----------
    prior:
        The taurex prior of a fitting parameter

    name:
        Parameter name, used in error messages

    dtype:
        Floating point dtype of the working arrays

    """

    def __init__(
        self,
        prior: Prior,
        name: str,
        dtype: jnp.dtype,
    ) -> None:
        """Initialise the transform.

        Parameters
        ----------
        prior:
            The taurex prior of a fitting parameter

        name:
            Parameter name, used in error messages

        dtype:
            Floating point dtype of the working arrays

        """
        self.name = name
        self.dtype = dtype

        # The kind is decided by the public class rather than by the presence
        # of a private attribute, and the uniform bounds come from the public
        # accessor, so a future prior that changes its internals does not
        # silently take the wrong branch here. LogUniform and LogGaussian are
        # subclasses and are handled by the same two cases: the parameter is
        # already in the prior's own space, which is log space for a log prior.
        if isinstance(prior, Uniform):
            self._kind = "uniform"
            low, high = prior.boundaries()
            self._low = float(low)
            self._high = float(high)
            self.bounds = (min(self._low, self._high), max(self._low, self._high))
        elif isinstance(prior, Gaussian):
            self._kind = "gaussian"
            self._loc = float(prior._loc)
            self._scale = float(prior._scale)
            self.bounds = None
        else:
            raise NotImplementedError(
                f"Cannot differentiate through the prior {type(prior).__name__} "
                f"of {name}. Uniform, LogUniform, Gaussian and LogGaussian are "
                "supported."
            )

    def forward(self, cube: Array) -> Array:
        """Map a cube coordinate onto the parameter.

        Parameters
        ----------
        cube:
            Value in ``(0, 1)``

        Returns
        -------
        :obj:`jax.Array`
            Parameter value in the space the optimizers use, which is the space
            the prior is defined in

        """
        if self._kind == "uniform":
            return self._low + cube * (self._high - self._low)
        # The inverse normal CDF diverges at the edges of the cube, so the
        # argument is pulled just inside them. The clip sits at about five and
        # a half sigma, far enough out that it does not change a usable prior.
        root_two = np.sqrt(2.0)
        argument = jnp.clip(2.0 * cube - 1.0, -1.0 + 1e-12, 1.0 - 1e-12)
        return self._loc + self._scale * root_two * jax.scipy.special.erfinv(argument)

    def inverse(self, theta: float) -> float:
        """Cube coordinate of a parameter value.

        Parameters
        ----------
        theta:
            Parameter in prior space

        Returns
        -------
        float
            Cube coordinate in ``(0, 1)``

        """
        if self._kind == "uniform":
            return float((theta - self._low) / (self._high - self._low))
        from scipy.special import ndtr

        return float(ndtr((theta - self._loc) / self._scale))

    def log_prior(self, theta: Array) -> Array:
        """Log density of the prior in parameter space.

        A flat prior contributes a constant, which is dropped: it shifts the
        posterior but neither its mode nor its curvature.

        Parameters
        ----------
        theta:
            Parameter value, possibly carrying a gradient

        Returns
        -------
        :obj:`jax.Array`
            Log prior density, up to an additive constant

        """
        if self._kind == "uniform":
            return jnp.zeros((), dtype=self.dtype)
        return -0.5 * ((theta - self._loc) / self._scale) ** 2

    def log_abs_derivative(self, cube: Array) -> Array:
        """Log of ``|d theta / d u|`` for the cube-to-parameter map.

        The maximum a posteriori fit deliberately drops this Jacobian so that
        its optimum is the chi-squared minimum, but a sampler must keep it: the
        density in unconstrained space is the density in parameter space times
        this factor. Dropping it biases the draws towards the bounds, and it is
        also what gives the unconstrained tails their slope, so without it a
        Hamiltonian trajectory in the saturated part of the sigmoid has no
        force to turn it around.

        Parameters
        ----------
        cube:
            Value in ``(0, 1)``, before the sigmoid is applied

        Returns
        -------
        :obj:`jax.Array`
            ``log|d theta / d u|``

        """
        # Every transform has d c / d u = c (1 - c) from the sigmoid.
        common = jnp.log(cube) + jnp.log1p(-cube)
        if self._kind == "uniform":
            return jnp.log(self._high - self._low) + common
        argument = jnp.clip(2.0 * cube - 1.0, -1.0 + 1e-12, 1.0 - 1e-12)
        standard = np.sqrt(2.0) * jax.scipy.special.erfinv(argument)
        return jnp.log(self._scale * np.sqrt(2.0 * np.pi)) + 0.5 * standard**2 + common


class LaplaceOptimizer(Optimizer):
    """Retrieval by maximum a posteriori fit plus a Laplace approximation."""

    def __init__(
        self,
        observed: t.Optional[BaseSpectrum] = None,
        model: t.Optional[ForwardModel] = None,
        num_samples: t.Optional[int] = 2000,
        max_iterations: t.Optional[int] = 200,
        max_lbfgs_iterations: t.Optional[int] = 40,
        memory_size: t.Optional[int] = 10,
        scale_init_precond: t.Optional[bool] = True,
        gradient_tolerance: t.Optional[float] = 1e-10,
        eigenvalue_floor: t.Optional[float] = 1e-3,
        seed: t.Optional[int] = 0,
        device: t.Optional[str] = None,
        sigma_fraction: t.Optional[float] = 0.02,
        sampler: t.Optional[str] = "laplace",
        n_starts: t.Optional[int] = 1,
        start_scale: t.Optional[float] = 0.5,
        num_warmup: t.Optional[int] = 500,
        num_chains: t.Optional[int] = 4,
        jitter: t.Optional[bool] = False,
        jitter_scale: t.Optional[float] = None,
        **kwargs: t.Any,
    ) -> None:
        """Initialise the optimizer.

        Parameters
        ----------
        observed:
            Observation to fit to, supplied by the input file driver

        model:
            Forward model to fit, supplied by the input file driver

        num_samples:
            Number of posterior samples drawn from the Laplace approximation

        max_iterations:
            Maximum number of quasi-Newton optimiser steps

        max_lbfgs_iterations:
            Maximum number of line search evaluations allowed per optimiser
            step.

        memory_size:
            Number of curvature pairs the L-BFGS keeps. Ten pairs is the usual
            choice and is ample for the handful of parameters a retrieval
            fits; setting it to zero turns the method into steepest descent
            with a line search.

        scale_init_precond:
            Scale the initial inverse Hessian estimate by the reciprocal of the
            gradient magnitude. This caps the first step at a unit Euclidean
            ball, which stops the line search from wasting its budget on a step
            that the wildly different scales of the parameters make hopeless.
            It is on by default and is what makes the fit converge from the
            parameter values in the input file without any hand tuning.

        gradient_tolerance:
            Convergence tolerance on the gradient norm. Retrieval objectives
            are flat enough that this is usually not reached before
            ``max_iterations``; the fit then stops there and says so.

        eigenvalue_floor:
            Ridge added to the scaled Hessian before inverting it. It bounds
            the variance of the directions the data does not constrain, so
            those parameters come back prior dominated rather than undefined.

        seed:
            Seed for the posterior draws, so a retrieval is reproducible

        device:
            JAX platform to run on, for example ``cpu``. By default JAX uses
            its own default device, which is the GPU when one is visible.

        sigma_fraction:
            Fraction of the posterior samples reused when the post-processing
            translates them into profile uncertainties. The Laplace draws are
            already independent samples of the posterior, so this can be far
            smaller than the chain thinning the sampling optimizers need. It
            is kept only for the fallback numpy post-processing: the JAX
            optimizer computes the profile bands from all the draws with
            :func:`jax.vmap` in one compiled call.

        sampler:
            ``laplace`` (default) describes the posterior with a Gaussian
            centred on the mode, which is fast and needs no tuning but cannot
            represent curvature or multimodality and gives no evidence.
            ``nuts`` runs a No-U-Turn Sampler (via :mod:`blackjax`) started at
            the mode, with the Laplace covariance as its initial inverse mass
            matrix, and returns exact, equally weighted posterior draws. Use
            ``nuts`` when the posterior is not Gaussian or when the error bars
            from the Laplace approximation look suspect.

        n_starts:
            Number of independent starting points for the mode search. Each
            start perturbs the unconstrained vector by ``start_scale`` in
            logit space and the fits are run together with :func:`jax.vmap`,
            so a multi-start costs little more than a single start and escapes
            the local minima a flat retrieval objective is prone to. The
            point with the best posterior is kept.

        start_scale:
            Standard deviation, in unconstrained logit space, of the
            perturbation applied to the extra starting points. One is a
            sensible default; the base point itself is always included.

        num_warmup:
            Warm-up steps for the NUTS window adaptation, per chain.

        num_chains:
            Number of NUTS chains. The posterior draws are split evenly across
            them, so a chain run also gives a crude convergence check.

        jitter:
            When True an extra nuisance coordinate is fitted: the per-bin
            errors become ``sqrt(error^2 + jitter^2)``. This is how a
            retrieval absorbs under-estimated error bars, and it is only
            tractable because the error bar is a traced argument of the
            likelihood.

        jitter_scale:
            Scale of the half-normal prior on the jitter, in the units of the
            observed spectrum. ``None`` uses the mean of the input error bars,
            which puts the prior at the right order of magnitude for a
            typical retrieval.

        kwargs:
            Extra keyword arguments are ignored, so an input file written for a
            sampling optimizer still loads.

        """
        super().__init__(
            "Jax", observed=observed, model=model, sigma_fraction=sigma_fraction
        )
        self.num_samples = int(num_samples)
        self.max_iterations = int(max_iterations)
        self.max_lbfgs_iterations = int(max_lbfgs_iterations)
        self.memory_size = int(memory_size)
        self.scale_init_precond = bool(scale_init_precond)
        self.gradient_tolerance = float(gradient_tolerance)
        self.eigenvalue_floor = float(eigenvalue_floor)
        self.seed = seed
        self.device = None if device is None else jax.devices(device)[0]
        self.ignored_options = kwargs

        self.sampler = str(sampler).lower()
        if self.sampler not in ("laplace", "nuts"):
            raise ValueError(f"Unknown sampler {sampler!r}; use 'laplace' or 'nuts'")
        self.n_starts = max(1, int(n_starts))
        self.start_scale = float(start_scale)
        self.num_warmup = int(num_warmup)
        self.num_chains = max(1, int(num_chains))
        self.jitter = bool(jitter)
        self.jitter_scale = None if jitter_scale is None else float(jitter_scale)

        self.atmosphere: t.Optional[Atmosphere] = None
        self.transforms: t.List[PriorTransform] = []
        self._samples: t.Optional[npt.NDArray[np.float64]] = None
        self._map: t.Optional[npt.NDArray[np.float64]] = None
        self._median: t.Optional[npt.NDArray[np.float64]] = None
        self._covariance: t.Optional[npt.NDArray[np.float64]] = None
        self._u_covariance: t.Optional[Array] = None
        self._log_posterior: t.Optional[float] = None
        self._log_likelihood: t.Optional[float] = None
        self._fisher: t.Optional[npt.NDArray[np.float64]] = None
        self._jitter_map: t.Optional[float] = None
        self._n_physical: int = 0
        self._nuts_parameters: t.Optional[t.Dict[str, t.Any]] = None

        self.iterations: t.Optional[int] = None
        self.function_evaluations: t.Optional[int] = None
        self.gradient_evaluations: t.Optional[int] = None
        self.compile_time: t.Optional[float] = None
        self.fit_time: t.Optional[float] = None
        self.laplace_time: t.Optional[float] = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def compute_fit(self) -> None:  # noqa: C901
        """Maximise the posterior and describe it around the mode.

        A quasi-Newton fit locates the mode; what happens next depends on
        :attr:`sampler`. ``laplace`` describes the posterior with the Gaussian
        the Hessian implies, ``nuts`` draws exact samples from it with the
        Laplace covariance as the initial mass matrix.
        """
        fit_params = self.fitting_parameters
        if not fit_params:
            raise ValueError(
                "No fitting parameters were enabled, so there is nothing to "
                "fit. Add entries under [Fitting] in the input file."
            )

        dtype = jnp.float64
        atmosphere = Atmosphere(
            self._model,
            self._observed,
            fit_params=fit_params,
            dtype=dtype,
        )
        self.atmosphere = atmosphere
        self._n_physical = len(fit_params)

        transforms = [
            PriorTransform(param.fit_prior, param.name, dtype) for param in fit_params
        ]
        self.transforms = transforms

        # The jitter prior is defined in the units of the data, so its default
        # scale is the mean input error: the size of a mistake the input file
        # would make if it under-reported its error bars.
        if self.jitter and self.jitter_scale is None:
            self.jitter_scale = float(
                np.mean(np.abs(np.asarray(atmosphere.errors, dtype=float)))
            )
        jitter = bool(self.jitter)

        def parameters_of(u: Array) -> Array:
            """Map the unconstrained vector onto the fitting parameters.

            Works for a single vector and for a batch of them: the parameters
            are always stacked along the last axis. When a jitter is fitted it
            is the last column, kept positive by an exponential.
            """
            cube = jax.nn.sigmoid(u)
            columns = [
                transform.forward(cube[..., index])
                for index, transform in enumerate(transforms)
            ]
            if jitter:
                columns.append(jnp.exp(u[..., len(transforms)]) * self.jitter_scale)
            return jnp.stack(columns, axis=-1)

        def negative_log_posterior(u: Array, data: Array, error: Array) -> Array:
            """Negative log posterior as a function of ``u``.

            The prior is treated as a density over the parameters themselves,
            so a flat prior contributes nothing and the mode is the chi-squared
            minimum. Writing the objective this way, rather than as the density
            of ``u``, is what keeps the mode and the covariance in the same
            space as the reported parameters.

            ``data`` and ``error`` are arguments rather than closed-over
            attributes so the objective is a pure function of the observation:
            a compiled program can be evaluated on a new dataset, a noise
            injection or a jittered error bar without retracing.
            """
            theta = parameters_of(u)
            log_prior = sum(
                transform.log_prior(theta[..., index])
                for index, transform in enumerate(transforms)
            )
            if jitter:
                size = theta[..., len(transforms)]
                # Half-normal prior on the jitter, up to an additive constant.
                log_prior = log_prior - 0.5 * (size / self.jitter_scale) ** 2
                error = jnp.sqrt(error**2 + size**2)
            physical = theta[..., : self._n_physical]
            return (
                -atmosphere.log_likelihood(physical, data=data, error=error) - log_prior
            )

        data = atmosphere.observation
        error = atmosphere.errors

        def objective(u: Array) -> Array:
            """Negative log posterior at the fitted observation."""
            return negative_log_posterior(u, data, error)

        def log_transform_jacobian(u: Array) -> Array:
            """Log of ``|d theta / d u|`` summed over the fitted parameters."""
            cube = jax.nn.sigmoid(u)
            total = sum(
                transform.log_abs_derivative(cube[..., index])
                for index, transform in enumerate(transforms)
            )
            if jitter:
                # theta_jitter = scale * exp(u_jitter), so its derivative is the
                # jitter itself.
                total = total + u[..., len(transforms)] + jnp.log(self.jitter_scale)
            return total

        def nuts_objective(u: Array) -> Array:
            """Target density in unconstrained space, transform Jacobian included.

            The mode search uses :func:`objective`, which drops the Jacobian so
            that the optimum is the chi-squared minimum. A sampler cannot: the
            density of ``u`` is the density of the parameters times
            ``|d theta / d u|``, and leaving it out both biases the draws
            towards the bounds and removes the slope the tails need.
            """
            return objective(u) - log_transform_jacobian(u)

        u = self._starting_point(transforms, dtype)
        if jitter:
            # A tenth of the prior scale: away from the jitter = 0 boundary the
            # exponential cannot reach, but small enough not to bias the fit.
            u = jnp.concatenate([u, jnp.array([np.log(0.1)], dtype=dtype)])

        def run(u0: Array) -> t.Tuple[Array, Array, Array, Array, Array]:
            """Run the whole fit as one compiled program.

            The L-BFGS step, its line search and the stopping test all live
            inside the loop, so the optimiser costs a device round trip per
            step rather than a python callback per step.

            Parameters
            ----------
            u0:
                Unconstrained starting point

            Returns
            -------
            u_map:
                Unconstrained optimum

            gradient_norm:
                Infinity norm of the gradient at the last step it was checked

            steps:
                Number of optimiser steps taken

            line_search_steps:
                Number of trial points the line search evaluated while doing so

            """
            solver = optax.lbfgs(
                learning_rate=1.0,
                memory_size=self.memory_size,
                scale_init_precond=self.scale_init_precond,
                linesearch=optax.scale_by_zoom_linesearch(
                    max_linesearch_steps=self.max_lbfgs_iterations
                ),
            )
            # The line search asks only for the objective at each trial point
            # and gets the directional derivative by linearising it, so a trial
            # costs a forward pass rather than a forward and a backward one.
            # This reuses the value and gradient stored at the last accepted
            # point instead of recomputing them.
            value_and_grad = optax.value_and_grad_from_state(objective)
            zero = jnp.zeros((), dtype=jnp.int32)

            def condition(carry):
                _, _, gradient_norm, steps, _ = carry
                return (gradient_norm > self.gradient_tolerance) & (
                    steps < self.max_iterations
                )

            def body(carry):
                u_current, state, gradient_norm, steps, trials = carry
                value, gradient = value_and_grad(u_current, state=state)
                updates, new_state = solver.update(
                    gradient,
                    state,
                    u_current,
                    value=value,
                    grad=gradient,
                    value_fn=objective,
                )
                u_next = optax.apply_updates(u_current, updates)
                taken = jnp.asarray(
                    optax.tree.get(state, "num_linesearch_steps"), dtype=jnp.int32
                )
                new_norm = jnp.linalg.norm(gradient, ord=jnp.inf)
                # A start that has already converged is frozen. Under vmap the
                # batched loop stops only once the slowest start is done, and
                # without this the extra steps would keep moving the starts that
                # finished first away from their own optimum.
                active = (gradient_norm > self.gradient_tolerance) & (
                    steps < self.max_iterations
                )
                u_next = jnp.where(active, u_next, u_current)
                new_state = jax.tree.map(
                    lambda new, old: (
                        jnp.where(active, new, old)
                        if isinstance(new, jax.Array)
                        else new
                    ),
                    new_state,
                    state,
                )
                return (
                    u_next,
                    new_state,
                    new_norm,
                    steps + 1,
                    trials + taken,
                )

            # The initial gradient norm is infinite so that the fit always
            # takes at least one step, whatever the starting point.
            return jax.lax.while_loop(
                condition,
                body,
                (u0, solver.init(u0), jnp.array(jnp.inf, dtype=dtype), zero, zero),
            )

        def run_starts(starts: Array) -> t.Tuple[Array, Array, Array, Array, Array]:
            """Fit every starting point and keep the best posterior.

            :func:`jax.lax.map` runs the starts one after another inside the
            compiled program. A :func:`jax.vmap` would batch them, but a
            batched ``while_loop`` stops only when the slowest start has
            finished and its extra iterations are not worth the risk of moving
            a converged start off its optimum, so each start gets its own loop.
            """
            u_maps, _, gradient_norms, steps, trials = jax.lax.map(run, starts)
            scores = jax.lax.map(objective, u_maps)
            best = jnp.argmin(scores)
            return (
                u_maps[best],
                gradient_norms[best],
                steps[best],
                trials[best],
                scores[best],
            )

        if self.n_starts > 1:
            key = jax.random.PRNGKey(0 if self.seed is None else int(self.seed))
            noise = jax.random.normal(key, (self.n_starts, u.shape[0]), dtype=dtype)
            # The first start is always the point the input file holds.
            noise = noise.at[0].set(0.0)
            starts = u[None, :] + self.start_scale * noise
        else:
            starts = u[None, :]

        hessian_of = jax.jit(jax.hessian(objective), device=self.device)
        transform_of = jax.jit(parameters_of, device=self.device)
        jacobian_of = jax.jit(jax.jacobian(parameters_of), device=self.device)

        def log_like_of(theta: Array, obs: Array, err: Array) -> Array:
            """Log likelihood with the observation as an explicit argument."""
            return atmosphere.log_likelihood(theta, data=obs, error=err)

        log_likelihood_of = jax.jit(log_like_of, device=self.device)
        solve = jax.jit(run_starts, device=self.device)

        # Tracing and compiling is a one off cost that no eager implementation
        # pays, so it is measured separately and reported: it is paid once per
        # fit here and once per process if the compiled functions stayed alive.
        # Every shape the fit will use is compiled now, so that the times below
        # are execution times rather than hidden compilation.
        draws = jnp.zeros((self.num_samples, u.shape[0]), dtype=dtype)
        started = time.perf_counter()
        solve.lower(starts).compile()
        hessian_of.lower(u).compile()
        transform_of.lower(u).compile()
        transform_of.lower(draws).compile()
        jacobian_of.lower(u).compile()
        log_likelihood_of.lower(u[: self._n_physical], data, error).compile()
        self.compile_time = time.perf_counter() - started

        started = time.perf_counter()
        # JAX dispatches asynchronously, so the result is waited on here: the
        # optimiser loop runs on the device and the wall clock only measures it
        # once something asks for the value.
        u_map, gradient_norm, steps, trials, _ = jax.block_until_ready(solve(starts))
        self.fit_time = time.perf_counter() - started

        self.iterations = int(steps)
        self.function_evaluations = int(steps + trials)
        self.gradient_evaluations = int(steps)
        if self.iterations >= self.max_iterations and (
            float(gradient_norm) > self.gradient_tolerance
        ):
            # The objective of a retrieval is flat enough that most fits stop
            # here rather than on the gradient tolerance. Saying so is the
            # difference between "this is the mode" and "this is as far as the
            # fit got"; the point returned is the best one it found.
            self.warning(
                "The quasi-Newton fit reached its iteration limit after %d "
                "steps with gradient norm %.3e, above the tolerance %.3e",
                self.iterations,
                float(gradient_norm),
                self.gradient_tolerance,
            )

        theta_full = transform_of(u_map)
        self._map = np.asarray(theta_full[: self._n_physical]).copy()
        if jitter:
            self._jitter_map = float(theta_full[self._n_physical])
            error_effective = jnp.sqrt(error**2 + theta_full[self._n_physical] ** 2)
        else:
            self._jitter_map = None
            error_effective = error
        self._log_likelihood = float(
            log_likelihood_of(theta_full[: self._n_physical], data, error_effective)
        )
        self._log_posterior = float(-objective(u_map))

        started = time.perf_counter()
        if self.sampler == "nuts":
            # The mass matrix has to describe the same density the sampler
            # targets, so the Hessian here includes the transform Jacobian.
            nuts_hessian_of = jax.jit(jax.hessian(nuts_objective), device=self.device)
            covariance, variance_scale = self._laplace_covariance(
                nuts_hessian_of, u_map
            )
        else:
            covariance, variance_scale = self._laplace_covariance(hessian_of, u_map)
        self._u_covariance = covariance * variance_scale
        if self.sampler == "nuts":
            self._samples = self._run_nuts(
                nuts_objective, u_map, covariance, variance_scale, transform_of
            )
            self._median = np.median(self._samples, axis=0)
            if self._samples.shape[0] > 1:
                self._covariance = np.atleast_2d(np.cov(self._samples, rowvar=False))
            else:
                self._covariance = np.zeros(
                    (self._n_physical, self._n_physical), dtype=float
                )
        else:
            draws_full = self._draw_samples(
                u_map, covariance, variance_scale, transform_of
            )
            draws_full = np.asarray(draws_full)
            self._samples = draws_full[:, : self._n_physical].copy()
            self._median = np.median(self._samples, axis=0)
            full_covariance = self._parameter_covariance(
                covariance, variance_scale, jacobian_of, u_map
            )
            self._covariance = np.asarray(full_covariance)[
                : self._n_physical, : self._n_physical
            ].copy()
        self.laplace_time = time.perf_counter() - started

    def _run_nuts(
        self,
        objective: t.Callable[[Array], Array],
        u_map: Array,
        covariance: Array,
        variance_scale: float,
        transform_of: t.Callable[[Array], Array],
    ) -> npt.NDArray[np.float64]:
        """Draw exact posterior samples with the No-U-Turn Sampler.

        The mode found by L-BFGS is the starting point and the Laplace
        covariance is the initial inverse mass matrix, so the sampler starts
        already scaled to the posterior and the window adaptation has little
        left to do. The Hessian that the Laplace approximation needs is
        computed once and then reused here, which is the point of keeping both
        descriptions of the posterior in the same object.

        Parameters
        ----------
        objective:
            Negative log posterior, a pure function of the unconstrained vector

        u_map:
            Location of the mode

        covariance:
            Covariance of the scaled unconstrained space

        variance_scale:
            Factor that turns it into the covariance of ``u``

        transform_of:
            Compiled map from ``u`` to the parameters

        Returns
        -------
        :obj:`numpy.ndarray`
            Posterior draws in parameter space, shape ``(nsamples, nparams)``

        """
        import blackjax

        dimension = u_map.shape[0]
        covariance_u = covariance * variance_scale
        # The sampler runs in coordinates whitened by the Laplace covariance:
        # ``u = u_map + factor @ v`` with ``factor`` the Cholesky factor of the
        # Laplace covariance. Whatever the parameter scales, the posterior is
        # then close to a standard normal, so an identity mass matrix is the
        # right one, a single step size works, and no adaptation is needed to
        # discover the geometry. Using the covariance as a preconditioner this
        # way is the well conditioned form of the "mass matrix from the
        # Hessian" idea; passing the raw covariance to an adaptive sampler is
        # not, because a flat direction makes it enormous.
        factor = jnp.linalg.cholesky(covariance_u)

        def logdensity_v(v: Array) -> Array:
            # ``objective`` is the negative log posterior; blackjax wants the
            # log density, so the sign is flipped here. Getting this wrong is
            # not subtle: the sampler then seeks the least probable region and
            # parks the chain on a bound.
            return -objective(u_map + factor @ v)

        key = jax.random.PRNGKey(0 if self.seed is None else int(self.seed))
        key, warmup_key, sample_key = jax.random.split(key, 3)

        # In the whitened space the target is close to a standard normal, so a
        # short adaptation only has to fine tune the step size rather than
        # discover a wildly anisotropic geometry. This is where the Laplace
        # covariance pays off a second time.
        warmup = blackjax.window_adaptation(
            blackjax.nuts, logdensity_v, is_mass_matrix_diagonal=True
        )
        (state, parameters), _ = warmup.run(
            warmup_key,
            jnp.zeros(dimension, dtype=covariance_u.dtype),
            num_steps=self.num_warmup,
        )
        self._nuts_parameters = {
            "step_size": float(parameters["step_size"]),
            "inverse_mass_matrix": np.asarray(parameters["inverse_mass_matrix"]).copy(),
        }
        algorithm = blackjax.nuts(logdensity_v, **parameters)

        steps_per_chain = max(1, self.num_samples // self.num_chains)
        chain_keys = jax.random.split(sample_key, self.num_chains)
        step_keys = jax.vmap(lambda k: jax.random.split(k, steps_per_chain))(chain_keys)

        def one_chain(keys: Array) -> Array:
            def one_step(current, step_key):
                current, _ = algorithm.step(step_key, current)
                return current, current.position

            return jax.lax.scan(one_step, state, keys)[1]

        positions_v = jax.vmap(one_chain)(step_keys).reshape(-1, dimension)
        positions = u_map[None, :] + jax.block_until_ready(positions_v) @ factor.T
        theta = transform_of(positions)
        return np.asarray(theta[:, : self._n_physical]).copy()

    def generate_profiles(
        self,
        solution: int,
        binning: npt.NDArray[np.float64],
    ) -> t.Tuple[
        t.Dict[str, npt.NDArray[np.float64]],
        t.Dict[str, npt.NDArray[np.float64]],
    ]:
        """Profile and spectrum uncertainties from the posterior draws.

        The base implementation walks the samples through the numpy model and
        accumulates an online variance, which the report identified as the
        dominant cost of a run. The draws are already a batch, so they are
        evaluated in one :func:`jax.vmap`'d, compiled call over the
        differentiable model instead, and the variances come straight out of
        the batch.

        Parameters
        ----------
        solution:
            Solution index, only 0 exists

        binning:
            Binning wavenumber grid; kept for interface compatibility, the
            binning operator is already built into the atmosphere

        Returns
        -------
        t.Tuple
            Profile and spectrum error dictionaries

        """
        if self._samples is None or self.atmosphere is None:
            return super().generate_profiles(solution, binning)

        atmosphere = self.atmosphere
        theta = jnp.asarray(self._samples, dtype=jnp.float64)
        profile_of = jax.jit(jax.vmap(atmosphere.profile_state), device=self.device)
        temperature, active, inactive, depth = jax.block_until_ready(profile_of(theta))

        profile_dict = {
            "temp_profile_std": np.asarray(jnp.std(temperature, axis=0)),
            "active_mix_profile_std": np.asarray(jnp.std(active, axis=0)),
            "inactive_mix_profile_std": np.asarray(jnp.std(inactive, axis=0)),
        }
        spectrum_dict = {"native_std": np.asarray(jnp.std(depth, axis=0))}
        if atmosphere.binning_matrix is not None:
            binned = jax.vmap(lambda value: atmosphere.binning_matrix @ value)(depth)
            spectrum_dict["binned_std"] = np.asarray(jnp.std(binned, axis=0))
        return profile_dict, spectrum_dict

    def fisher_information(
        self,
        theta: t.Optional[npt.NDArray[np.float64]] = None,
        data: t.Optional[Array] = None,
        error: t.Optional[Array] = None,
    ) -> npt.NDArray[np.float64]:
        """Fisher information matrix in parameter space.

        ``J^T W J`` with ``J`` the jacobian of the binned spectrum and ``W``
        the inverse variance. Its inverse is the Cramer-Rao bound, so it says
        how well each parameter can ever be measured with this observation, and
        its eigenvectors expose the degenerate directions a retrieval will
        struggle with. Both jacobians are one compiled call, so the diagnostic
        is essentially free once a fit has run.

        Parameters
        ----------
        theta:
            Parameter values to evaluate at; the MAP when None

        data:
            Observation to evaluate against; the fitted one when None

        error:
            Per-bin uncertainty; the fitted one when None

        Returns
        -------
        :obj:`numpy.ndarray`
            Matrix of shape ``(nparams, nparams)``

        """
        if self.atmosphere is None:
            raise ValueError(
                "compute_fit must be run before asking for the Fisher information"
            )
        atmosphere = self.atmosphere
        if theta is None:
            theta = self._map
        if data is None:
            data = atmosphere.observation
        if error is None:
            error = atmosphere.errors

        jacobian_of = jax.jit(jax.jacobian(atmosphere.spectrum), device=self.device)
        jacobian = np.asarray(jacobian_of(jnp.asarray(theta, dtype=jnp.float64)))
        scaled = jacobian / np.asarray(error, dtype=float)[:, None]
        self._fisher = scaled.T @ scaled
        return self._fisher

    @property
    def jitter_map(self) -> t.Optional[float]:
        """Fitted jitter at the mode, or None when no jitter was fitted."""
        return self._jitter_map

    @property
    def nuts_parameters(self) -> t.Optional[t.Dict[str, t.Any]]:
        """Step size and inverse mass matrix the NUTS adaptation settled on."""
        return self._nuts_parameters

    @property
    def fisher(self) -> t.Optional[npt.NDArray[np.float64]]:
        """Cached Fisher information, filled in by :meth:`fisher_information`."""
        return self._fisher

    def _laplace_covariance(
        self,
        hessian_of: t.Callable[[Array], Array],
        u_map: Array,
    ) -> t.Tuple[Array, float]:
        """Gaussian approximation of the posterior, in unconstrained space.

        The Hessian is taken in the unconstrained space because that is where
        the posterior is well conditioned: the sigmoid saturation that makes
        the parameter space awkward is exactly what keeps ``u`` of order one.

        A retrieval Hessian still spans many orders of magnitude across its
        eigenvalues, and its numerically null directions are what break a plain
        inversion. Everything is therefore divided by the largest entry of the
        Hessian, which bounds the matrix, and a ridge of ``eigenvalue_floor`` is
        added on top. The ridge caps the variance of the directions the data
        does not constrain, so those parameters come back prior dominated
        instead of undefined; directions with real curvature sit orders of
        magnitude above it and are unaffected.

        The Hessian itself is taken with :func:`jax.hessian`, which is
        forward-over-reverse: one reverse pass per parameter, rather than the
        reverse-over-reverse nesting a jacobian of a gradient would cost.

        Parameters
        ----------
        hessian_of:
            Compiled Hessian of the negative log posterior

        u_map:
            Location of the optimum

        Returns
        -------
        covariance:
            Covariance of the scaled Hessian's inverse

        variance_scale:
            Factor that turns that covariance into the covariance of ``u``

        """
        hessian = hessian_of(u_map)
        hessian = 0.5 * (hessian + hessian.T)

        largest = float(jnp.max(jnp.abs(hessian)))
        if largest == 0.0 or not np.isfinite(largest):
            # A flat posterior is not an error: there is simply nothing for the
            # data to constrain, so every sample is the optimum itself.
            size = hessian.shape[0]
            return jnp.eye(size, dtype=hessian.dtype), 0.0

        scaled = hessian / largest
        diagonal = jnp.diagonal(scaled)
        # The regularisation is chosen so that the scaled matrix is strictly
        # diagonally dominant, which by Gershgorin's theorem makes it positive
        # definite whatever the numerical noise in the Hessian did to its
        # spectrum. Without it a flat or rank deficient Hessian produces a
        # matrix that cannot be factored at all.
        off_diagonal = jnp.sum(jnp.abs(scaled), axis=1) - jnp.abs(diagonal)
        shift = off_diagonal + jnp.clip(-diagonal, 0.0, None) + self.eigenvalue_floor
        regularized = scaled + jnp.diag(shift)

        covariance = jnp.linalg.inv(regularized)
        covariance = 0.5 * (covariance + covariance.T)
        return covariance, 1.0 / largest

    def _parameter_covariance(
        self,
        covariance: Array,
        variance_scale: float,
        jacobian_of: t.Callable[[Array], Array],
        u_map: Array,
    ) -> npt.NDArray[np.float64]:
        """Carry the unconstrained covariance back to parameter space.

        Parameters
        ----------
        covariance:
            Covariance in the scaled unconstrained space

        variance_scale:
            Factor that turns it into the covariance of ``u``

        jacobian_of:
            Compiled jacobian of the map from ``u`` to the parameters

        u_map:
            Location of the optimum

        Returns
        -------
        :obj:`numpy.ndarray`
            Covariance in parameter space

        """
        jacobian = jacobian_of(u_map)
        full = jacobian @ (covariance * variance_scale) @ jacobian.T
        full = 0.5 * (full + full.T)
        return np.asarray(full).copy()

    def _draw_samples(
        self,
        u_map: Array,
        covariance: Array,
        variance_scale: float,
        transform_of: t.Callable[[Array], Array],
    ) -> npt.NDArray[np.float64]:
        """Draw posterior samples from the Gaussian approximation.

        The draws are made in unconstrained space and pushed through the prior
        transforms, so every sample lands inside the support of its prior
        however wide the Gaussian is. The Cholesky factor is taken in the scaled
        space for conditioning and applied to the samples only at the end,
        which avoids ever forming the ill-conditioned covariance of ``u``
        itself.

        Parameters
        ----------
        u_map:
            Unconstrained vector at the optimum

        covariance:
            Covariance in the scaled unconstrained space

        variance_scale:
            Factor that turns it into the covariance of ``u``

        transform_of:
            Compiled map from ``u`` to the parameters

        Returns
        -------
        :obj:`numpy.ndarray`
            Samples with shape ``(num_samples, nparams)``

        """
        key = jax.random.PRNGKey(0 if self.seed is None else int(self.seed))
        factor = jnp.linalg.cholesky(covariance)
        normal = jax.random.normal(
            key, (self.num_samples, u_map.shape[0]), dtype=covariance.dtype
        )
        offset = normal @ factor.T * float(np.sqrt(variance_scale))
        draws = u_map[None, :] + offset
        # One traced call for all the samples: the transform is a handful of
        # scalar operations per parameter, so a python loop over two thousand
        # draws would be pure dispatch overhead.
        theta = transform_of(draws)
        return np.asarray(theta).copy()

    def _starting_point(
        self, transforms: t.Sequence[PriorTransform], dtype: jnp.dtype
    ) -> Array:
        """Unconstrained vector that reproduces the current parameter values.

        Parameters
        ----------
        transforms:
            Prior transforms of the fitted parameters

        dtype:
            Floating point dtype to use

        Returns
        -------
        :obj:`jax.Array`
            Unconstrained vector

        """
        cube = []
        for param, transform in zip(self.fitting_parameters, transforms, strict=True):
            value = float(param.fit_value)
            cube.append(min(max(transform.inverse(value), 1e-6), 1.0 - 1e-6))
        cube_array = jnp.asarray(cube, dtype=dtype)
        return jax.scipy.special.logit(cube_array)

    # ------------------------------------------------------------------
    # Sampler interface
    # ------------------------------------------------------------------
    def get_solution(self):
        """Yield the MAP and median parameter vectors.

        Yields
        ------
        solution:
            Solution index

        map:
            Parameter values at the maximum a posteriori point

        median:
            Median of the Laplace samples

        extra:
            Empty; a Laplace approximation produces no extra outputs

        """
        yield 0, self._map, self._median, []

    def get_samples(self, solution_id: int) -> npt.NDArray[np.float64]:
        """Posterior samples drawn from the Laplace approximation.

        Parameters
        ----------
        solution_id:
            Solution index, only 0 exists

        Returns
        -------
        :obj:`numpy.ndarray`
            Samples in the same parameter space as the MAP

        """
        if self._samples is None:
            raise ValueError("compute_fit must be run before asking for samples")
        return self._samples

    def get_weights(self, solution_id: int) -> npt.NDArray[np.float64]:
        """Weights of the posterior samples.

        The Laplace draws are already samples from the posterior, so they carry
        equal weight.

        Parameters
        ----------
        solution_id:
            Solution index, only 0 exists

        Returns
        -------
        :obj:`numpy.ndarray`
            Uniform weights

        """
        samples = self.get_samples(solution_id)
        return np.full(samples.shape[0], 1.0 / samples.shape[0])

    @property
    def covariance(self) -> t.Optional[npt.NDArray[np.float64]]:
        """Covariance of the Laplace approximation in parameter space."""
        return self._covariance

    @property
    def log_posterior(self) -> t.Optional[float]:
        """Log posterior at the maximum a posteriori point."""
        return self._log_posterior

    @property
    def log_likelihood_map(self) -> t.Optional[float]:
        """Log likelihood at the maximum a posteriori point."""
        return self._log_likelihood

    @classmethod
    def input_keywords(cls) -> t.Tuple[str, ...]:
        """Return the input file keywords for this optimizer."""
        return ("jax", "jax_laplace", "laplace", "differentiable")
