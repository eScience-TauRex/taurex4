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

import numpy as np
import numpy.typing as npt
import jax
import jax.numpy as jnp
import optax

from taurex.core.priors import Prior
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

        if hasattr(prior, "_low_bounds") and hasattr(prior, "_up_bounds"):
            self._kind = "uniform"
            self._low = float(prior._low_bounds)
            self._high = float(prior._up_bounds)
            self.bounds = (min(self._low, self._high), max(self._low, self._high))
        elif hasattr(prior, "_loc") and hasattr(prior, "_scale"):
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
            smaller than the chain thinning the sampling optimizers need, and
            it is the dominant cost of the run because that step goes through
            the numpy model on the full native grid.

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

        self.atmosphere: t.Optional[Atmosphere] = None
        self._samples: t.Optional[npt.NDArray[np.float64]] = None
        self._map: t.Optional[npt.NDArray[np.float64]] = None
        self._median: t.Optional[npt.NDArray[np.float64]] = None
        self._covariance: t.Optional[npt.NDArray[np.float64]] = None
        self._log_posterior: t.Optional[float] = None
        self._log_likelihood: t.Optional[float] = None

        self.iterations: t.Optional[int] = None
        self.function_evaluations: t.Optional[int] = None
        self.gradient_evaluations: t.Optional[int] = None
        self.compile_time: t.Optional[float] = None
        self.fit_time: t.Optional[float] = None
        self.laplace_time: t.Optional[float] = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def compute_fit(self) -> None:
        """Maximise the posterior and build the Laplace approximation."""
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

        transforms = [
            PriorTransform(param.fit_prior, param.name, dtype) for param in fit_params
        ]
        self.transforms = transforms

        def parameters_of(u: Array) -> Array:
            """Map the unconstrained vector onto the fitting parameters.

            Works for a single vector and for a batch of them: the parameters
            are always stacked along the last axis.
            """
            cube = jax.nn.sigmoid(u)
            return jnp.stack(
                [
                    transform.forward(cube[..., index])
                    for index, transform in enumerate(transforms)
                ],
                axis=-1,
            )

        def negative_log_posterior(u: Array) -> Array:
            """Negative log posterior as a function of ``u``.

            The prior is treated as a density over the parameters themselves,
            so a flat prior contributes nothing and the mode is the chi-squared
            minimum. Writing the objective this way, rather than as the density
            of ``u``, is what keeps the mode and the covariance in the same
            space as the reported parameters.
            """
            theta = parameters_of(u)
            log_prior = sum(
                transform.log_prior(theta[index])
                for index, transform in enumerate(transforms)
            )
            return -atmosphere.log_likelihood(theta) - log_prior

        u = self._starting_point(transforms, dtype)

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

            state:
                Optimiser state at the last step

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
            value_and_grad = optax.value_and_grad_from_state(negative_log_posterior)
            zero = jnp.zeros((), dtype=jnp.int32)

            def condition(carry):
                _, _, gradient_norm, steps, _ = carry
                return (gradient_norm > self.gradient_tolerance) & (
                    steps < self.max_iterations
                )

            def body(carry):
                u_current, state, _, steps, trials = carry
                value, gradient = value_and_grad(u_current, state=state)
                updates, state = solver.update(
                    gradient,
                    state,
                    u_current,
                    value=value,
                    grad=gradient,
                    value_fn=negative_log_posterior,
                )
                u_next = optax.apply_updates(u_current, updates)
                taken = jnp.asarray(
                    optax.tree.get(state, "num_linesearch_steps"), dtype=jnp.int32
                )
                return (
                    u_next,
                    state,
                    jnp.linalg.norm(gradient, ord=jnp.inf),
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

        hessian_of = jax.jit(jax.hessian(negative_log_posterior), device=self.device)
        transform_of = jax.jit(parameters_of, device=self.device)
        jacobian_of = jax.jit(jax.jacobian(parameters_of), device=self.device)
        log_likelihood_of = jax.jit(atmosphere.log_likelihood, device=self.device)
        solve = jax.jit(run, device=self.device)

        # Tracing and compiling is a one off cost that no eager implementation
        # pays, so it is measured separately and reported: it is paid once per
        # fit here and once per process if the compiled functions stayed alive.
        # Every shape the fit will use is compiled now, so that the times below
        # are execution times rather than hidden compilation.
        draws = jnp.zeros((self.num_samples, u.shape[0]), dtype=dtype)
        started = time.perf_counter()
        solve.lower(u).compile()
        hessian_of.lower(u).compile()
        transform_of.lower(u).compile()
        transform_of.lower(draws).compile()
        jacobian_of.lower(u).compile()
        log_likelihood_of.lower(u).compile()
        self.compile_time = time.perf_counter() - started

        started = time.perf_counter()
        # JAX dispatches asynchronously, so the result is waited on here: the
        # optimiser loop runs on the device and the wall clock only measures it
        # once something asks for the value.
        u_map, _, gradient_norm, steps, trials = jax.block_until_ready(solve(u))
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

        theta_map = transform_of(u_map)
        self._log_likelihood = float(log_likelihood_of(theta_map))
        self._log_posterior = float(-negative_log_posterior(u_map))
        self._map = np.asarray(theta_map).copy()

        started = time.perf_counter()
        covariance, variance_scale = self._laplace_covariance(hessian_of, u_map)
        self._samples = self._draw_samples(
            u_map, covariance, variance_scale, transform_of
        )
        self._median = np.median(self._samples, axis=0)
        self._covariance = self._parameter_covariance(
            covariance, variance_scale, jacobian_of, u_map
        )
        self.laplace_time = time.perf_counter() - started

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
        for param, transform in zip(
            self.fitting_parameters, transforms, strict=True
        ):
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
