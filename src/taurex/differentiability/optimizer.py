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

from taurex import OutputSize
from taurex.core.priors import Gaussian
from taurex.core.priors import Prior
from taurex.core.priors import Uniform
from taurex.model import ForwardModel
from taurex.optimizer.optimizer import Optimizer
from taurex.spectrum import BaseSpectrum
from taurex.util import quantile_corner

from .model import Atmosphere


Array = jax.Array

_PROFILE_BATCH_BYTES = 8 * 1024 * 1024
"""Working set the batched profile evaluation is allowed to use."""

_PROFILE_ARRAYS_PER_DRAW = 5
"""Roughly how many ``(nlayers, nwn)`` arrays one forward pass keeps live."""

_NUTS_PROGRESS_INTERVAL = 300.0
"""Seconds of NUTS sampling between two progress messages."""

_NUTS_PROGRESS_PROBE = 100
"""Length of the measuring block as a fraction ``1/n`` of the run."""

_SMC_BATCH_BYTES = 4 * 1024 * 1024 * 1024
"""Working set one batched SMC particle update is allowed to use."""

_SMC_ARRAYS_PER_PARTICLE = 12
"""Roughly how many ``(nlayers, nwn)`` arrays one particle keeps live."""

_SMC_PROPOSAL_DOF = 4.0
"""Degrees of freedom of the Student-t proposal built from the mode."""

_SMC_PRIOR_MIX = 0.1
"""Prior share of the SMC proposal, which bounds the importance weights."""

_SMC_TARGET_ACCEPT = 0.8
"""HMC acceptance rate the step size of the SMC inner kernel aims for."""


def _as_bool(value: t.Any) -> bool:
    """Read an option the input file hands over as text.

    The config reader returns every value as a string, and ``bool("False")``
    is ``True``, so a switch set to off in an input file would silently be on.
    Only the spellings below are accepted; anything else raises rather than
    guessing which way the user meant.
    """
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "yes", "1", "on"):
            return True
        if lowered in ("false", "no", "0", "off"):
            return False
        raise ValueError(f"Cannot read {value!r} as a boolean")
    return bool(value)

_BROADCAST_FIT_FIELDS = (
    "_map",
    "_median",
    "_samples",
    "_covariance",
    "_u_covariance",
    "_log_posterior",
    "_log_likelihood",
    "_fisher",
    "_jitter_map",
    "_n_physical",
    "_weights",
    "_log_evidence",
    "_log_evidence_error",
    "_smc_parameters",
    "_nuts_parameters",
    "iterations",
    "function_evaluations",
    "gradient_evaluations",
    "compile_time",
    "fit_time",
    "laplace_time",
)
"""Results a fit produces, sent from the root rank to the rest of the job."""


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

    def log_normalisation(self) -> float:
        """Constant that turns the sampler prior into a normalised density.

        :meth:`log_prior` drops the constant of the parameter-space prior, and
        :meth:`log_abs_derivative` keeps the constant of the cube-to-parameter
        map. For a mode search that is harmless, since a constant has no
        gradient, but an evidence integral is an absolute number: without this
        correction ``log Z`` carries one ``log`` of a prior width per
        parameter. The two constants cancel for a Gaussian and do not for a
        uniform prior, so the correction is the difference.

        Returns
        -------
        float
            Constant to add to ``log_prior + log_abs_derivative``

        """
        if self._kind == "uniform":
            return -float(np.log(self._high - self._low))
        return -float(np.log(self._scale * np.sqrt(2.0 * np.pi)))


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
        smc_fit_tolerance: t.Optional[float] = 1e-4,
        smc_max_iterations: t.Optional[int] = 60,
        eigenvalue_floor: t.Optional[float] = 1e-3,
        seed: t.Optional[int] = 0,
        device: t.Optional[str] = None,
        sigma_fraction: t.Optional[float] = 0.02,
        sampler: t.Optional[str] = "laplace",
        n_starts: t.Optional[int] = 1,
        start_scale: t.Optional[float] = 0.5,
        num_warmup: t.Optional[int] = 500,
        num_chains: t.Optional[int] = 4,
        max_num_doublings: t.Optional[int] = 10,
        dense_mass_matrix: t.Optional[bool] = False,
        num_particles: t.Optional[int] = 128,
        num_mcmc_steps: t.Optional[int] = 1,
        smc_hmc_steps: t.Optional[int] = 3,
        smc_start: t.Optional[str] = "laplace",
        smc_target_ess: t.Optional[float] = 0.5,
        smc_polish_ess: t.Optional[float] = 0.15,
        smc_max_delta: t.Optional[float] = 0.5,
        smc_min_delta: t.Optional[float] = 0.0,
        smc_max_steps: t.Optional[int] = 50,
        smc_step_size: t.Optional[float] = 0.5,
        smc_batch_size: t.Optional[int] = None,
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

        smc_fit_tolerance:
            Convergence tolerance on the gradient norm for the mode search
            that seeds the ``smc`` proposal. It defaults to ``1e-4`` rather
            than ``gradient_tolerance``: the mode only centres the proposal,
            the posterior comes from the particles, and a retrieval fit that
            is still chasing a gradient of ``1e-10`` can spend most of the
            run doing it. On a three-parameter test the same posterior and
            ``log Z`` came out of a dozen iterations at ``1e-4`` and 200 at
            ``1e-10``. Set it equal to ``gradient_tolerance`` to restore the
            strict mode search.

        smc_max_iterations:
            Hard cap on the mode-search iterations for ``smc``, whatever the
            gradient tolerance is. A retrieval objective can plateau above
            any useful gradient tolerance, and the fit then runs until
            ``max_iterations``; the proposal only needs the mode it has
            already found. ``max_iterations`` is still the cap for the other
            samplers, whose reported MAP is the point of the fit.

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
            ``importance`` reweights proposals drawn from a Student-t centred
            on the mode with the Laplace covariance by the exact density. It
            costs one forward pass per proposal and no gradients, and the
            effective sample size it reports says how much of the posterior
            the Gaussian captured: high means the Laplace description was
            right, low means neither it nor these draws can be trusted.
            ``mclmc`` runs Microcanonical Langevin Monte Carlo from
            :mod:`blackjax`, which produces one sample per gradient rather
            than the tens to hundreds a NUTS trajectory needs, and needs no
            accept/reject and no trajectory length. It is the option to use
            when the posterior is genuinely non-Gaussian and the gradient is
            too expensive for NUTS; it needs at least two fitted parameters.
            Every chain starts at the mode found by the fit, so the draws
            describe the basin around that mode and nothing else: agreement
            between its chains is not evidence that another region of the
            prior does not carry posterior mass, and a nested sampler is the
            only one of these that can answer that.
            ``smc`` runs adaptive tempered Sequential Monte Carlo with a
            fixed-length HMC kernel, the particle filter version of the same
            gradient sampler. It draws a defensive mixture of a Student-t
            built from the mode and the Laplace covariance and of the prior,
            so the proposal can miss a region without the weights becoming
            unbounded, and it computes the initial weights exactly. When the
            proposal is good the posterior and the evidence come out of those
            weights with no gradient evaluations at all; only the part the
            proposal gets wrong is corrected by tempering and HMC. It also has
            a prior start, which reaches a second mode that the mode search
            never found, but pays for that with far more stages. The evidence
            is returned as ``log_evidence``. It requires
            ``blackjax >= 1.7``.

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
            Warm-up steps for the NUTS window adaptation, per chain. NUTS only:
            ``mclmc`` spends about a third of its own number of steps tuning
            its step size and trajectory length, and ``importance`` and
            ``laplace`` need no warm-up at all.

        num_chains:
            Number of chains, for the samplers that run chains (``nuts`` and
            ``mclmc``). The posterior draws are split evenly across them, so a
            chain run also gives a crude convergence check.

        max_num_doublings:
            Longest NUTS trajectory, as a power of two: a draw integrates at
            most ``2 ** max_num_doublings`` leapfrog steps, and each of those
            is one gradient of the model. This is therefore the worst case
            cost of one draw, and the knob to turn down when the progress
            message reports a trajectory that long; lowering it truncates
            trajectories that had not turned around yet, which biases the
            draws, so it is a budget rather than a tuning parameter.

        dense_mass_matrix:
            Adapt a full mass matrix in the warm-up instead of a diagonal one.
            The diagonal form can only fix the scale of each whitened
            coordinate, not the correlations between them, so it leaves
            whatever the Laplace covariance got wrong in place; a dense metric
            measures the correlation from the warm-up draws and removes it. It
            needs ``num_warmup`` to be at least a few times the number of
            fitted parameters to estimate the matrix, and it changes nothing
            when the Laplace whitening was already accurate.

        num_particles:
            Number of particles the ``smc`` sampler carries. It defaults to
            128, not ``num_samples``: the proposal path costs one forward pass
            per particle, so this is also the number of posterior draws and of
            likelihood evaluations before any gradient is spent. The Monte
            Carlo error of the evidence and the draws falls as one over its
            square root.

        num_mcmc_steps:
            HMC transitions applied to every particle in every tempering
            stage of the ``smc`` sampler. One full transition already moves a
            particle across the posterior, so the default is a single one; a
            second is only worth it when a stage makes a large jump in the
            tempering parameter.

        smc_hmc_steps:
            Leapfrog steps in each HMC transition of the ``smc`` sampler. This
            fixes the cost of a transition: ``smc_hmc_steps`` model gradients
            per particle, independent of the posterior. It replaces the
            adaptive NUTS trajectory, whose worst case is thousands of
            gradients per draw. Three is the default because the tempering
            stages, not the trajectory, do most of the decorrelation; raise it
            when a stage has to make a large jump.

        smc_start:
            ``laplace`` (default) starts the particles from a Student-t built
            from the mode and the Laplace covariance, with prior-corrected
            initial weights. The Student-t has four degrees of freedom, so it
            has heavier tails than the Gaussian and its weights are bounded.
            ``prior`` draws the particles from the prior instead, which is
            slower but can find a posterior mode the mode search never
            reached, and is the right choice when multi-modality is suspected.

        smc_target_ess:
            Fraction of the particles the ``smc`` tempering schedule keeps
            effective at the end of a stage, and the fraction the initial
            proposal must keep for the run to skip HMC at all. The next
            tempering increment is the largest one that leaves this fraction
            of the particles effective; a smaller value takes fewer, larger
            stages and a larger value takes more, smaller ones.

        smc_polish_ess:
            Fraction of the particles the proposal must keep for ``smc`` to
            accept it with one HMC polish at ``beta = 1`` instead of
            tempering it. This covers the common middle case: the weights are
            not good enough to trust as they are, but they are good enough
            that resampling and moving the particles once is cheaper and more
            accurate than a ladder of tempering stages. The evidence error
            reports the weight degeneracy either way; below this fraction the
            proposal is abandoned for the prior start.

        smc_max_delta:
            Largest increase of the tempering parameter in one ``smc`` stage.
            The default of one half allows two stages, which is enough to
            correct a proposal that is off by a moderate amount without
            spending a long chain of stages on it.

        smc_min_delta:
            Smallest increase of the tempering parameter in one ``smc`` stage.
            The ESS schedule can ask for increments a thousandth of the range
            when the prior and the likelihood disagree strongly, and each of
            those stages pays for a full HMC update of every particle. A
            positive floor bounds the number of stages at
            ``ceil(1 / smc_min_delta)``, at the cost of a lower effective
            sample size in those stages; the estimate stays unbiased but its
            variance grows. The default of zero leaves the ESS schedule free.

        smc_max_steps:
            Maximum number of tempering stages. Reaching it before the
            tempering parameter is one means the run stopped on a tempered
            target: the draws and the evidence would then describe that
            target, not the posterior, and the run says so.

        smc_step_size:
            Initial HMC step size of the ``smc`` inner kernel. The metric is
            the Laplace covariance, so a step of order one is the natural
            scale and this only sets where the dual averaging starts.

        smc_batch_size:
            Number of particles evaluated together in the ``smc`` sampler.
            ``None`` (default) sizes the batch from a fixed memory budget
            using the model's layer and wavenumber counts. A larger batch is
            faster on a GPU and costs memory; ``num_particles`` maps all
            particles at once.

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
        self.scale_init_precond = _as_bool(scale_init_precond)
        self.gradient_tolerance = float(gradient_tolerance)
        self.smc_fit_tolerance = float(smc_fit_tolerance)
        if self.smc_fit_tolerance <= 0.0:
            raise ValueError(
                "smc_fit_tolerance must be positive, "
                f"not {smc_fit_tolerance!r}"
            )
        self.smc_max_iterations = max(1, int(smc_max_iterations))
        self.eigenvalue_floor = float(eigenvalue_floor)
        self.seed = seed
        self.device = None if device is None else jax.devices(device)[0]
        self.ignored_options = kwargs

        self.sampler = str(sampler).lower()
        if self.sampler not in ("laplace", "nuts", "importance", "mclmc", "smc"):
            raise ValueError(
                f"Unknown sampler {sampler!r}; use 'laplace', 'nuts', "
                "'importance', 'mclmc' or 'smc'"
            )
        self.n_starts = max(1, int(n_starts))
        self.start_scale = float(start_scale)
        self.num_warmup = int(num_warmup)
        self.num_chains = max(1, int(num_chains))
        self.max_num_doublings = int(max_num_doublings)
        self.dense_mass_matrix = _as_bool(dense_mass_matrix)
        self.num_particles = int(
            self.num_samples if num_particles is None else num_particles
        )
        if self.num_particles < 2:
            raise ValueError(
                "SMC needs at least two particles for the tempering to have an "
                "effective sample size to work with"
            )
        self.num_mcmc_steps = max(1, int(num_mcmc_steps))
        self.smc_hmc_steps = max(1, int(smc_hmc_steps))
        self.smc_start = str(smc_start).lower()
        if self.smc_start not in ("laplace", "prior"):
            raise ValueError(
                f"smc_start must be 'laplace' or 'prior', not {smc_start!r}"
            )
        self.smc_target_ess = float(smc_target_ess)
        if not 0.0 < self.smc_target_ess <= 1.0:
            raise ValueError(
                f"smc_target_ess must lie in (0, 1], not {smc_target_ess!r}"
            )
        self.smc_polish_ess = float(smc_polish_ess)
        if not 0.0 <= self.smc_polish_ess <= 1.0:
            raise ValueError(
                f"smc_polish_ess must lie in [0, 1], not {smc_polish_ess!r}"
            )
        self.smc_max_delta = float(smc_max_delta)
        if not 0.0 < self.smc_max_delta <= 1.0:
            raise ValueError(
                f"smc_max_delta must lie in (0, 1], not {smc_max_delta!r}"
            )
        self.smc_min_delta = float(smc_min_delta)
        if not 0.0 <= self.smc_min_delta <= self.smc_max_delta:
            raise ValueError(
                "smc_min_delta must lie in [0, smc_max_delta], "
                f"not {smc_min_delta!r}"
            )
        self.smc_max_steps = max(1, int(smc_max_steps))
        self.smc_step_size = float(smc_step_size)
        if self.smc_step_size <= 0.0:
            raise ValueError(
                f"smc_step_size must be positive, not {smc_step_size!r}"
            )
        self.smc_batch_size = None if smc_batch_size is None else int(smc_batch_size)
        if self.smc_batch_size is not None and self.smc_batch_size < 1:
            raise ValueError(
                f"smc_batch_size must be positive, not {smc_batch_size!r}"
            )
        self.jitter = _as_bool(jitter)
        self.jitter_scale = None if jitter_scale is None else float(jitter_scale)

        self.atmosphere: t.Optional[Atmosphere] = None
        self.transforms: t.List[PriorTransform] = []
        self._samples: t.Optional[npt.NDArray[np.float64]] = None
        self._map: t.Optional[npt.NDArray[np.float64]] = None
        self._median: t.Optional[npt.NDArray[np.float64]] = None
        self._covariance: t.Optional[npt.NDArray[np.float64]] = None
        self._u_covariance: t.Optional[npt.NDArray[np.float64]] = None
        self._log_posterior: t.Optional[float] = None
        self._log_likelihood: t.Optional[float] = None
        self._fisher: t.Optional[npt.NDArray[np.float64]] = None
        self._jitter_map: t.Optional[float] = None
        self._n_physical: int = 0
        self._weights: t.Optional[npt.NDArray[np.float64]] = None
        self._log_evidence: t.Optional[float] = None
        self._log_evidence_error: t.Optional[float] = None
        self._smc_parameters: t.Optional[t.Dict[str, t.Any]] = None
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

        Under MPI only the root rank fits. The quasi-Newton chain is serial and
        the gradient is already spread over every core by XLA, so extra ranks
        cannot shorten it: they would each build their own copy of the
        atmosphere, compile their own executables and repeat the identical
        fit. The root broadcasts the posterior instead, which keeps the peak
        memory of a job at one compiled model however many ranks it has.
        """
        from taurex.mpi import broadcast
        from taurex.mpi import get_rank
        from taurex.mpi import nprocs

        rank = get_rank()
        size = nprocs()
        # SMC is the one sampler that shards: a particle's likelihood and
        # gradient are independent of every other particle, so the expensive
        # part of a stage can run on different MPI ranks. The mode fit stays
        # on rank 0 because the L-BFGS chain is sequential.
        mpi_smc = self.sampler == "smc" and size > 1
        fit_on_this_rank = not (mpi_smc and rank != 0)
        if size > 1 and rank != 0 and not mpi_smc:
            self._receive_fit()
            return
        if size > 1 and not mpi_smc:
            # A pipeline that allocates a whole node per rank still gets a
            # correct answer, but only this rank does the fit and the others
            # sit in the broadcast, so the job is slower than one rank with
            # the node to itself. Saying so is cheaper than a walltime limit
            # spent wondering why 128 ranks did not help.
            self.warning(
                "Fitting on 1 of %d MPI ranks: the L-BFGS chain and the SMC "
                "particles run inside one JAX process, and XLA spreads each "
                "step over that process's threads or device; the other ranks "
                "wait for the broadcast. Launch the JAX optimizer with one "
                "task and all the cores of the node (srun --ntasks=1 "
                "--cpus-per-task=N) rather than many MPI ranks, which only "
                "add model copies and can oversubscribe the CPUs",
                nprocs(),
            )
        elif mpi_smc:
            self.info(
                "SMC: %d MPI ranks share the particle axis; the mode fit runs "
                "on rank 0 and is broadcast",
                size,
            )

        import os

        # What XLA is actually working with. On a cluster this is the first
        # thing to check when a run ignores the cores it was given: os.cpu_count
        # is the node's CPU count, not necessarily the cgroup limit, and
        # jax.devices is what the process can see.
        self.info(
            "JAX: %d host CPUs, devices %s",
            os.cpu_count() or 0,
            [f"{device.platform}:{device.id}" for device in jax.devices()],
        )

        fit_params = self.fitting_parameters
        if not fit_params:
            raise ValueError(
                "No fitting parameters were enabled, so there is nothing to "
                "fit. Add entries under [Fitting] in the input file."
            )

        # A second fit on the same object must not inherit the weights or the
        # evidence of the first one, which a different sampler may not set.
        self._weights = None
        self._log_evidence = None
        self._log_evidence_error = None
        self._smc_parameters = None

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

        def log_likelihood_u(u: Array) -> Array:
            """Log likelihood at an unconstrained vector.

            Tempered SMC needs the likelihood on its own, because the tempering
            exponent multiplies it and leaves the prior alone; the concatenation
            of this and :func:`log_prior_u` is :func:`nuts_objective`.
            """
            theta = parameters_of(u)
            physical = theta[..., : self._n_physical]
            if jitter:
                size = theta[..., len(transforms)]
                error_here = jnp.sqrt(error**2 + size**2)
            else:
                error_here = error
            return atmosphere.log_likelihood(physical, data=data, error=error_here)

        def log_prior_u(u: Array) -> Array:
            """Log prior density in unconstrained space, Jacobian included.

            The density of ``u`` is the parameter prior times
            ``|d theta / d u|``, which is the distribution a tempered SMC
            starts from before any likelihood is applied. The dropped prior
            constants are restored here: a mode search is blind to them but
            the evidence is not.
            """
            theta = parameters_of(u)
            log_prior = sum(
                transform.log_prior(theta[..., index])
                + transform.log_normalisation()
                for index, transform in enumerate(transforms)
            )
            if jitter:
                size = theta[..., len(transforms)]
                # Half-normal prior on the jitter, normalised in u as well.
                log_prior = (
                    log_prior
                    - 0.5 * (size / self.jitter_scale) ** 2
                    + np.log(2.0)
                    - np.log(self.jitter_scale)
                    - 0.5 * np.log(2.0 * np.pi)
                )
            return log_prior + log_transform_jacobian(u)

        u = self._starting_point(transforms, dtype)
        if jitter:
            # A tenth of the prior scale: away from the jitter = 0 boundary the
            # exponential cannot reach, but small enough not to bias the fit.
            u = jnp.concatenate([u, jnp.array([np.log(0.1)], dtype=dtype)])

        # SMC's proposal needs a good mode, not a converged one, and a
        # retrieval objective that is still chasing a 1e-10 gradient can spend
        # most of the run doing it. The posterior comes from the particles, so
        # only SMC relaxes the stopping test.
        fit_tolerance = (
            self.smc_fit_tolerance
            if self.sampler == "smc"
            else self.gradient_tolerance
        )
        fit_iterations = (
            self.smc_max_iterations
            if self.sampler == "smc"
            else self.max_iterations
        )

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
                return (gradient_norm > fit_tolerance) & (
                    steps < fit_iterations
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
                active = (gradient_norm > fit_tolerance) & (
                    steps < fit_iterations
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

        def run_starts(starts: Array) -> t.Tuple[t.Any, ...]:
            """Fit every starting point and keep the best posterior.

            :func:`jax.lax.map` runs the starts one after another inside the
            compiled program. A :func:`jax.vmap` would batch them, but a
            batched ``while_loop`` stops only when the slowest start has
            finished and its extra iterations are not worth the risk of moving
            a converged start off its optimum, so each start gets its own loop.
            """
            u_maps, states, gradient_norms, steps, trials = jax.lax.map(
                run, starts
            )
            scores = jax.lax.map(objective, u_maps)
            best = jnp.argmin(scores)
            return (
                u_maps[best],
                jax.tree.map(lambda leaf: leaf[best], states),
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

        u_map: t.Optional[Array] = None
        lbfgs_state: t.Any = None
        if fit_on_this_rank:
            # Tracing and compiling is a one off cost that no eager
            # implementation pays, so it is measured separately and reported:
            # it is paid once per fit here and once per process if the
            # compiled functions stayed alive. Every shape the fit will use is
            # compiled now, so that the times below are execution times rather
            # than hidden compilation.
            draws = jnp.zeros((self.num_samples, u.shape[0]), dtype=dtype)
            started = time.perf_counter()
            solve.lower(starts).compile()
            transform_of.lower(u).compile()
            transform_of.lower(draws).compile()
            log_likelihood_of.lower(u[: self._n_physical], data, error).compile()
            if self.sampler == "laplace":
                # These two are only used by the Laplace covariance and its
                # push forward to parameter space. Every other sampler builds
                # its own Hessian from the objective with the transform
                # Jacobian, so compiling these programs here would be paid for
                # nothing.
                hessian_of.lower(u).compile()
                jacobian_of.lower(u).compile()
            self.compile_time = time.perf_counter() - started

            started = time.perf_counter()
            # JAX dispatches asynchronously, so the result is waited on here:
            # the optimiser loop runs on the device and the wall clock only
            # measures it once something asks for the value.
            u_map, lbfgs_state, gradient_norm, steps, trials, score = (
                jax.block_until_ready(solve(starts))
            )
            self.fit_time = time.perf_counter() - started

            self.iterations = int(steps)
            self.function_evaluations = int(steps + trials)
            self.gradient_evaluations = int(steps)
            if self.iterations >= fit_iterations and (
                float(gradient_norm) > fit_tolerance
            ):
                # The objective of a retrieval is flat enough that most fits
                # stop here rather than on the gradient tolerance. Saying so is
                # the difference between "this is the mode" and "this is as far
                # as the fit got"; the point returned is the best one it found.
                self.warning(
                    "The quasi-Newton fit reached its iteration limit after %d "
                    "steps with gradient norm %.3e, above the tolerance %.3e",
                    self.iterations,
                    float(gradient_norm),
                    fit_tolerance,
                )

            theta_full = transform_of(u_map)
            self._map = np.asarray(theta_full[: self._n_physical]).copy()
            if jitter:
                self._jitter_map = float(theta_full[self._n_physical])
                # Host arithmetic: four operations on a vector of the length of
                # the data, which are cheaper to compute than to dispatch.
                error_effective = np.sqrt(
                    np.asarray(error) ** 2 + self._jitter_map**2
                )
            else:
                self._jitter_map = None
                error_effective = error
            self._log_likelihood = float(
                log_likelihood_of(
                    theta_full[: self._n_physical], data, error_effective
                )
            )
            # The compiled fit already returned the negative log posterior at
            # the point it kept, so it does not have to be evaluated again:
            # calling ``objective`` here would run the whole atmosphere
            # eagerly, which costs a small compilation per operation rather
            # than a dispatch.
            self._log_posterior = -float(score)

        started = time.perf_counter()
        if self.sampler == "smc":
            if fit_on_this_rank:
                # The mode and its covariance are the proposal for the
                # particles, so the Hessian is needed even when the run ends in
                # the importance-sampling shortcut and never takes a gradient.
                nuts_hessian_of = jax.jit(
                    jax.hessian(nuts_objective), device=self.device
                )
                covariance, variance_scale = self._laplace_covariance(
                    nuts_hessian_of, u_map
                )
                covariance_u = covariance * variance_scale
                self._u_covariance = covariance_u
                lbfgs_covariance = None
                if not self._is_usable_covariance(covariance_u):
                    lbfgs_covariance = self._lbfgs_covariance(
                        lbfgs_state, u.shape[0]
                    )
            else:
                covariance_u = None
                lbfgs_covariance = None
            if mpi_smc:
                # The proposal, the metric and every result of the mode fit are
                # small; send them once and let every rank run the same
                # particle state with a shard of the likelihood and gradients.
                setup = None
                if rank == 0:
                    setup = {
                        "u_map": np.asarray(u_map),
                        "covariance_u": np.asarray(covariance_u),
                        "lbfgs_covariance": lbfgs_covariance,
                        "map": self._map,
                        "log_likelihood": self._log_likelihood,
                        "log_posterior": self._log_posterior,
                        "jitter_map": self._jitter_map,
                        "iterations": self.iterations,
                        "function_evaluations": self.function_evaluations,
                        "gradient_evaluations": self.gradient_evaluations,
                        "compile_time": self.compile_time,
                        "fit_time": self.fit_time,
                    }
                setup = broadcast(setup)
                u_map = jnp.asarray(setup["u_map"], dtype=dtype)
                covariance_u = jnp.asarray(setup["covariance_u"], dtype=dtype)
                lbfgs_covariance = setup["lbfgs_covariance"]
                self._u_covariance = covariance_u
                self._map = setup["map"]
                self._log_likelihood = setup["log_likelihood"]
                self._log_posterior = setup["log_posterior"]
                self._jitter_map = setup["jitter_map"]
                self.iterations = setup["iterations"]
                self.function_evaluations = setup["function_evaluations"]
                self.gradient_evaluations = setup["gradient_evaluations"]
                self.compile_time = setup["compile_time"]
                self.fit_time = setup["fit_time"]
            self._samples, self._weights = self._run_smc(
                log_prior_u,
                log_likelihood_u,
                transform_of,
                u_map,
                covariance_u,
                lbfgs_covariance,
            )
            self._median = np.array(
                [
                    quantile_corner(
                        self._samples[:, index], [0.5], weights=self._weights
                    )[0]
                    for index in range(self._n_physical)
                ]
            )
            if np.count_nonzero(self._weights) > 1:
                self._covariance = np.atleast_2d(
                    np.cov(self._samples, rowvar=False, aweights=self._weights)
                )
            else:
                self._covariance = np.zeros(
                    (self._n_physical, self._n_physical), dtype=float
                )
        else:
            if self.sampler != "laplace":
                # The mass matrix has to describe the same density the sampler
                # targets, so the Hessian here includes the transform Jacobian.
                nuts_hessian_of = jax.jit(
                    jax.hessian(nuts_objective), device=self.device
                )
                covariance, variance_scale = self._laplace_covariance(
                    nuts_hessian_of, u_map
                )
            else:
                covariance, variance_scale = self._laplace_covariance(
                    hessian_of, u_map
                )
            self._u_covariance = covariance * variance_scale
            if self.sampler != "laplace":
                if self.sampler == "nuts":
                    self._samples = self._run_nuts(
                        nuts_objective,
                        u_map,
                        covariance,
                        variance_scale,
                        transform_of,
                    )
                elif self.sampler == "importance":
                    self._samples = self._run_importance_sampling(
                        nuts_objective,
                        u_map,
                        covariance,
                        variance_scale,
                        transform_of,
                    )
                else:
                    self._samples = self._run_mclmc(
                        nuts_objective,
                        u_map,
                        covariance,
                        variance_scale,
                        transform_of,
                    )
                # These samplers produce independent draws, so the weights are
                # uniform and are reported as such.
                self._weights = None
                self._median = np.median(self._samples, axis=0)
                if self._samples.shape[0] > 1:
                    self._covariance = np.atleast_2d(
                        np.cov(self._samples, rowvar=False)
                    )
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
                self._weights = None
                self._median = np.median(self._samples, axis=0)
                full_covariance = self._parameter_covariance(
                    covariance, variance_scale, jacobian_of, u_map
                )
                self._covariance = np.asarray(full_covariance)[
                    : self._n_physical, : self._n_physical
                ].copy()
        self.laplace_time = time.perf_counter() - started

        if nprocs() > 1:
            self._broadcast_fit()

    def _broadcast_fit(self) -> None:
        """Send the fitted posterior from the root rank to the other ranks.

        The payload is the posterior itself - mode, draws, covariance and the
        fitted nuisance values - plus the counters that describe the fit. It
        is a few thousand floats, so broadcasting it costs far less than a
        second copy of the atmosphere or a second compilation.
        """
        from taurex.mpi import broadcast

        broadcast({name: getattr(self, name) for name in _BROADCAST_FIT_FIELDS})

    def _receive_fit(self) -> None:
        """Take the fitted posterior broadcast by the root rank."""
        from taurex.mpi import broadcast

        payload = broadcast(None)
        for name, value in payload.items():
            setattr(self, name, value)

    def _run_smc(
        self,
        log_prior_fn: t.Callable[[Array], Array],
        log_likelihood_fn: t.Callable[[Array], Array],
        transform_of: t.Callable[[Array], Array],
        u_map: Array,
        covariance_u: Array,
        lbfgs_covariance: t.Optional[npt.NDArray[np.float64]] = None,
    ) -> t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Draw posterior samples and the evidence with tempered SMC.

        Sequential Monte Carlo samples

        .. math::

            \\pi_\\beta(u) \\propto p(u) \\, L(u)^\\beta

        and raises :math:`\\beta` from its starting value to one. A tempering
        stage chooses the largest increase of :math:`\\beta` that leaves
        ``smc_target_ess`` of the particles effective, reweights them by the
        likelihood increment, resamples, and moves every particle with the HMC
        kernel. The product of the stage normalising constants is the evidence.

        The starting point is what makes this affordable. With
        ``smc_start='laplace'`` the particles come from a Student-t centred on
        the mode with the Laplace covariance, and the initial weights are
        ``p(u) L(u)^\\beta / q(u)``. The effective sample size of those weights
        decides how much work is left: when the proposal already describes the
        posterior, the weights are the answer and the run spends no gradient
        at all; when it does not, the run starts at the largest
        :math:`\\beta` whose weights are still effective and corrects the rest
        with HMC. This is the same proposal the importance sampler uses, with
        SMC as the fallback that makes it exact where the proposal fails.

        ``smc_start='prior'`` instead draws from the prior and tempers from
        zero. It reaches modes the proposal cannot, at the price of far more
        stages.

        Parameters
        ----------
        log_prior_fn:
            Log prior density in unconstrained space, Jacobian included

        log_likelihood_fn:
            Log likelihood of one unconstrained vector

        transform_of:
            Compiled map from ``u`` to the parameters

        u_map:
            Location of the mode, the centre of the proposal

        covariance_u:
            Covariance of the unconstrained posterior, the proposal's scale
            and the HMC metric

        Returns
        -------
        samples, weights:
            Posterior draws in parameter space with shape
            ``(num_particles, nparams)`` and their normalised weights

        """
        from blackjax.mcmc import hmc as hmc_kernel
        from blackjax.smc import base as smc_base
        from blackjax.smc import ess as smc_ess
        from blackjax.smc import solver as smc_solver
        from blackjax.smc import tempered as smc_tempered
        from blackjax.smc.resampling import systematic
        from scipy import special as scipy_special

        import logging

        # ``compute_fit`` runs with the root logger at CRITICAL so that the
        # numpy model does not flood the log, which would also hide the only
        # progress this sampler reports. Raising this optimizer's own logger
        # shows the stages without re-enabling everything else.
        self._logger.setLevel(logging.INFO)

        dtype = jnp.float64
        from taurex.mpi import get_rank, nprocs

        rank = get_rank()
        size = nprocs()
        mpi = size > 1
        particles_count = self.num_particles
        dimension = u_map.shape[0]
        if mpi:
            # The particle axis is split into equal contiguous chunks, so the
            # allgather concatenates back into the global order. A count that
            # does not divide is rounded up rather than leaving a rank with an
            # empty shard and a shape of its own to compile.
            if particles_count % size:
                adjusted = ((particles_count + size - 1) // size) * size
                self.info(
                    "SMC: %d particles rounded up to %d so the %d MPI ranks "
                    "split them evenly",
                    particles_count,
                    adjusted,
                    size,
                )
                particles_count = adjusted
            if particles_count < size:
                raise ValueError(
                    f"SMC with {size} MPI ranks needs at least {size} "
                    f"particles, not {particles_count}; raise num_particles"
                )
        chunk = particles_count // size if mpi else particles_count
        local_slice = slice(rank * chunk, (rank + 1) * chunk)

        # Batching the particle axis keeps the working set bounded. The budget
        # has to be generous: one particle holds several ``(nlayers, nwn)``
        # arrays, and a batch of one serialises the whole sampler.
        atmosphere = self.atmosphere
        per_particle = (
            atmosphere.wngrid.dtype.itemsize
            * atmosphere.model.nLayers
            * atmosphere.wngrid.shape[0]
            * _SMC_ARRAYS_PER_PARTICLE
        )
        if self.smc_batch_size is None:
            batch = max(1, _SMC_BATCH_BYTES // max(per_particle, 1))
        else:
            batch = self.smc_batch_size
        batch = min(batch, chunk)
        # ``batch_size=0`` asks blackjax for a single vmap over the local
        # chunk, which is the fastest option whenever the memory estimate
        # allows it.
        vectorised_batch = 0 if batch >= chunk else batch
        batched_likelihood = smc_base.map_fn(
            log_likelihood_fn, vectorised_batch
        )
        if mpi:
            from mpi4py import MPI

            comm = MPI.COMM_WORLD

            def likelihood_of(
                positions: npt.NDArray[np.float64],
            ) -> npt.NDArray[np.float64]:
                """Log likelihood of the global particles, one shard per rank."""
                local = np.asarray(
                    jax.block_until_ready(
                        batched_likelihood(jnp.asarray(positions[local_slice]))
                    )
                )
                pieces = comm.allgather(np.ascontiguousarray(local))
                return np.concatenate(pieces, axis=0)

        else:
            comm = None

            def likelihood_of(
                positions: npt.NDArray[np.float64],
            ) -> npt.NDArray[np.float64]:
                """Log likelihood of a batch of unconstrained vectors."""
                return np.asarray(
                    jax.block_until_ready(
                        batched_likelihood(jnp.asarray(positions))
                    )
                )

        self.info(
            "SMC: %d particles on %d rank(s), %d per rank, estimated %.0f MB "
            "per particle, batches of %s, HMC %d leapfrogs x %d transition(s)",
            particles_count,
            size,
            chunk,
            per_particle / 1e6,
            "all" if vectorised_batch == 0 else str(batch),
            self.smc_hmc_steps,
            self.num_mcmc_steps,
        )
        self.info(
            "SMC: running on %s; the particle axis is the only parallel one, "
            "the leapfrog steps and tempering stages are sequential",
            "JAX's default device" if self.device is None else str(self.device),
        )

        def log_ess(values: npt.NDArray[np.float64]) -> float:
            """Effective sample size of an array of unnormalised log weights."""
            return float(
                np.exp(
                    2.0 * scipy_special.logsumexp(values)
                    - scipy_special.logsumexp(2.0 * values)
                )
            )

        def likelihood_of(
            positions: npt.NDArray[np.float64],
        ) -> npt.NDArray[np.float64]:
            """Log likelihood of a batch of unconstrained vectors."""
            return np.asarray(
                jax.block_until_ready(batched_likelihood(jnp.asarray(positions)))
            )

        def prior_of(positions: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            """Log prior density of a batch, Jacobian included."""
            return np.asarray(
                jax.block_until_ready(jax.vmap(log_prior_fn)(jnp.asarray(positions)))
            )

        key = jax.random.PRNGKey(0 if self.seed is None else int(self.seed))
        key, t_key, prior_key, chi_key, jitter_key, mix_key = jax.random.split(
            key, 6
        )

        covariance_np = np.asarray(covariance_u, dtype=float)
        covariance_np = 0.5 * (covariance_np + covariance_np.T)
        # A parameter pushed to a prior bound makes ``log|d theta / d u|``
        # diverge, and the Hessian then comes back non-finite; a flat
        # direction comes back with a zero scale. Either way the covariance
        # cannot be factored, and the proposal built from it would be garbage.
        # The prior start needs no proposal, so fall back to it with a
        # prior-scale metric instead of crashing in the factorisation.
        prior_variance = np.pi**2 / 3.0
        if np.all(np.isfinite(covariance_np)):
            smallest = float(np.min(np.linalg.eigvalsh(covariance_np)))
        else:
            smallest = float("-inf")
        usable_covariance = smallest > 0.0
        if not usable_covariance and lbfgs_covariance is not None:
            candidate = np.asarray(lbfgs_covariance, dtype=float)
            if self._is_usable_covariance(candidate):
                self.warning(
                    "SMC: the Laplace Hessian is not positive definite "
                    "(smallest eigenvalue %s); using the L-BFGS "
                    "inverse-Hessian approximation for the proposal and the "
                    "metric",
                    smallest,
                )
                covariance_np = candidate
                usable_covariance = True
        if usable_covariance:
            factor = np.linalg.cholesky(covariance_np)
            log_det = float(np.sum(np.log(np.diag(factor))))
        else:
            self.warning(
                "SMC: the Laplace covariance is not positive definite "
                "(smallest eigenvalue %s), so no proposal can be built from "
                "it; tempering from the prior with a prior-scale metric",
                smallest,
            )
            covariance_np = prior_variance * np.eye(dimension)
            factor = np.linalg.cholesky(covariance_np)
            log_det = float(np.sum(np.log(np.diag(factor))))
        u_map_np = np.asarray(u_map, dtype=float)

        def prior_positions_of(draw_key: Array) -> npt.NDArray[np.float64]:
            """Draw particles from the prior in unconstrained space.

            A uniform cube pushed through the prior transform is a prior draw
            for every prior type the transforms support; for the jitter the
            prior is half normal on the size, so a normal magnitude is exact.
            The cube is kept just inside its edges because the logit is
            infinite there.
            """
            cube = jax.random.uniform(
                draw_key,
                (particles_count, dimension),
                dtype=dtype,
                minval=1e-6,
                maxval=1.0 - 1e-6,
            )
            positions = np.array(
                np.asarray(jax.scipy.special.logit(cube)), dtype=float, copy=True
            )
            if self.jitter:
                magnitude = np.asarray(
                    jnp.abs(
                        jax.random.normal(
                            jitter_key, (particles_count,), dtype=dtype
                        )
                    )
                )
                positions[:, dimension - 1] = np.log(
                    np.clip(magnitude, 1e-6, None)
                )
            return positions

        start = "prior"
        beta0 = 0.0
        polish = False
        proposal_ess = float(particles_count)
        log_weights = np.zeros(particles_count, dtype=float)
        log_evidence = 0.0
        log_evidence_variance = 0.0
        log_like: t.Optional[npt.NDArray[np.float64]] = None
        if self.smc_start == "laplace" and usable_covariance:
            # A Student-t with four degrees of freedom has twice the variance
            # of the Gaussian it is centred on and polynomial tails. A pure
            # Gaussian proposal would leave the importance weights unbounded
            # in the tails, and a pure t is still unbounded against a logistic
            # prior; adding a defensive prior share makes the density at least
            # ``_SMC_PRIOR_MIX`` times the prior, so no weight can exceed
            # ``1 / _SMC_PRIOR_MIX`` and the evidence no longer hinges on a
            # few tail draws.
            normal = np.asarray(
                jax.random.normal(
                    t_key, (particles_count, dimension), dtype=dtype
                )
            )
            chi = np.asarray(
                jax.random.chisquare(
                    chi_key, _SMC_PROPOSAL_DOF, (particles_count,), dtype=dtype
                )
            )
            t_whitened = normal * np.sqrt(_SMC_PROPOSAL_DOF / chi)[:, None]
            t_positions = u_map_np[None, :] + t_whitened @ factor.T
            prior_positions = prior_positions_of(prior_key)
            from_prior = np.asarray(
                jax.random.uniform(mix_key, (particles_count,), dtype=dtype)
            ) < _SMC_PRIOR_MIX
            positions = np.where(
                from_prior[:, None], prior_positions, t_positions
            )

            whitened = np.linalg.solve(
                factor, (positions - u_map_np[None, :]).T
            ).T
            quadratic = np.sum(whitened**2, axis=1)
            log_t = (
                scipy_special.gammaln(0.5 * (_SMC_PROPOSAL_DOF + dimension))
                - scipy_special.gammaln(0.5 * _SMC_PROPOSAL_DOF)
                - 0.5 * dimension * np.log(_SMC_PROPOSAL_DOF * np.pi)
                - 0.5 * (_SMC_PROPOSAL_DOF + dimension)
                * np.log1p(quadratic / _SMC_PROPOSAL_DOF)
                - log_det
            )
            log_prior = prior_of(positions)
            log_proposal = np.logaddexp(
                np.log(_SMC_PRIOR_MIX) + log_prior,
                np.log1p(-_SMC_PRIOR_MIX) + log_t,
            )
            log_like = likelihood_of(positions)

            # β = 1 is the posterior itself. If the proposal already carries
            # it, the weighted particles are exact posterior draws and no
            # tempering or gradient is needed.
            log_full = log_prior + log_like - log_proposal
            proposal_ess = log_ess(log_full)
            target_count = self.smc_target_ess * particles_count
            self.info(
                "SMC: mixture proposal ESS %.0f/%d at beta=1 (%.0f needed to "
                "skip the gradient stages)",
                proposal_ess,
                particles_count,
                target_count,
            )
            if proposal_ess >= target_count:
                log_evidence = float(
                    scipy_special.logsumexp(log_full) - np.log(particles_count)
                )
                log_evidence_variance = (
                    max(particles_count / proposal_ess - 1.0, 0.0) / particles_count
                )
                self._log_evidence = log_evidence
                self._log_evidence_error = float(np.sqrt(log_evidence_variance))
                self._smc_parameters = {
                    "start": "laplace",
                    "beta0": 1.0,
                    "proposal_ess": proposal_ess,
                    "stages": 0,
                    "gradient_evaluations": 0,
                    "batch_size": batch,
                    "hmc_steps": self.smc_hmc_steps,
                    "polish": False,
                    "ranks": size,
                    "particles": particles_count,
                }
                weights = np.exp(
                    log_full - scipy_special.logsumexp(log_full)
                )
                self.info(
                    "SMC: the Laplace/prior mixture keeps ESS %.0f/%d, so its "
                    "weighted particles are the posterior and the evidence; "
                    "no tempering or gradient is spent",
                    proposal_ess,
                    particles_count,
                )
                theta = transform_of(jnp.asarray(positions, dtype=dtype))
                return np.asarray(theta[:, : self._n_physical]).copy(), weights
            polish = proposal_ess >= self.smc_polish_ess * particles_count
            if polish:
                # The weights are not good enough to use as they are, but they
                # are good enough that resampling them and moving the particles
                # once at beta = 1 is far cheaper than a ladder of stages. HMC
                # at beta = 1 leaves the posterior invariant, so the draws stay
                # exact; only the evidence keeps the weight variance, which is
                # reported.
                start = "laplace"
                beta0 = 1.0
                log_weights = log_full
                log_evidence = float(
                    scipy_special.logsumexp(log_full) - np.log(particles_count)
                )
                log_evidence_variance = (
                    max(particles_count / proposal_ess - 1.0, 0.0)
                    / particles_count
                )
                self.info(
                    "SMC: proposal ESS %.0f/%d is below the shortcut target; "
                    "resampling and polishing with one HMC transition at "
                    "beta=1",
                    proposal_ess,
                    particles_count,
                )
            else:
                # The proposal is not good enough to skip HMC. Start tempering
                # at the furthest point where the weights still have a usable
                # effective sample; HMC moves what survives, and the initial
                # ESS is carried in the evidence error rather than hidden.
                grid = np.linspace(0.0, 1.0, 21)[:-1]
                ess_grid = np.array(
                    [
                        log_ess(log_prior + beta * log_like - log_proposal)
                        for beta in grid
                    ]
                )
                usable = max(2.0, 0.25 * target_count)
                qualifying = grid[ess_grid >= usable]
                beta0 = float(qualifying.max()) if qualifying.size else 0.0
                if beta0 > 0.0:
                    start = "laplace"
                    log_weights = log_prior + beta0 * log_like - log_proposal
                    log_evidence = float(
                        scipy_special.logsumexp(log_weights)
                        - np.log(particles_count)
                    )
                    ess0 = log_ess(log_weights)
                    log_evidence_variance = (
                        max(particles_count / ess0 - 1.0, 0.0) / particles_count
                    )

        if start == "prior":
            positions = prior_positions_of(prior_key)
            log_weights = np.zeros(particles_count, dtype=float)
            beta0 = 0.0
            log_evidence = 0.0
            log_evidence_variance = 0.0
            proposal_ess = float(particles_count)

        self.info(
            "SMC: starting at tempering %.3f with ESS %.0f/%d (%s start)",
            beta0,
            log_ess(log_weights),
            particles_count,
            start,
        )

        if mpi:
            return self._run_smc_sharded(
                log_prior_fn=log_prior_fn,
                log_likelihood_fn=log_likelihood_fn,
                transform_of=transform_of,
                positions=positions,
                log_weights=log_weights,
                log_like=log_like,
                beta0=beta0,
                log_evidence=log_evidence,
                log_evidence_variance=log_evidence_variance,
                polish=polish,
                covariance_np=covariance_np,
                batched_likelihood=batched_likelihood,
                local_slice=local_slice,
                particles_count=particles_count,
                start=start,
                proposal_ess=proposal_ess,
            )

        # The HMC metric is the Laplace covariance, so the posterior is about
        # isotropic in the metric and a step of order one works. The trajectory
        # is a fixed number of leapfrogs: a NUTS trajectory adapts to the local
        # geometry but its worst case is thousands of gradients per draw.
        raw_hmc = hmc_kernel.build_kernel()

        def mcmc_step_fn(
            stage_key: Array,
            state: t.Any,
            logdensity_fn: t.Callable[[Array], Array],
            step_size: Array,
        ) -> t.Tuple[t.Any, t.Any]:
            """One fixed-length HMC transition.

            The inverse mass matrix is closed over instead of passed as an
            MCMC parameter: it is constant, so keeping it in the trace would
            make the kernel recompute its factorisation on every call.

            Returns
            -------
            :obj:`tuple`
                The new HMC state and the HMC transition information

            """
            return raw_hmc(
                stage_key,
                state,
                logdensity_fn,
                step_size,
                metric,
                self.smc_hmc_steps,
            )

        smc_kernel = smc_tempered.build_kernel(
            log_prior_fn,
            log_likelihood_fn,
            mcmc_step_fn,
            hmc_kernel.init,
            systematic,
            batch_size=vectorised_batch,
        )
        # ``ess_solver`` multiplies this by the particle count, so it is the
        # fraction of the particles that must stay effective, not a count.
        target_ess = self.smc_target_ess
        metric = jnp.asarray(covariance_np, dtype=dtype)

        def one_stage(
            stage_key: Array,
            state: t.Any,
            step_size: Array,
            remaining: Array,
        ) -> t.Tuple[t.Any, t.Any]:
            """Advance the particles by one adaptive tempering stage."""
            delta = smc_ess.ess_solver(
                batched_likelihood,
                state.particles,
                target_ess,
                remaining,
                smc_solver.dichotomy,
            )
            lower = jnp.minimum(
                jnp.asarray(self.smc_min_delta, dtype=dtype), remaining
            )
            delta = jnp.clip(delta, lower, remaining)
            return smc_kernel(
                stage_key,
                state,
                self.num_mcmc_steps,
                state.tempering_param + delta,
                {
                    "step_size": step_size,
                },
            )

        one_stage = jax.jit(one_stage, device=self.device)

        weights = np.exp(log_weights - scipy_special.logsumexp(log_weights))
        state = smc_tempered.TemperedSMCState(
            jnp.asarray(positions, dtype=dtype),
            jnp.asarray(weights, dtype=dtype),
            jnp.asarray(beta0, dtype=dtype),
        )
        # Dual averaging on the log step size, driven by the acceptance rate
        # of the stage that just finished. The tempering parameter changes
        # between stages, so the step size follows the narrowing target rather
        # than being tuned once.
        log_step = np.log(self.smc_step_size)
        log_step_bar = log_step
        h_bar = 0.0
        mu = np.log(10.0 * self.smc_step_size)
        t0, gamma = 10.0, 0.05

        beta = beta0
        stage = 0
        accepted = 0.0
        ess_value = log_ess(log_weights)
        started = time.perf_counter()
        if polish:
            # The particles were resampled from the proposal weights, so they
            # approximate the posterior; one HMC transition at beta = 1
            # decorrelates the duplicates without changing the target. The
            # tempered kernel with a zero increment does exactly that.
            key, stage_key = jax.random.split(key)
            step_size = jnp.asarray([np.exp(log_step)], dtype=dtype)
            state, info = one_stage(
                stage_key, state, step_size, jnp.asarray(0.0, dtype=dtype)
            )
            state, info = jax.block_until_ready((state, info))
            stage = 1
            ess_value = 1.0 / float(jnp.sum(state.weights**2))
            accepted = float(jnp.mean(info.update_info.acceptance_rate))
            self.info(
                "SMC: polished the resampled proposal with one HMC transition "
                "at beta=1 (acceptance %.2f)",
                accepted,
            )
        while beta < 1.0 and stage < self.smc_max_steps:
            stage += 1
            key, stage_key = jax.random.split(key)
            # The ESS solver is allowed to ask for at most this much of the
            # remaining temperature range. Capping the increment is what keeps
            # a weak likelihood from jumping straight to beta = 1 after one
            # round of transitions.
            remaining = jnp.asarray(
                min(self.smc_max_delta, 1.0 - beta), dtype=dtype
            )
            step_size = jnp.asarray([np.exp(log_step)], dtype=dtype)
            state, info = one_stage(stage_key, state, step_size, remaining)
            # Waiting here makes the timings meaningful and the numbers below
            # readable; the state is needed by the next stage anyway.
            state, info = jax.block_until_ready((state, info))

            beta = float(state.tempering_param)
            increment = float(info.log_likelihood_increment)
            log_evidence += increment

            # The normalising constant of a stage is estimated from N weighted
            # particles, so its relative variance is (N / ESS - 1) / N. The
            # stage estimates multiply, which adds their log variances.
            ess_value = 1.0 / float(jnp.sum(state.weights**2))
            log_evidence_variance += max(
                particles_count / max(ess_value, 1.0) - 1.0, 0.0
            ) / particles_count

            acceptance = info.update_info.acceptance_rate
            accepted = float(jnp.mean(acceptance))

            step_number = float(stage)
            eta = 1.0 / (step_number + t0)
            h_bar = (1.0 - eta) * h_bar + eta * (_SMC_TARGET_ACCEPT - accepted)
            log_step = mu - np.sqrt(step_number) / gamma * h_bar
            log_step_bar = (
                1.0 - 1.0 / step_number
            ) * log_step_bar + (1.0 / step_number) * log_step
            log_step = float(
                np.clip(min(log_step, log_step_bar), np.log(1e-8), np.log(1e2))
            )

            self.info(
                "SMC: stage %d, tempering %.4f, ESS %.0f/%d, acceptance %.2f, "
                "step size %.3g, log Z %.3f, %.0f s elapsed",
                stage,
                beta,
                ess_value,
                particles_count,
                accepted,
                float(np.exp(log_step)),
                log_evidence,
                time.perf_counter() - started,
            )

        if beta < 1.0:
            self.warning(
                "SMC stopped after %d stages with the tempering parameter at "
                "%.4f instead of 1, so these particles and log Z %.3f "
                "describe the tempered target, not the posterior. Raise "
                "smc_max_steps or the particle count",
                stage,
                beta,
                log_evidence,
            )

        self._log_evidence = log_evidence
        self._log_evidence_error = float(np.sqrt(log_evidence_variance))
        weights = np.asarray(state.weights, dtype=float)
        weights = weights / weights.sum()
        gradient_evaluations = (
            stage
            * self.num_mcmc_steps
            * particles_count
            * self.smc_hmc_steps
        )
        self._smc_parameters = {
            "start": start,
            "beta0": beta0,
            "proposal_ess": proposal_ess,
            "stages": stage,
            "gradient_evaluations": gradient_evaluations,
            "batch_size": batch,
            "hmc_steps": self.smc_hmc_steps,
            "polish": polish,
            "ranks": size,
            "particles": particles_count,
        }

        # The particles carry the final weights; the transform is applied to
        # all of them in one call, and the weights are returned alongside them
        # rather than resampling away the information the SMC already has.
        theta = transform_of(state.particles)
        samples = np.asarray(theta[:, : self._n_physical]).copy()
        self.info(
            "SMC: finished after %d stages in %.0f s, %d model gradients, "
            "log Z = %.3f +/- %.3f, final ESS %.0f/%d (%.1f%%)",
            stage,
            time.perf_counter() - started,
            gradient_evaluations,
            log_evidence,
            self._log_evidence_error,
            ess_value,
            particles_count,
            100.0 * ess_value / particles_count,
        )
        return samples, weights

    def _run_smc_sharded(
        self,
        log_prior_fn: t.Callable[[Array], Array],
        log_likelihood_fn: t.Callable[[Array], Array],
        transform_of: t.Callable[[Array], Array],
        positions: npt.NDArray[np.float64],
        log_weights: npt.NDArray[np.float64],
        log_like: t.Optional[npt.NDArray[np.float64]],
        beta0: float,
        log_evidence: float,
        log_evidence_variance: float,
        polish: bool,
        covariance_np: npt.NDArray[np.float64],
        batched_likelihood: t.Callable[[Array], Array],
        local_slice: slice,
        particles_count: int,
        start: str,
        proposal_ess: float,
    ) -> t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Run the SMC stages with the particle axis split over MPI ranks.

        The particle state itself is small (a few thousand floats), so it is
        replicated on every rank and only the expensive part is sharded: each
        rank evaluates the likelihood and the HMC gradients of its contiguous
        chunk, then an allgather reconstructs the global likelihood and
        position arrays. Weights, the effective sample size, the tempering
        increment, the resampling indices and the evidence are therefore
        identical on every rank, and the stage decisions need no collective of
        their own.

        The mode fit is not sharded: its L-BFGS chain is sequential. Rank 0
        runs it and broadcasts the mode and covariance before this method is
        called.

        Parameters
        ----------
        log_prior_fn:
            Log prior density of one unconstrained vector

        log_likelihood_fn:
            Log likelihood of one unconstrained vector

        transform_of:
            Compiled map from ``u`` to the parameters

        positions:
            Global particle positions, shape ``(particles, dimension)``

        log_weights:
            Global unnormalised log weights of the proposal

        log_like:
            Global log likelihood at ``positions``, or None when it still has
            to be evaluated (the prior start)

        beta0:
            Starting tempering parameter

        log_evidence, log_evidence_variance:
            Evidence estimate and variance accumulated before the stages

        polish:
            Whether the run only resamples and moves once at ``beta = 1``

        covariance_np:
            HMC metric

        batched_likelihood:
            Local-chunk likelihood map

        local_slice:
            This rank's contiguous slice of the global particle axis

        particles_count:
            Global number of particles, divisible by the rank count

        start:
            Proposal description for the diagnostics

        proposal_ess:
            Proposal effective sample size for the diagnostics

        Returns
        -------
        samples, weights:
            Global posterior draws and their normalised weights

        """
        from blackjax.mcmc import hmc as hmc_kernel
        from mpi4py import MPI
        from scipy import special as scipy_special

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        chunk = particles_count // size
        dtype = jnp.float64

        def gather(local: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            """Allgather equal contiguous shards back into global order."""
            pieces = comm.allgather(np.ascontiguousarray(local))
            return np.concatenate(pieces, axis=0)

        def log_ess(values: npt.NDArray[np.float64]) -> float:
            """Effective sample size of an array of unnormalised log weights."""
            return float(
                np.exp(
                    2.0 * scipy_special.logsumexp(values)
                    - scipy_special.logsumexp(2.0 * values)
                )
            )

        def likelihood_of(
            global_positions: npt.NDArray[np.float64],
        ) -> npt.NDArray[np.float64]:
            """Log likelihood of the global particles, one shard per rank."""
            local = np.asarray(
                jax.block_until_ready(
                    batched_likelihood(
                        jnp.asarray(global_positions[local_slice])
                    )
                )
            )
            return gather(local)

        def resample_indices(
            weights: npt.NDArray[np.float64], seed_offset: int
        ) -> npt.NDArray[np.int64]:
            """Systematic resampling, computed once by rank 0 and shared."""
            if rank == 0:
                rng = np.random.default_rng(
                    (0 if self.seed is None else int(self.seed)) + seed_offset
                )
                points = (
                    rng.random() + np.arange(particles_count)
                ) / particles_count
                cumulative = np.cumsum(weights)
                cumulative[-1] = 1.0
                indices = np.searchsorted(cumulative, points)
                indices = np.minimum(indices, particles_count - 1)
            else:
                indices = None
            return comm.bcast(indices, root=0)

        def choose_delta(
            log_likelihood: npt.NDArray[np.float64],
            target: float,
            max_delta: float,
        ) -> float:
            """Tempering increment that leaves ``target`` particles effective."""
            target_log = np.log(target * particles_count)

            def objective(delta: float) -> float:
                return log_ess(delta * log_likelihood) - target_log

            if objective(max_delta) >= 0.0:
                return max_delta
            if objective(0.0) <= 0.0:
                return 0.0
            low, high = 0.0, max_delta
            for _ in range(60):
                middle = 0.5 * (low + high)
                if objective(middle) > 0.0:
                    low = middle
                else:
                    high = middle
            return low

        raw_hmc = hmc_kernel.build_kernel()
        metric = jnp.asarray(covariance_np, dtype=dtype)

        def local_transition(
            key: Array,
            local_positions: Array,
            beta: Array,
            step_size: Array,
        ) -> t.Tuple[Array, Array]:
            """Move this rank's shard with fixed-length HMC transitions."""

            def one(particle: Array, particle_key: Array) -> t.Tuple[Array, Array]:
                def logdensity(u: Array) -> Array:
                    return log_prior_fn(u) + beta * log_likelihood_fn(u)

                state = hmc_kernel.init(particle, logdensity)

                def step(carry: t.Any, subkey: Array) -> t.Tuple[t.Any, Array]:
                    return raw_hmc(
                        subkey,
                        carry,
                        logdensity,
                        step_size,
                        metric,
                        self.smc_hmc_steps,
                    )

                state, infos = jax.lax.scan(
                    step,
                    state,
                    jax.random.split(particle_key, self.num_mcmc_steps),
                )
                return state.position, jnp.mean(infos.acceptance_rate)

            return jax.vmap(one)(
                local_positions,
                jax.random.split(key, local_positions.shape[0]),
            )

        local_transition = jax.jit(local_transition, device=self.device)

        weights = np.exp(log_weights - scipy_special.logsumexp(log_weights))
        if log_like is None:
            log_like = likelihood_of(positions)
        beta = float(beta0)
        log_z = float(log_evidence)
        log_z_variance = float(log_evidence_variance)
        stage = 0
        accepted = 0.0
        ess_value = log_ess(log_weights)
        log_step = np.log(self.smc_step_size)
        log_step_bar = log_step
        h_bar = 0.0
        mu = np.log(10.0 * self.smc_step_size)
        t0, gamma = 10.0, 0.05
        key = jax.random.PRNGKey(0 if self.seed is None else int(self.seed))
        started = time.perf_counter()

        if polish:
            indices = resample_indices(weights, 1)
            positions = positions[indices]
            key, stage_key = jax.random.split(key)
            stage_key = jax.random.fold_in(stage_key, rank)
            step_size = jnp.asarray(np.exp(log_step))
            local_new, local_acceptance = local_transition(
                stage_key,
                jnp.asarray(positions[local_slice]),
                jnp.asarray(1.0, dtype=dtype),
                step_size,
            )
            positions = gather(np.asarray(local_new))
            weights = np.full(particles_count, 1.0 / particles_count)
            ess_value = float(particles_count)
            stage = 1
            accepted = float(
                comm.allreduce(
                    float(jnp.mean(local_acceptance)) * chunk, op=MPI.SUM
                )
            ) / particles_count
            if rank == 0:
                self.info(
                    "SMC: polished the resampled proposal with one HMC "
                    "transition at beta=1 across %d ranks (acceptance %.2f)",
                    size,
                    accepted,
                )

        while beta < 1.0 and stage < self.smc_max_steps:
            stage += 1
            remaining = min(self.smc_max_delta, 1.0 - beta)
            delta = choose_delta(log_like, self.smc_target_ess, remaining)
            if self.smc_min_delta > 0.0:
                delta = max(delta, min(self.smc_min_delta, remaining))
            indices = resample_indices(weights, 104729 * stage)
            positions = positions[indices]
            key, stage_key = jax.random.split(key)
            stage_key = jax.random.fold_in(stage_key, rank)
            step_size = jnp.asarray(np.exp(log_step))
            local_new, local_acceptance = local_transition(
                stage_key,
                jnp.asarray(positions[local_slice]),
                jnp.asarray(beta, dtype=dtype),
                step_size,
            )
            positions = gather(np.asarray(local_new))
            log_like = likelihood_of(positions)
            log_increment = delta * log_like
            log_z += float(
                scipy_special.logsumexp(log_increment)
                - np.log(particles_count)
            )
            weights = np.exp(
                log_increment - scipy_special.logsumexp(log_increment)
            )
            ess_value = 1.0 / float(np.sum(weights**2))
            log_z_variance += (
                max(particles_count / max(ess_value, 1.0) - 1.0, 0.0)
                / particles_count
            )
            accepted = float(
                comm.allreduce(
                    float(jnp.mean(local_acceptance)) * chunk, op=MPI.SUM
                )
            ) / particles_count

            step_number = float(stage)
            eta = 1.0 / (step_number + t0)
            h_bar = (1.0 - eta) * h_bar + eta * (
                _SMC_TARGET_ACCEPT - accepted
            )
            log_step = mu - np.sqrt(step_number) / gamma * h_bar
            log_step_bar = (
                1.0 - 1.0 / step_number
            ) * log_step_bar + (1.0 / step_number) * log_step
            log_step = float(
                np.clip(min(log_step, log_step_bar), np.log(1e-8), np.log(1e2))
            )
            beta += delta
            if rank == 0:
                self.info(
                    "SMC: stage %d, tempering %.4f, ESS %.0f/%d, acceptance "
                    "%.2f, step size %.3g, log Z %.3f, %.0f s elapsed "
                    "(%d ranks)",
                    stage,
                    beta,
                    ess_value,
                    particles_count,
                    accepted,
                    float(np.exp(log_step)),
                    log_z,
                    time.perf_counter() - started,
                    size,
                )

        if beta < 1.0 and rank == 0:
            self.warning(
                "SMC stopped after %d stages with the tempering parameter at "
                "%.4f instead of 1, so these particles and log Z %.3f "
                "describe the tempered target, not the posterior. Raise "
                "smc_max_steps or the particle count",
                stage,
                beta,
                log_z,
            )

        self._log_evidence = log_z
        self._log_evidence_error = float(np.sqrt(log_z_variance))
        weights = weights / weights.sum()
        gradient_evaluations = (
            stage
            * self.num_mcmc_steps
            * particles_count
            * self.smc_hmc_steps
        )
        self._smc_parameters = {
            "start": start,
            "beta0": beta0,
            "proposal_ess": proposal_ess,
            "stages": stage,
            "gradient_evaluations": gradient_evaluations,
            "batch_size": chunk,
            "hmc_steps": self.smc_hmc_steps,
            "polish": polish,
            "ranks": size,
            "particles": particles_count,
        }
        theta = transform_of(jnp.asarray(positions))
        samples = np.asarray(theta[:, : self._n_physical]).copy()
        if rank == 0:
            self.info(
                "SMC: finished after %d stages in %.0f s on %d ranks, %d "
                "model gradients (%d per rank), log Z = %.3f +/- %.3f, final "
                "ESS %.0f/%d (%.1f%%)",
                stage,
                time.perf_counter() - started,
                size,
                gradient_evaluations,
                gradient_evaluations // size,
                log_z,
                self._log_evidence_error,
                ess_value,
                particles_count,
                100.0 * ess_value / particles_count,
            )
        return samples, weights

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
            blackjax.nuts,
            logdensity_v,
            is_mass_matrix_diagonal=not self.dense_mass_matrix,
            max_num_doublings=self.max_num_doublings,
        )
        self.info("NUTS: adapting the step size over %d warm-up steps", self.num_warmup)
        (state, parameters), _ = warmup.run(
            warmup_key,
            jnp.zeros(dimension, dtype=covariance_u.dtype),
            num_steps=self.num_warmup,
        )
        self._nuts_parameters = {
            "step_size": float(parameters["step_size"]),
            "inverse_mass_matrix": np.asarray(parameters["inverse_mass_matrix"]).copy(),
            "max_num_doublings": self.max_num_doublings,
            "dense_mass_matrix": self.dense_mass_matrix,
        }
        algorithm = blackjax.nuts(
            logdensity_v,
            # The window adaptation returns every algorithm parameter,
            # including the trajectory budget it was given, so it has to be
            # filtered out before being passed alongside the explicit one.
            **{
                name: value
                for name, value in parameters.items()
                if name != "max_num_doublings"
            },
            max_num_doublings=self.max_num_doublings,
        )

        steps_per_chain = max(1, self.num_samples // self.num_chains)
        self.info(
            "NUTS: %d chains x %d steps, step size %.3g after %d warm-up steps, "
            "at most %d leapfrog steps per draw",
            self.num_chains,
            steps_per_chain,
            self._nuts_parameters["step_size"],
            self.num_warmup,
            2**self.max_num_doublings,
        )
        chain_keys = jax.random.split(sample_key, self.num_chains)
        step_keys = jax.vmap(lambda k: jax.random.split(k, steps_per_chain))(chain_keys)

        def run_block(
            states: t.Any, keys: t.Any
        ) -> t.Tuple[t.Any, t.Tuple[Array, Array]]:
            """Advance every chain by ``keys.shape[1]`` steps in one program.

            The steps of a block are scanned inside a single compiled call, so
            the block boundary costs a compilation and no python work per step.
            Besides the positions, the number of leapfrog steps each draw took
            is returned: that is the count of gradient evaluations, which is
            what a NUTS draw actually costs.
            """

            def one_chain(
                current: t.Any, chain_keys: Array
            ) -> t.Tuple[t.Any, t.Tuple[Array, Array]]:
                def one_step(
                    current: t.Any, step_key: Array
                ) -> t.Tuple[t.Any, t.Tuple[Array, Array]]:
                    current, info = algorithm.step(step_key, current)
                    return current, (current.position, info.num_integration_steps)

                return jax.lax.scan(one_step, current, chain_keys)

            return jax.vmap(one_chain)(states, keys)

        run_block = jax.jit(run_block)
        # Every chain starts from the state the warm-up stopped at; the keys
        # below are the only thing that differs between them.
        states = jax.tree_util.tree_map(
            lambda leaf: jnp.broadcast_to(
                jnp.asarray(leaf), (self.num_chains,) + jnp.shape(leaf)
            ),
            state,
        )
        # The run is split into blocks only so that progress can be reported:
        # each block is a compiled scan over its own steps and all chains, so
        # the sampled sequence is the same as one long scan and nothing runs
        # in python inside the sampler. The price is a compilation per block
        # size, and there are at most three of those in a run: the measuring
        # block, the size chosen from it, and the remainder.
        probe = max(1, steps_per_chain // _NUTS_PROGRESS_PROBE)
        block = probe
        probe_calls = 0
        per_step: t.Optional[float] = None
        blocks: t.List[Array] = []
        leapfrogs = 0
        warned = False
        done = 0
        started = time.perf_counter()
        while done < steps_per_chain:
            size = min(block, steps_per_chain - done)
            call_started = time.perf_counter()
            states, (block_positions, block_leapfrogs) = run_block(
                states, step_keys[:, done : done + size]
            )
            # Waiting here is what makes the timing below meaningful; the
            # positions are needed anyway, so it costs nothing extra.
            blocks.append(jax.block_until_ready(block_positions))
            leapfrogs += int(np.asarray(block_leapfrogs).sum())
            done += size
            call_time = time.perf_counter() - call_started
            if per_step is None:
                probe_calls += 1
                if probe_calls < 2 and done + probe <= steps_per_chain:
                    # This call paid for the compilation of the scan body, so
                    # its time is not the cost of a step. The next one, at the
                    # same size and therefore the same compiled program, is.
                    continue
                per_step = call_time / size
                block = max(
                    probe,
                    min(steps_per_chain, int(_NUTS_PROGRESS_INTERVAL / per_step)),
                )
            per_draw = leapfrogs / (done * self.num_chains)
            if not warned and per_draw >= 0.95 * 2**self.max_num_doublings:
                # Nothing here is a tuning detail: the trajectories are being
                # cut off before they turn, which both biases the draws and
                # makes every draw cost the worst case.
                warned = True
                self.warning(
                    "NUTS: draws average %.0f leapfrog steps, the ceiling set "
                    "by max_num_doublings=%d, so trajectories are truncated "
                    "before they turn around. The step size %.3g is too small "
                    "for the spread of the posterior; a dense mass matrix or a "
                    "larger num_warmup would find a better one",
                    per_draw,
                    self.max_num_doublings,
                    self._nuts_parameters["step_size"],
                )
            self.info(
                "NUTS: %d/%d steps per chain (%.0f%%), %.0f leapfrog steps and "
                "gradients per draw, %.0f s elapsed, about %.0f s left",
                done,
                steps_per_chain,
                100.0 * done / steps_per_chain,
                per_draw,
                time.perf_counter() - started,
                (steps_per_chain - done) * per_step,
            )
        positions_v = jnp.concatenate(blocks, axis=1).reshape(-1, dimension)
        positions = u_map[None, :] + positions_v @ factor.T
        theta = transform_of(positions)
        return np.asarray(theta[:, : self._n_physical]).copy()

    def _run_importance_sampling(
        self,
        objective: t.Callable[[Array], Array],
        u_map: Array,
        covariance: Array,
        variance_scale: float,
        transform_of: t.Callable[[Array], Array],
    ) -> npt.NDArray[np.float64]:
        """Draw posterior samples by reweighting a proposal around the mode.

        A draw from a gradient sampler costs a trajectory of model gradients;
        a draw from a proposal costs one forward pass. When the Laplace
        Gaussian is close to the posterior, drawing proposals from it and
        reweighting each by ``posterior / proposal`` therefore gives posterior
        draws for the price of a few thousand model evaluations instead of a
        few hundred thousand gradients, and the effective sample size measures
        how close the Gaussian was: a nearly uniform weight distribution means
        the Gaussian is the posterior, a collapsed one says it is not and that
        neither the Laplace errors nor these draws describe it.

        The proposal is a Student-t with four degrees of freedom rather than a
        Gaussian, because the weights of a Gaussian proposal are unbounded
        where the posterior has longer tails than the Laplace approximation;
        the polynomial tails of the t make the ratio decay there whatever the
        posterior does. In the whitened coordinates the t has unit scale, which
        gives it twice the Laplace variance.

        Parameters
        ----------
        objective:
            Negative log posterior including the transform Jacobian

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
        dimension = u_map.shape[0]
        covariance_u = covariance * variance_scale
        factor = jnp.linalg.cholesky(covariance_u)
        degrees = 4.0
        num = self.num_samples

        key = jax.random.PRNGKey(0 if self.seed is None else int(self.seed))
        key, normal_key, chi_key = jax.random.split(key, 3)
        normal = jax.random.normal(
            normal_key, (num, dimension), dtype=covariance_u.dtype
        )
        chi = jax.random.chisquare(chi_key, degrees, (num,), dtype=covariance_u.dtype)
        # A t variate is a normal divided by the square root of a chi-square
        # over its degrees of freedom, which is what makes the tails polynomial.
        whitened = normal * jnp.sqrt(degrees / chi)[:, None]

        _, logdet = jnp.linalg.slogdet(covariance_u)

        def log_proposal(v: Array) -> Array:
            """Log density of the proposal at a whitened point."""
            quadratic = jnp.sum(v * v, axis=-1)
            return (
                jax.scipy.special.gammaln(0.5 * (degrees + dimension))
                - jax.scipy.special.gammaln(0.5 * degrees)
                - 0.5 * dimension * jnp.log(degrees * jnp.pi)
                - 0.5 * (degrees + dimension) * jnp.log1p(quadratic / degrees)
                - 0.5 * logdet
            )

        def log_weight(v: Array) -> Array:
            return -objective(u_map + v @ factor.T) - log_proposal(v)

        # The density is evaluated a batch at a time for the same reason the
        # profile bands are: each draw holds several ``(nlayers, nwn)`` arrays,
        # and a few thousand of them at once is gigabytes.
        atmosphere = self.atmosphere
        per_draw = (
            atmosphere.wngrid.dtype.itemsize
            * atmosphere.model.nLayers
            * atmosphere.wngrid.shape[0]
            * 2
        )
        batch = max(1, _PROFILE_BATCH_BYTES // max(per_draw, 1))
        log_weights = np.asarray(
            jax.block_until_ready(jax.lax.map(log_weight, whitened, batch_size=batch))
        )

        weights = np.exp(log_weights - log_weights.max())
        total = float(weights.sum())
        effective = total**2 / float(np.sum(weights**2))
        self.info(
            "Importance sampling: %d proposals, effective sample size %.0f "
            "(%.1f%% of them), largest weight %.3f",
            num,
            effective,
            100.0 * effective / num,
            float(weights.max()) / total,
        )
        if effective < 0.1 * num:
            self.warning(
                "Importance sampling is degenerate: the effective sample size "
                "is %.0f of %d proposals, so these draws carry the information "
                "of %.0f independent points and the weights are dominated by "
                "a few of them; the Laplace Gaussian is not a good description "
                "of this posterior. Raise num_samples, or use sampler='nuts' "
                "for exact draws at the cost of gradients",
                effective,
                num,
                effective,
            )

        # Resampling with replacement turns the weighted proposals into the
        # equal-weight draws the rest of the pipeline expects; the duplication
        # it introduces is the loss of information the effective sample size
        # already reports.
        generator = np.random.default_rng(0 if self.seed is None else int(self.seed))
        picks = generator.choice(num, size=num, replace=True, p=weights / total)
        positions = u_map[None, :] + jnp.asarray(np.asarray(whitened)[picks]) @ factor.T
        theta = transform_of(positions)
        return np.asarray(theta[:, : self._n_physical]).copy()

    def _run_mclmc(
        self,
        objective: t.Callable[[Array], Array],
        u_map: Array,
        covariance: Array,
        variance_scale: float,
        transform_of: t.Callable[[Array], Array],
    ) -> npt.NDArray[np.float64]:
        """Draw posterior samples with Microcanonical Langevin Monte Carlo.

        MCLMC integrates a Hamiltonian flow on a fixed energy shell, with no
        momentum refreshment and no Metropolis test, so one step costs one
        gradient and yields one sample, where a NUTS draw spends tens to
        hundreds of gradients building and probing a trajectory. What it gives
        up is exactness: the discretisation error is ``O(step_size**2)`` rather
        than zero, and nothing here corrects it after the fact - blackjax's
        energy-variance cutoff is off at its default of
        ``desired_energy_var_max_ratio=inf``, so only steps that produce a NaN
        are reverted. The step size the tuning settles on is what keeps the
        energy change per step small, and ``L`` sets how far the flow travels
        before its momentum is redrawn.

        The Laplace covariance whitens the target, which is what lets one step
        size and one ``L`` work across directions whose parameter scales differ
        by orders of magnitude; the tuning therefore only has to find those two
        numbers and not a preconditioner as well.

        Parameters
        ----------
        objective:
            Negative log posterior including the transform Jacobian

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
        import blackjax.mcmc.mclmc as mclmc
        from blackjax.adaptation.mclmc_adaptation import mclmc_find_L_and_step_size

        dimension = u_map.shape[0]
        if dimension < 2:
            raise ValueError(
                "MCLMC needs at least two fitted parameters, because blackjax "
                "refuses a one dimensional target; use sampler='nuts' for a "
                "single parameter"
            )
        covariance_u = covariance * variance_scale
        factor = jnp.linalg.cholesky(covariance_u)

        def logdensity_v(v: Array) -> Array:
            return -objective(u_map + factor @ v)

        key = jax.random.PRNGKey(0 if self.seed is None else int(self.seed))
        key, init_key, tune_key, sample_key = jax.random.split(key, 4)
        steps_per_chain = max(1, self.num_samples // self.num_chains)
        kernel = mclmc.build_kernel()
        start = mclmc.init(
            jnp.zeros(dimension, dtype=covariance_u.dtype), logdensity_v, init_key
        )
        # The whitening is only as accurate as the Hessian it came from, and a
        # covariance that is wrong by a large factor in some direction leaves
        # the whitened target anisotropic: one step size then has to serve the
        # narrowest and the widest direction at once, and a chain of a few
        # hundred steps covers only the narrow ones - the draws collapse around
        # the mode and look like the Laplace Gaussian they were supposed to
        # correct. The diagonal preconditioner costs about a quarter more tuning
        # steps and measures that residual variance from the tuning draws,
        # which makes the geometry isotropic again whatever the whitening got
        # wrong.
        self.info(
            "MCLMC: tuning the step size, trajectory length and diagonal "
            "preconditioner over about a third of %d steps per chain",
            steps_per_chain,
        )
        state, parameters, tuning_steps = mclmc_find_L_and_step_size(
            mclmc_kernel=kernel,
            logdensity_fn=logdensity_v,
            num_steps=max(steps_per_chain, 100),
            state=start,
            rng_key=tune_key,
            diagonal_preconditioning=True,
        )
        step_size, length = jax.block_until_ready(
            (parameters.step_size, parameters.L)
        )
        self.info(
            "MCLMC: step size %.3g and trajectory length %.3g after %d "
            "tuning steps; %d chains x %d steps, one gradient each",
            float(step_size),
            float(length),
            int(tuning_steps),
            self.num_chains,
            steps_per_chain,
        )
        sampler = mclmc.as_top_level_api(
            logdensity_v,
            L=parameters.L,
            step_size=parameters.step_size,
            inverse_mass_matrix=parameters.inverse_mass_matrix,
        )

        def run_block(
            states: t.Any, keys: t.Any
        ) -> t.Tuple[t.Any, t.Tuple[Array, Array]]:
            """Advance every chain by ``keys.shape[1]`` steps in one program.

            Besides the positions, whether each step was clean is returned: a
            step that produced a NaN is reverted by the kernel, so a chain whose
            steps are all reverted stays where it started and its draws are a
            spike at the mode rather than a posterior.
            """

            def one_chain(
                current: t.Any, chain_keys: Array
            ) -> t.Tuple[t.Any, t.Tuple[Array, Array]]:
                def one_step(
                    current: t.Any, step_key: Array
                ) -> t.Tuple[t.Any, t.Tuple[Array, Array]]:
                    current, info = sampler.step(step_key, current)
                    return current, (current.position, info.nonans)

                return jax.lax.scan(one_step, current, chain_keys)

            return jax.vmap(one_chain)(states, keys)

        run_block = jax.jit(run_block)
        # The tuning produced one chain with one momentum. Starting every chain
        # at the tuned position with its own key gives each an independent
        # momentum and a recomputed gradient, which costs one gradient per
        # chain and makes them independent chains instead of four copies of the
        # same deterministic flow.
        init_key, step_key = jax.random.split(sample_key)
        init_keys = jax.random.split(init_key, self.num_chains)
        states = jax.vmap(lambda x, k: mclmc.init(x, logdensity_v, k))(
            jnp.broadcast_to(state.position, (self.num_chains, dimension)), init_keys
        )
        step_keys = jax.vmap(lambda k: jax.random.split(k, steps_per_chain))(
            jax.random.split(step_key, self.num_chains)
        )
        # Every step costs the same here, so unlike NUTS there is no trajectory
        # length to discover: the blocks exist only to report progress, and the
        # size of the first two measures the cost of a step without the
        # compilation folded in.
        probe = max(1, steps_per_chain // _NUTS_PROGRESS_PROBE)
        block = probe
        probe_calls = 0
        per_step: t.Optional[float] = None
        blocks: t.List[Array] = []
        clean = 0
        done = 0
        started = time.perf_counter()
        while done < steps_per_chain:
            size = min(block, steps_per_chain - done)
            call_started = time.perf_counter()
            states, (block_positions, block_clean) = run_block(
                states, step_keys[:, done : done + size]
            )
            blocks.append(jax.block_until_ready(block_positions))
            clean += int(np.asarray(jax.block_until_ready(block_clean)).sum())
            done += size
            call_time = time.perf_counter() - call_started
            if per_step is None:
                probe_calls += 1
                if probe_calls < 2 and done + probe <= steps_per_chain:
                    continue
                per_step = call_time / size
                block = max(
                    probe,
                    min(steps_per_chain, int(_NUTS_PROGRESS_INTERVAL / per_step)),
                )
            self.info(
                "MCLMC: %d/%d steps per chain (%.0f%%), %.0f s elapsed, "
                "about %.0f s left",
                done,
                steps_per_chain,
                100.0 * done / steps_per_chain,
                time.perf_counter() - started,
                (steps_per_chain - done) * per_step,
            )

        # Two numbers decide whether these draws are a posterior or a spike at
        # the mode. Reverted steps say whether the flow is running at all. The
        # spread in the whitened coordinates says whether the walk covered the
        # target: an accurate whitening puts the posterior at a spread of 1 in
        # every coordinate, so a spread far below 1 is a chain that stayed near
        # its starting point, and a spread far above 1 is a posterior wider than
        # the Laplace covariance, which makes the Laplace error bars too small.
        positions_v = jnp.concatenate(blocks, axis=1).reshape(-1, dimension)
        taken = done * self.num_chains
        reverted = taken - clean
        spread = np.std(np.asarray(positions_v), axis=0)
        self.info(
            "MCLMC: %d of %d steps reverted (%.1f%%); the draws span %.2f to "
            "%.2f in the whitened coordinates, where the Laplace Gaussian is 1",
            reverted,
            taken,
            100.0 * reverted / max(taken, 1),
            float(np.min(spread)),
            float(np.max(spread)),
        )
        if reverted > taken // 2:
            self.warning(
                "MCLMC: %d of %d steps produced a NaN and were reverted, so "
                "the chains stayed where they started and these draws are that "
                "starting point, not a posterior. The whole run spans %.2f in "
                "the whitened coordinates, where it should be about 1",
                reverted,
                taken,
                float(np.max(spread)),
            )
        elif float(np.max(spread)) < 0.2:
            self.warning(
                "MCLMC: the draws span only %.2f of the whitened posterior, "
                "whose scale is 1, so the chains moved by a hundredth of their "
                "own width and these draws are not the posterior. Raise "
                "num_samples (the cost is one gradient per step) or check the "
                "step size %.3g above",
                float(np.max(spread)),
                float(step_size),
            )
        elif float(np.median(spread)) > 3.0:
            self.warning(
                "MCLMC: the posterior is %.1f times wider than the Laplace "
                "covariance along its median direction, so the Laplace error "
                "bars understate it; the draws here are the ones to use",
                float(np.median(spread)),
            )

        positions = u_map[None, :] + positions_v @ factor.T
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
        from taurex import mpi

        if self._samples is None:
            return super().generate_profiles(solution, binning)

        if self.atmosphere is None:
            # A rank that did not run the fit has no compiled atmosphere, so
            # it takes the bands the fitting rank computed. These are a few
            # vectors the length of the profile and spectrum grids, which is
            # cheaper to send than to re-evaluate the draws here.
            return mpi.broadcast(None)

        atmosphere = self.atmosphere
        # The draws are evaluated inside the compiled program, in a batch
        # bounded by a fixed budget. One vmapped call would put every draw's
        # working set in memory at once - several ``(nlayers, nwn)`` arrays
        # each - which is gigabytes for a few thousand draws of a normal
        # atmosphere, all of it for four small variance vectors.
        # :func:`jax.lax.map` loops instead, so only a batch is resident, and
        # sizing the batch from the model rather than fixing a count keeps the
        # working set bounded whatever the atmosphere looks like. The loop
        # stays inside the compiled program, which a python loop over batches
        # does not.
        per_draw = (
            atmosphere.wngrid.dtype.itemsize
            * atmosphere.model.nLayers
            * atmosphere.wngrid.shape[0]
            * _PROFILE_ARRAYS_PER_DRAW
        )
        batch = max(1, _PROFILE_BATCH_BYTES // max(per_draw, 1))

        def profile_of(draws: Array) -> t.Tuple[Array, Array, Array, Array]:
            return jax.lax.map(atmosphere.profile_state, draws, batch_size=batch)

        theta = jnp.asarray(self._samples, dtype=jnp.float64)
        temperature, active, inactive, depth = jax.block_until_ready(
            jax.jit(profile_of, device=self.device)(theta)
        )

        # The SMC particles carry weights, so the band is the weighted
        # standard deviation of the same draws; for equal weights this is the
        # population standard deviation the chain samplers already used.
        weights = jnp.asarray(self.get_weights(0), dtype=jnp.float64)

        def weight_of(values: Array) -> Array:
            """Reshape the weights so they broadcast over the draw axis."""
            return weights.reshape((-1,) + (1,) * (values.ndim - 1))

        def weighted_std(values: Array) -> Array:
            """Standard deviation of the draws along the draw axis."""
            w = weight_of(values)
            mean = jnp.sum(w * values, axis=0)
            variance = jnp.sum(w * (values - mean) ** 2, axis=0)
            return jnp.sqrt(variance)

        profile_dict = {
            "temp_profile_std": np.asarray(weighted_std(temperature)),
            "active_mix_profile_std": np.asarray(weighted_std(active)),
            "inactive_mix_profile_std": np.asarray(weighted_std(inactive)),
        }
        spectrum_dict = {"native_std": np.asarray(weighted_std(depth))}
        if atmosphere.binning_matrix is not None:
            # One batched product instead of a vmap over the draws: the
            # operator is a plain matrix so there is nothing to map.
            binned = jnp.matmul(depth, atmosphere.binning_matrix.T)
            spectrum_dict["binned_std"] = np.asarray(weighted_std(binned))
        result = (profile_dict, spectrum_dict)
        if mpi.nprocs() > 1:
            mpi.broadcast(result)
        return result

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
    def smc_parameters(self) -> t.Optional[t.Dict[str, t.Any]]:
        """How the SMC run spent its budget: start, beta0, stages, gradients."""
        return self._smc_parameters

    @property
    def fisher(self) -> t.Optional[npt.NDArray[np.float64]]:
        """Cached Fisher information, filled in by :meth:`fisher_information`."""
        return self._fisher

    @staticmethod
    def _is_usable_covariance(covariance: t.Any) -> bool:
        """Whether a matrix can serve as a covariance and HMC metric.

        Parameters
        ----------
        covariance:
            Candidate matrix

        Returns
        -------
        bool
            True when it is finite, symmetric and positive definite

        """
        matrix = np.asarray(covariance, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            return False
        if not np.all(np.isfinite(matrix)):
            return False
        matrix = 0.5 * (matrix + matrix.T)
        return float(np.min(np.linalg.eigvalsh(matrix))) > 0.0

    @staticmethod
    def _lbfgs_covariance(state: t.Any, dimension: int) -> t.Optional[np.ndarray]:
        """Inverse-Hessian approximation from an optax L-BFGS state.

        A parameter pinned to a prior bound makes the likelihood Hessian of the
        constrained vector non-finite, and the Laplace proposal then cannot be
        built. The L-BFGS history is already there from the mode fit: its
        curvature pairs are finite even at a bound, and running the two-loop
        recursion on them gives a positive definite inverse Hessian without a
        single extra model evaluation. It describes the objective without the
        transform Jacobian, so it is broader than the posterior; for a proposal
        that is the safe direction.

        Parameters
        ----------
        state:
            State returned by the compiled L-BFGS solve

        dimension:
            Number of unconstrained coordinates

        Returns
        -------
        :obj:`numpy.ndarray` or None
            Positive definite approximation, or None when the history is empty

        """
        curvature = state[0]
        count = int(np.asarray(curvature.count))
        memory = int(curvature.diff_params_memory.shape[0])
        pairs = min(count, memory)
        if pairs <= 0:
            return None

        def ordered(values: np.ndarray) -> np.ndarray:
            order = [int((count - 1 - j) % memory) for j in range(pairs)]
            return np.asarray(values, dtype=float)[order]

        steps = ordered(curvature.diff_params_memory)
        gradients = ordered(curvature.diff_updates_memory)
        weights = ordered(curvature.weights_memory)

        curvature_products = np.sum(steps * gradients, axis=1)
        gradient_norms = np.sum(gradients * gradients, axis=1)
        positive = curvature_products > 0.0
        if not np.any(positive):
            return None
        steps = steps[positive]
        gradients = gradients[positive]
        weights = weights[positive]
        gradient_norms = gradient_norms[positive]
        pairs = steps.shape[0]
        if gradient_norms[0] <= 0.0:
            return None
        gamma = float(curvature_products[positive][0] / gradient_norms[0])

        inverse = np.empty((dimension, dimension), dtype=float)
        for column in range(dimension):
            vector = np.zeros(dimension, dtype=float)
            vector[column] = 1.0
            alpha = np.empty(pairs, dtype=float)
            for index in range(pairs):
                alpha[index] = weights[index] * float(steps[index] @ vector)
                vector = vector - alpha[index] * gradients[index]
            result = gamma * vector
            for index in range(pairs - 1, -1, -1):
                beta = weights[index] * float(gradients[index] @ result)
                result = result + steps[index] * (alpha[index] - beta)
            inverse[:, column] = result

        inverse = 0.5 * (inverse + inverse.T)
        if not np.all(np.isfinite(inverse)):
            return None
        eigenvalues, eigenvectors = np.linalg.eigh(inverse)
        floor = max(float(np.max(eigenvalues)) * 1e-10, 1e-12)
        clipped = np.clip(eigenvalues, floor, None)
        return (eigenvectors * clipped) @ eigenvectors.T

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

        Notes
        -----
        Only the Hessian itself is a device computation. The matrix it
        produces is ``nparams`` by ``nparams``, so the scaling, the ridge and
        the inversion are done in numpy on the host: each operation of a few
        hundred entries costs more to dispatch on the device than to compute,
        and leaving them in JAX compiles one small program per operation.

        """
        hessian = np.asarray(hessian_of(u_map))
        hessian = 0.5 * (hessian + hessian.T)

        largest = float(np.max(np.abs(hessian)))
        if largest == 0.0 or not np.isfinite(largest):
            # A flat posterior is not an error: there is simply nothing for the
            # data to constrain, so every sample is the optimum itself.
            size = hessian.shape[0]
            return np.eye(size, dtype=hessian.dtype), 0.0

        scaled = hessian / largest
        diagonal = np.diagonal(scaled)
        # The regularisation is chosen so that the scaled matrix is strictly
        # diagonally dominant, which by Gershgorin's theorem makes it positive
        # definite whatever the numerical noise in the Hessian did to its
        # spectrum. Without it a flat or rank deficient Hessian produces a
        # matrix that cannot be factored at all.
        off_diagonal = np.sum(np.abs(scaled), axis=1) - np.abs(diagonal)
        shift = off_diagonal + np.clip(-diagonal, 0.0, None) + self.eigenvalue_floor
        regularized = scaled + np.diag(shift)

        covariance = np.linalg.inv(regularized)
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
        jacobian = np.asarray(jacobian_of(u_map))
        full = jacobian @ (np.asarray(covariance) * variance_scale) @ jacobian.T
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
        seed = 0 if self.seed is None else int(self.seed)
        # The draws are built on the host: a Gaussian with a handful of
        # columns costs more to dispatch on the device than to compute, and
        # the compiled transform below is the only part that needs to be a
        # program of its own. Drawing the deviates through
        # :func:`jax.random.normal` outside a compiled function compiles the
        # whole bit generator chain for two thousand by four numbers, which
        # costs about a second on the first call.
        factor = np.linalg.cholesky(np.asarray(covariance))
        normal = np.random.default_rng(seed).standard_normal(
            (self.num_samples, u_map.shape[0])
        )
        offset = normal @ factor.T * float(np.sqrt(variance_scale))
        draws = np.asarray(u_map)[None, :] + offset
        # One traced call for all the samples: the transform is a handful of
        # scalar operations per parameter, so a python loop over two thousand
        # draws would be pure dispatch overhead.
        theta = transform_of(jnp.asarray(draws))
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
            Samples, their weights and the per-parameter summaries taken from
            them, under the same keys :class:`~taurex.optimizer.multinest.
            MultiNestOptimizer` writes. The samplers here already hold the
            draws: the Laplace approximation makes ``num_samples`` of them
            from the fitted covariance, NUTS and MCLMC make them from their
            chains, and SMC returns its weighted particles. Until now only the
            median of those draws was used and the draws themselves dropped,
            which left the output file with no ``fit_params``/``tracedata``/
            ``weights`` and made ``taurex-plot --plot-posteriors`` (and any
            comparison against a nested-sampling corner plot) impossible.
            Storing them costs one array of ``num_samples`` x ``nparams``
            floats, so no extra likelihood evaluations.

        """
        yield 0, self._map, self._median, [
            ("fit_params", self._fit_params()),
            ("tracedata", self.get_samples(0)),
            ("weights", self.get_weights(0)),
        ]

    def _fit_params(self) -> t.Dict[str, t.Dict[str, t.Any]]:
        """Summarise the samples of every fitted parameter.

        ``value``, ``sigma_m`` and ``sigma_p`` are the 16/50/84 weighted
        quantiles of the draws, computed with
        :func:`~taurex.util.quantile_corner` and read with the same keys
        ``multinest.get_solution`` writes, so the plotter and the parameter
        table need no branch per optimizer. ``laplace_sigma`` additionally
        carries the diagonal of the fitted covariance, which is the error bar
        the approximation itself quotes; comparing it with ``sigma_m`` and
        ``sigma_p`` shows where the prior transform has bent the draws away
        from the Gaussian.

        Returns
        -------
        :obj:`dict`
            Mapping of parameter name to its summary

        """
        samples = self.get_samples(0)
        weights = self.get_weights(0)
        covariance = self._covariance
        summary = {}
        for idx, param_name in enumerate(self.fit_names):
            trace = samples[:, idx]
            q_16, q_50, q_84 = quantile_corner(
                trace,
                [0.16, 0.5, 0.84],
                weights=weights,
            )
            summary[param_name] = {
                "value": q_50,
                "sigma_m": q_50 - q_16,
                "sigma_p": q_84 - q_50,
                "mean": np.average(trace, weights=weights),
                "laplace_sigma": (
                    float(np.sqrt(covariance[idx, idx]))
                    if covariance is not None
                    else float("nan")
                ),
                "trace": trace,
            }
        return summary

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

        The Laplace, NUTS and MCLMC draws are already samples from the
        posterior, so they carry equal weight. The SMC particles carry the
        weights the run ended with - the proposal weights when the shortcut
        applied, or the last stage's weights after tempering - and those are
        returned so the reported quantiles use them instead of pretending the
        particles are equally probable.

        Parameters
        ----------
        solution_id:
            Solution index, only 0 exists

        Returns
        -------
        :obj:`numpy.ndarray`
            Normalised weights

        """
        samples = self.get_samples(solution_id)
        if self._weights is not None:
            return self._weights
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
    def log_evidence(self) -> t.Optional[float]:
        """Log Bayesian evidence, when the sampler estimates one.

        Only ``sampler='smc'`` returns this: the chain samplers normalise the
        posterior internally and drop the constant. ``None`` when no evidence
        was computed.
        """
        return self._log_evidence

    @property
    def log_evidence_error(self) -> t.Optional[float]:
        """Estimated standard error of :attr:`log_evidence`, or None.

        It is the quadrature sum of each SMC stage's relative variance of its
        normalising constant, which is what the particle weights of that stage
        can measure. It reports the Monte Carlo error of the evidence estimate
        and not the error of the model.
        """
        return self._log_evidence_error

    @property
    def log_likelihood_map(self) -> t.Optional[float]:
        """Log likelihood at the maximum a posteriori point."""
        return self._log_likelihood

    def generate_solution(
        self, output_size: OutputSize = OutputSize.heavy
    ) -> t.Dict[str, t.Any]:
        """Generate the solution dictionary, adding the evidence when SMC ran.

        The nested samplers write ``GlobalStats`` with their evidence, so the
        SMC evidence is placed under the same key and the output file has one
        shape whichever sampler produced it.
        """
        solution = super().generate_solution(output_size=output_size)
        if self._log_evidence is not None:
            solution["GlobalStats"] = {
                "global evidence": self._log_evidence,
                "global evidence error": self._log_evidence_error,
            }
        return solution

    def write_optimizer(self, output: t.Any) -> t.Any:
        """Write the sampler settings and the evidence to the output file."""
        opt = super().write_optimizer(output)
        opt.write_string("sampler", self.sampler)
        if self._log_evidence is not None:
            opt.write_scalar("log_evidence", self._log_evidence)
            opt.write_scalar("log_evidence_error", self._log_evidence_error)
        if self.sampler == "smc":
            opt.write_scalar("num_particles", self.num_particles)
            opt.write_scalar("num_mcmc_steps", self.num_mcmc_steps)
            opt.write_scalar("smc_target_ess", self.smc_target_ess)
            opt.write_scalar("smc_polish_ess", self.smc_polish_ess)
            opt.write_scalar("smc_max_delta", self.smc_max_delta)
            opt.write_scalar("smc_min_delta", self.smc_min_delta)
            opt.write_scalar("smc_hmc_steps", self.smc_hmc_steps)
            opt.write_scalar("smc_fit_tolerance", self.smc_fit_tolerance)
            opt.write_scalar("smc_max_iterations", self.smc_max_iterations)
            opt.write_string("smc_start", self.smc_start)
            if self._smc_parameters is not None:
                for name in ("beta0", "proposal_ess", "stages"):
                    opt.write_scalar(f"smc_{name}", self._smc_parameters[name])
                opt.write_scalar(
                    "smc_gradient_evaluations",
                    self._smc_parameters["gradient_evaluations"],
                )
        return opt

    @classmethod
    def input_keywords(cls) -> t.Tuple[str, ...]:
        """Return the input file keywords for this optimizer."""
        return ("jax", "jax_laplace", "laplace", "differentiable")
