"""Retrieval that uses the gradient of the forward model.

Nested sampling needs thousands of likelihood evaluations because it has no
gradient to follow. The differentiable atmosphere does have one, so the same
posterior can be located with a quasi-Newton optimiser and then described by a
Laplace approximation around the maximum a posteriori point:

1. the posterior is maximised over an unconstrained vector ``u`` that is mapped
   onto the fitting parameters through their own priors, so the bounds and
   priors from the input file are respected exactly;
2. the Hessian of the negative log posterior at the optimum gives a Gaussian
   covariance that is then transformed back to parameter space, from which
   posterior samples, medians and error bars follow.

That is one forward pass plus one backward pass per iteration instead of one
forward pass per sample, which is the whole point of making the model
differentiable.
"""

import typing as t

import numpy as np
import numpy.typing as npt
import torch

from taurex.core.priors import Prior
from taurex.model import ForwardModel
from taurex.optimizer.optimizer import Optimizer
from taurex.spectrum import BaseSpectrum

from .model import Atmosphere


class PriorTransform:
    """Cube to parameter map built from a taurex prior.

    Mirrors :meth:`taurex.optimizer.optimizer.Optimizer.prior_transform`, which
    evaluates the inverse cumulative distribution of the prior, but in torch so
    that a gradient can flow from the parameters back to the cube.

    Parameters
    ----------
    prior:
        The taurex prior of a fitting parameter

    name:
        Parameter name, used in error messages

    dtype:
        Floating point dtype of the working tensors

    device:
        Device of the working tensors

    """

    def __init__(
        self,
        prior: Prior,
        name: str,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Initialise the transform.

        Parameters
        ----------
        prior:
            The taurex prior of a fitting parameter

        name:
            Parameter name, used in error messages

        dtype:
            Floating point dtype of the working tensors

        device:
            Device of the working tensors

        """
        self.name = name
        self.dtype = dtype
        self.device = device

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

    def forward(self, cube: torch.Tensor) -> torch.Tensor:
        """Map a cube coordinate onto the parameter.

        Parameters
        ----------
        cube:
            Value in ``(0, 1)``

        Returns
        -------
        :obj:`torch.Tensor`
            Parameter value in the space the optimizers use, which is the space
            the prior is defined in

        """
        if self._kind == "uniform":
            return self._low + cube * (self._high - self._low)
        # The inverse normal CDF diverges at the edges of the cube, so the
        # argument is pulled just inside them. The clip sits at about five and
        # a half sigma, far enough out that it does not change a usable prior.
        root_two = np.sqrt(2.0)
        argument = torch.clamp(2.0 * cube - 1.0, -1.0 + 1e-12, 1.0 - 1e-12)
        return self._loc + self._scale * root_two * torch.erfinv(argument)

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

    def log_prior(self, theta: torch.Tensor) -> torch.Tensor:
        """Log density of the prior in parameter space.

        A flat prior contributes a constant, which is dropped: it shifts the
        posterior but neither its mode nor its curvature.

        Parameters
        ----------
        theta:
            Parameter value, possibly carrying a gradient

        Returns
        -------
        :obj:`torch.Tensor`
            Log prior density, up to an additive constant

        """
        if self._kind == "uniform":
            return torch.zeros((), dtype=self.dtype, device=self.device)
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
            Maximum number of L-BFGS optimiser steps

        max_lbfgs_iterations:
            Maximum number of line search evaluations per step

        gradient_tolerance:
            Convergence tolerance on the gradient norm

        eigenvalue_floor:
            Ridge added to the scaled Hessian before inverting it. It bounds
            the variance of the directions the data does not constrain, so
            those parameters come back prior dominated rather than undefined.

        seed:
            Seed for the posterior draws, so a retrieval is reproducible

        device:
            Torch device, defaults to the CPU

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
            "Torch", observed=observed, model=model, sigma_fraction=sigma_fraction
        )
        self.num_samples = int(num_samples)
        self.max_iterations = int(max_iterations)
        self.max_lbfgs_iterations = int(max_lbfgs_iterations)
        self.gradient_tolerance = float(gradient_tolerance)
        self.eigenvalue_floor = float(eigenvalue_floor)
        self.seed = seed
        self.device = torch.device(device or "cpu")
        self.ignored_options = kwargs

        self.atmosphere: t.Optional[Atmosphere] = None
        self._samples: t.Optional[npt.NDArray[np.float64]] = None
        self._map: t.Optional[npt.NDArray[np.float64]] = None
        self._median: t.Optional[npt.NDArray[np.float64]] = None
        self._covariance: t.Optional[npt.NDArray[np.float64]] = None
        self._log_posterior: t.Optional[float] = None
        self._log_likelihood: t.Optional[float] = None

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

        dtype = torch.float64
        atmosphere = Atmosphere(
            self._model,
            self._observed,
            fit_params=fit_params,
            device=self.device,
            dtype=dtype,
        )
        self.atmosphere = atmosphere

        transforms = [
            PriorTransform(param.fit_prior, param.name, dtype, self.device)
            for param in fit_params
        ]
        self.transforms = transforms

        def parameters_of(u: torch.Tensor) -> torch.Tensor:
            """Map the unconstrained vector onto the fitting parameters.

            Works for a single vector and for a batch of them: the parameters
            are always stacked along the last axis.
            """
            cube = torch.sigmoid(u)
            return torch.stack(
                [
                    transform.forward(cube[..., index])
                    for index, transform in enumerate(transforms)
                ],
                dim=-1,
            )

        def negative_log_posterior(u: torch.Tensor) -> torch.Tensor:
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
        optimiser = torch.optim.LBFGS(
            [u],
            max_iter=self.max_iterations,
            max_eval=self.max_lbfgs_iterations * self.max_iterations,
            tolerance_grad=self.gradient_tolerance,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimiser.zero_grad()
            loss = negative_log_posterior(u)
            loss.backward()
            return loss

        optimiser.step(closure)

        u_map = u.detach().clone()
        with torch.no_grad():
            theta_map = parameters_of(u_map)
            self._log_likelihood = float(atmosphere.log_likelihood(theta_map))
            self._log_posterior = float(-negative_log_posterior(u_map))
            self._map = theta_map.cpu().numpy().copy()

        covariance, variance_scale = self._laplace_covariance(
            negative_log_posterior, u_map
        )
        self._samples = self._draw_samples(
            u_map, covariance, variance_scale, parameters_of
        )
        self._median = np.median(self._samples, axis=0)
        self._covariance = self._parameter_covariance(
            covariance, variance_scale, parameters_of, u_map
        )

    def _laplace_covariance(
        self,
        objective: t.Callable[[torch.Tensor], torch.Tensor],
        u_map: torch.Tensor,
    ) -> t.Tuple[torch.Tensor, float]:
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

        Parameters
        ----------
        objective:
            Negative log posterior as a function of ``u``

        u_map:
            Location of the optimum

        Returns
        -------
        covariance:
            Covariance of the scaled Hessian's inverse

        variance_scale:
            Factor that turns that covariance into the covariance of ``u``

        """
        hessian = torch.autograd.functional.hessian(objective, u_map)
        hessian = 0.5 * (hessian + hessian.transpose(0, 1))

        largest = float(hessian.abs().max())
        if largest == 0.0 or not np.isfinite(largest):
            # A flat posterior is not an error: there is simply nothing for the
            # data to constrain, so every sample is the optimum itself.
            size = hessian.shape[0]
            return torch.eye(size, dtype=hessian.dtype), 0.0

        scaled = hessian / largest
        diagonal = scaled.diagonal().clone()
        # The regularisation is chosen so that the scaled matrix is strictly
        # diagonally dominant, which by Gershgorin's theorem makes it positive
        # definite whatever the numerical noise in the Hessian did to its
        # spectrum. Without it a flat or rank deficient Hessian produces a
        # matrix that cannot be factored at all.
        off_diagonal = scaled.abs().sum(dim=1) - diagonal.abs()
        shift = (
            off_diagonal
            + torch.clamp(-diagonal, min=0.0)
            + self.eigenvalue_floor
        )
        regularized = scaled + torch.diag(shift)

        covariance = torch.linalg.inv(regularized)
        covariance = 0.5 * (covariance + covariance.transpose(0, 1))
        return covariance, 1.0 / largest

    def _parameter_covariance(
        self,
        covariance: torch.Tensor,
        variance_scale: float,
        parameters_of: t.Callable[[torch.Tensor], torch.Tensor],
        u_map: torch.Tensor,
    ) -> npt.NDArray[np.float64]:
        """Carry the unconstrained covariance back to parameter space.

        Parameters
        ----------
        covariance:
            Covariance in the scaled unconstrained space

        variance_scale:
            Factor that turns it into the covariance of ``u``

        parameters_of:
            Map from ``u`` to the parameters

        u_map:
            Location of the optimum

        Returns
        -------
        :obj:`numpy.ndarray`
            Covariance in parameter space

        """
        jacobian = torch.autograd.functional.jacobian(parameters_of, u_map)
        full = jacobian @ (covariance * variance_scale) @ jacobian.transpose(0, 1)
        full = 0.5 * (full + full.transpose(0, 1))
        return full.detach().cpu().numpy().copy()

    def _draw_samples(
        self,
        u_map: torch.Tensor,
        covariance: torch.Tensor,
        variance_scale: float,
        parameters_of: t.Callable[[torch.Tensor], torch.Tensor],
    ) -> npt.NDArray[np.float64]:
        """Draw posterior samples from the Gaussian approximation.

        The draws are made in unconstrained space and pushed through the prior
        transforms, so every sample lands inside the support of its prior
        however wide the Gaussian is. The Cholesky factor is taken in the scaled
        space for conditioning and applied to the samples only at the end, which
        avoids ever forming the ill-conditioned covariance of ``u`` itself.

        Parameters
        ----------
        u_map:
            Unconstrained vector at the optimum

        covariance:
            Covariance in the scaled unconstrained space

        variance_scale:
            Factor that turns it into the covariance of ``u``

        parameters_of:
            Map from ``u`` to the parameters

        Returns
        -------
        :obj:`numpy.ndarray`
            Samples with shape ``(num_samples, nparams)``

        """
        generator = torch.Generator()
        if self.seed is not None:
            generator.manual_seed(int(self.seed))

        factor = torch.linalg.cholesky(covariance.cpu())
        normal = torch.randn(
            self.num_samples,
            u_map.shape[0],
            generator=generator,
            dtype=covariance.dtype,
        )
        offset = normal @ factor.transpose(0, 1) * float(np.sqrt(variance_scale))
        draws = u_map.cpu()[None, :] + offset
        with torch.no_grad():
            theta = parameters_of(draws.to(u_map.device))
        return theta.cpu().numpy().copy()

    def _starting_point(
        self, transforms: t.Sequence[PriorTransform], dtype: torch.dtype
    ) -> torch.Tensor:
        """Unconstrained vector that reproduces the current parameter values.

        Parameters
        ----------
        transforms:
            Prior transforms of the fitted parameters

        dtype:
            Floating point dtype to use

        Returns
        -------
        :obj:`torch.Tensor`
            Unconstrained vector with ``requires_grad`` set

        """
        cube = []
        for param, transform in zip(
            self.fitting_parameters, transforms, strict=True
        ):
            value = float(param.fit_value)
            cube.append(min(max(transform.inverse(value), 1e-6), 1.0 - 1e-6))
        cube_tensor = torch.as_tensor(cube, dtype=dtype, device=self.device)
        return torch.logit(cube_tensor).clone().requires_grad_(True)

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
        return ("torch", "laplace", "torch_laplace", "differentiable")
