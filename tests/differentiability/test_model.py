"""Tests for the differentiable transmission forward model.

The central claim is that the JAX port reproduces the numpy model exactly, so
a retrieval driven by its gradient is fitting the same likelihood. These tests
check the spectrum, the optical depth and the log likelihood against the numpy
model, check that the gradient the optimizer follows is the derivative of that
same quantity, and check that the whole forward pass can be traced and compiled
- which is the part of the design that a python level branch or a slice would
silently break.
"""

import numpy as np
import pytest

from taurex.differentiability import Atmosphere

from .conftest import build_model, compile_fit_params


jax = pytest.importorskip("jax")
jnp = jax.numpy


def test_native_spectrum_matches_numpy_model(data_path, taurex_model, observation):
    """The JAX atmosphere reproduces the numpy transmission spectrum."""
    atmosphere = Atmosphere(taurex_model, observation, fit_params=[])
    theta = jnp.zeros(0, dtype=jnp.float64)

    _, depth = atmosphere.native_spectrum(theta)
    _, expected, _, _ = taurex_model.model(wngrid=observation.wavenumberGrid)

    np.testing.assert_allclose(np.asarray(depth), expected, rtol=1e-10, atol=0.0)


def test_transmission_matches_numpy_path_integral(data_path, taurex_model, observation):
    """Optical depth matches the numpy path integral to machine precision."""
    atmosphere = Atmosphere(taurex_model, observation, fit_params=[])
    theta = jnp.zeros(0, dtype=jnp.float64)

    _, tau, _ = atmosphere.forward(theta)
    _, transmission = taurex_model.path_integral(atmosphere.wngrid)

    np.testing.assert_allclose(
        np.asarray(jnp.exp(-tau)), transmission, rtol=1e-10, atol=1e-14
    )


def test_binned_spectrum_matches_flux_binner(data_path, taurex_model, observation):
    """The precomputed binning matrix reproduces the taurex binner."""
    atmosphere = Atmosphere(taurex_model, observation, fit_params=[])
    theta = jnp.zeros(0, dtype=jnp.float64)

    binned = atmosphere.spectrum(theta)

    output = taurex_model.model(wngrid=observation.wavenumberGrid)
    expected = observation.create_binner().bin_model(output)[1]

    np.testing.assert_allclose(np.asarray(binned), np.ravel(expected), rtol=1e-10)


def test_log_likelihood_matches_numpy_optimizer(data_path, taurex_model, observation):
    """The JAX log likelihood equals the one the numpy optimizer computes."""
    from taurex.differentiability import LaplaceOptimizer

    names = ("planet_radius", "T", "H2O")

    optimizer = LaplaceOptimizer()
    optimizer.set_model(taurex_model)
    optimizer.set_observed(observation)
    for name in names:
        optimizer.enable_fit(name)

    theta_values = np.array([1.30, 800.0, -4.0])
    optimizer.update_model(theta_values)
    expected = optimizer.log_likelihood(theta_values)

    atmosphere = Atmosphere(
        taurex_model, observation, fit_params=optimizer.fitting_parameters
    )
    value = float(
        atmosphere.log_likelihood(jnp.asarray(theta_values, dtype=jnp.float64))
    )

    assert value == pytest.approx(expected, rel=1e-10)


def test_gradient_matches_finite_difference(data_path, taurex_model, observation):
    """Autodiff of the chi-squared agrees with central differences."""
    names = ("planet_radius", "T", "H2O")
    fit_params = compile_fit_params(taurex_model, observation, names)
    atmosphere = Atmosphere(taurex_model, observation, fit_params=fit_params)

    start = np.array([1.30, 800.0, -4.0])
    point = jnp.asarray(start, dtype=jnp.float64)
    analytic = np.asarray(jax.grad(atmosphere.chi_squared)(point)).copy()

    steps = np.array([1e-5, 1e-2, 1e-5])
    numeric = np.zeros_like(analytic)
    for index in range(start.shape[0]):
        offset = np.zeros_like(start)
        offset[index] = steps[index]
        plus = jnp.asarray(start + offset, dtype=jnp.float64)
        minus = jnp.asarray(start - offset, dtype=jnp.float64)
        numeric[index] = float(
            atmosphere.chi_squared(plus) - atmosphere.chi_squared(minus)
        ) / (2.0 * steps[index])

    assert np.all(np.abs(analytic) > 0.0)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-6)


def test_forward_pass_compiles(data_path, taurex_model, observation):
    """The whole forward pass, its gradient and its Hessian can be traced.

    A python level branch on a traced value, or a layer index used as a python
    slice, would only show up here: eager execution accepts them and the one
    call to :func:`jax.jit` does not.
    """
    names = ("planet_radius", "T", "H2O")
    fit_params = compile_fit_params(taurex_model, observation, names)
    atmosphere = Atmosphere(taurex_model, observation, fit_params=fit_params)
    point = jnp.array([1.30, 800.0, -4.0], dtype=jnp.float64)

    chi_squared = jax.jit(atmosphere.chi_squared)
    gradient = jax.jit(jax.grad(atmosphere.chi_squared))
    hessian = jax.jit(jax.hessian(atmosphere.chi_squared))

    assert float(chi_squared(point)) == pytest.approx(
        float(atmosphere.chi_squared(point)), rel=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(gradient(point)),
        np.asarray(jax.grad(atmosphere.chi_squared)(point)),
        rtol=1e-12,
    )
    assert np.all(np.isfinite(np.asarray(hessian(point))))


def test_npoint_temperature_compiles_and_matches_numpy(
    data_path, taurex_model, observation
):
    """A layered temperature profile survives tracing and matches numpy.

    The isothermal shortcut in the numpy profile branches on the values of the
    nodes rather than on their shape, which is the one place in the model where
    a python ``if`` had to become a :func:`jax.numpy.where`.
    """
    from taurex.data.profiles.temperature import NPoint

    taurex_model = build_model(
        NPoint(
            T_surface=1200.0,
            T_top=500.0,
            temperature_points=[900.0],
            pressure_points=[1e2],
            smoothing_window=10,
        )
    )

    atmosphere = Atmosphere(taurex_model, observation, fit_params=[])
    theta = jnp.zeros(0, dtype=jnp.float64)

    depth = jax.jit(atmosphere.native_spectrum)(theta)[1]
    _, expected, _, _ = taurex_model.model(wngrid=observation.wavenumberGrid)

    np.testing.assert_allclose(np.asarray(depth), expected, rtol=1e-10, atol=0.0)


def test_native_grid_is_clipped_to_the_observation(data_path, taurex_model, observation):
    """Only the part of the native grid the observation covers is modelled."""
    from taurex.util import clip_native_to_wngrid

    atmosphere = Atmosphere(taurex_model, observation, fit_params=[])
    expected = clip_native_to_wngrid(
        taurex_model.nativeWavenumberGrid, observation.wavenumberGrid
    )

    np.testing.assert_array_equal(atmosphere.wngrid, expected)
    assert atmosphere.wngrid.size < taurex_model.nativeWavenumberGrid.size


def test_chi_squared_is_zero_for_a_perfect_fit(data_path, taurex_model, observation):
    """Data generated by the model has a vanishing chi-squared."""
    atmosphere = Atmosphere(taurex_model, observation, fit_params=[])
    theta = jnp.zeros(0, dtype=jnp.float64)

    atmosphere._data = atmosphere.spectrum(theta)
    value = atmosphere.chi_squared(theta)

    assert float(value) < 1e-20


def test_unsupported_contribution_is_rejected(data_path, taurex_model, observation):
    """A contribution without a differentiable plan fails loudly."""
    from taurex.contributions import LeeMieContribution

    taurex_model.add_contribution(LeeMieContribution())
    with pytest.raises(NotImplementedError, match="LeeMie"):
        Atmosphere(taurex_model, observation, fit_params=[])


def test_unsupported_temperature_profile_is_rejected(
    data_path, taurex_model, observation
):
    """A temperature profile without a differentiable plan fails loudly."""
    from taurex.data.profiles.temperature import Guillot2010

    taurex_model._temperature_profile = Guillot2010()
    atmosphere = Atmosphere(taurex_model, observation, fit_params=[])
    with pytest.raises(NotImplementedError, match="Guillot2010"):
        atmosphere.native_spectrum(jnp.zeros(0, dtype=jnp.float64))
