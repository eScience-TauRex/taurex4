"""Test main contribution function."""

import numpy as np
import pytest

import taurex.contributions.contribution as contrib


def test_contribute_numpy():
    """Tests the numpy version of the contribution function."""
    sigma = np.random.rand(100, 200)
    density = np.random.rand(100)
    path = np.random.rand(100)
    tau = np.zeros((100, 200))
    contrib.contribute_tau_numpy(0, 100, 0, sigma, density, path, 100, 200, 0, tau)

    assert not np.all(tau == 0.0)


def test_contribute_numba():
    """Tests the numba version of the contribution function."""
    numba = pytest.importorskip("numba")  # noqa: F841
    sigma = np.random.rand(100, 200)
    density = np.random.rand(100)
    path = np.random.rand(100)
    tau = np.zeros((100, 200))
    contrib.contribute_tau_numba(0, 100, 0, sigma, density, path, 100, 200, 0, tau)

    assert not np.all(tau == 0.0)


def test_contribute_consistent():
    """Tests the consistency of the numpy and numba contribution functions."""
    numba = pytest.importorskip("numba")  # noqa: F841
    sigma = np.random.rand(100, 200)
    density = np.random.rand(100)

    tau1 = np.zeros((100, 200))
    tau2 = np.zeros((100, 200))

    for x in range(100):
        path = np.random.rand(100 - x)
        contrib.contribute_tau_numpy(
            0,
            100 - x,
            x,
            sigma,
            density,
            path,
            100,
            200,
            x,
            tau1,
        )
        contrib.contribute_tau_numba(
            0,
            100 - x,
            x,
            sigma,
            density,
            path,
            100,
            200,
            x,
            tau2,
        )

    np.testing.assert_array_equal(tau1, tau2)


def test_leemie_no_break():
    """Tests if LeeMie breaks the optimizer."""
    from taurex.contributions import AbsorptionContribution
    from taurex.contributions import LeeMieContribution
    from taurex.model import TransmissionModel

    tm = TransmissionModel(nlayers=100)
    tm.add_contribution(AbsorptionContribution())
    tm.add_contribution(LeeMieContribution())

    assert True


def _synthetic_triangle(nlayers, seed=0):
    """A synthetic path matrix and its ragged per-layer equivalent."""
    rng = np.random.default_rng(seed)
    matrix = np.zeros((nlayers, nlayers))
    for layer in range(nlayers):
        matrix[layer, layer:] = rng.random(nlayers - layer)
    return matrix, [matrix[layer, layer:].copy() for layer in range(nlayers)]


def test_path_matrix_matches_ragged_path_lengths():
    """compute_path_matrix must reproduce compute_path_length_old."""
    from taurex.model import TransmissionModel

    model = TransmissionModel(nlayers=20)
    model.build()
    model.initialize_profiles()

    dz = model.deltaz
    nlayers = model.nLayers

    matrix = model.compute_path_matrix(dz)
    ragged = model.compute_path_length_old(dz)

    assert matrix.shape == (nlayers, nlayers)
    for layer in range(nlayers):
        np.testing.assert_allclose(
            matrix[layer, layer:], ragged[layer], rtol=1e-12, atol=0.0
        )
        np.testing.assert_array_equal(matrix[layer, :layer], 0.0)


def test_path_matrix_accumulation_matches_loop():
    """tau from a single matrix product matches the per-layer kernel."""
    nlayers, ngrid = 20, 32
    matrix, ragged = _synthetic_triangle(nlayers)
    density = np.linspace(0.5, 1.5, nlayers)
    sigma = np.random.rand(nlayers, ngrid)

    tau_matrix = matrix @ (sigma * density[:, None])

    tau_loop = np.zeros((nlayers, ngrid))
    for layer in range(nlayers):
        contrib.contribute_tau_numpy(
            0,
            nlayers - layer,
            layer,
            sigma,
            density,
            ragged[layer],
            nlayers,
            ngrid,
            layer,
            tau_loop,
        )

    np.testing.assert_allclose(tau_matrix, tau_loop, rtol=1e-12, atol=1e-14)


def test_cia_density_squared_matches_kernel():
    """CIAContribution weights the cross-section by density squared."""
    from taurex.contributions.cia import contribute_cia_numpy

    nlayers, ngrid = 20, 32
    matrix, ragged = _synthetic_triangle(nlayers)
    density = np.linspace(0.5, 1.5, nlayers)
    sigma = np.random.rand(nlayers, ngrid)

    tau_matrix = matrix @ (sigma * (density**2)[:, None])

    tau_loop = np.zeros((nlayers, ngrid))
    for layer in range(nlayers):
        contribute_cia_numpy(
            0,
            nlayers - layer,
            layer,
            sigma,
            density,
            ragged[layer],
            nlayers,
            ngrid,
            layer,
            tau_loop,
        )

    np.testing.assert_allclose(tau_matrix, tau_loop, rtol=1e-12, atol=1e-14)


def test_ktau_matrix_matches_loop():
    """The correlated-k reduction matches contribute_ktau_numpy."""
    from taurex.contributions.absorption import contribute_ktau_numpy

    nlayers, ngrid, ngauss = 20, 32, 4
    matrix, ragged = _synthetic_triangle(nlayers)
    density = np.linspace(0.5, 1.5, nlayers)
    sigma = np.random.rand(nlayers, ngrid, ngauss)
    weights = np.full(ngauss, 1.0 / ngauss)

    tau_matrix = contrib.accumulate_ktau_matrix(
        sigma * density[:, None, None], matrix, weights
    )

    tau_loop = np.zeros((nlayers, ngrid))
    for layer in range(nlayers):
        contribute_ktau_numpy(
            0,
            nlayers - layer,
            layer,
            sigma,
            density,
            ragged[layer],
            weights,
            tau_loop,
            ngrid,
            layer,
            ngauss,
        )

    np.testing.assert_allclose(tau_matrix, tau_loop, rtol=1e-12, atol=1e-14)


def test_contributions_declare_accumulation_weight():
    """Every in-tree contribution must declare how it is accumulated."""
    from taurex.contributions import AbsorptionContribution
    from taurex.contributions import CIAContribution
    from taurex.contributions import FlatMieContribution
    from taurex.contributions import LeeMieContribution
    from taurex.contributions import RayleighContribution
    from taurex.contributions import SimpleCloudsContribution

    assert AbsorptionContribution().uses_path_matrix
    assert RayleighContribution().density_power == 1
    assert FlatMieContribution().density_power == 1
    assert CIAContribution().density_power == 2
    assert LeeMieContribution().density_power == 0

    clouds = SimpleCloudsContribution()
    assert clouds.uses_path_matrix
    assert clouds.direct_accumulation


class _FixedContribution(contrib.Contribution):
    """Contribution with a preset cross-section and a declared weight."""

    def __init__(self, sigma):
        """Store the cross-section directly, skipping prepare()."""
        super().__init__("Fixed")
        self.sigma_xsec = sigma


class _CustomContribution(contrib.Contribution):
    """Contribution with its own integration rule, so it keeps the loop."""

    def __init__(self, sigma):
        """Store the cross-section directly, skipping prepare()."""
        super().__init__("Custom")
        self.sigma_xsec = sigma

    def contribute(
        self,
        model,
        start_layer,
        end_layer,
        density_offset,
        layer,
        density,
        tau,
        path_length=None,
    ):
        """Add the cross-section of this layer straight to tau."""
        tau[layer] += self.sigma_xsec[layer]


def test_accumulate_path_matrix_dispatch():
    """Unweighted, standard, density-squared and direct contributions compose."""
    from taurex.model import TransmissionModel

    nlayers, ngrid = 8, 16
    model = TransmissionModel(nlayers=nlayers)

    density = np.linspace(0.5, 1.5, nlayers)
    matrix, _ = _synthetic_triangle(nlayers)
    wngrid = np.arange(ngrid, dtype=float)

    # Unweighted first, so it seeds the accumulator without a density weight.
    unweighted = _FixedContribution(np.random.rand(nlayers, ngrid))
    unweighted.density_power = 0
    standard = _FixedContribution(np.random.rand(nlayers, ngrid))
    squared = _FixedContribution(np.random.rand(nlayers, ngrid))
    squared.density_power = 2
    direct = _FixedContribution(np.random.rand(nlayers, ngrid))
    direct.direct_accumulation = True

    for contribution in (unweighted, standard, squared, direct):
        model.add_contribution(contribution)

    expected = (
        matrix @ unweighted.sigma_xsec
        + matrix @ (standard.sigma_xsec * density[:, None])
        + matrix @ (squared.sigma_xsec * (density**2)[:, None])
        + direct.sigma_xsec
    )

    tau = model.accumulate_path_matrix(wngrid, matrix, density)

    np.testing.assert_allclose(tau, expected, rtol=1e-12, atol=1e-14)


def test_accumulate_path_matrix_keeps_custom_contribute():
    """A contribution with its own rule still runs through the loop."""
    from taurex.model import TransmissionModel

    model = TransmissionModel(nlayers=8)
    model.build()
    model.initialize_profiles()

    nlayers = model.nLayers
    ngrid = 16

    density = model.densityProfile
    wngrid = np.arange(ngrid, dtype=float)
    matrix = model.compute_path_matrix(model.deltaz)

    standard = _FixedContribution(np.random.rand(nlayers, ngrid))
    custom = _CustomContribution(np.random.rand(nlayers, ngrid))

    assert standard.uses_path_matrix
    assert not custom.uses_path_matrix

    model.add_contribution(standard)
    model.add_contribution(custom)

    expected = (
        matrix @ (standard.sigma_xsec * density[:, None]) + custom.sigma_xsec
    )

    tau = model.accumulate_path_matrix(wngrid, matrix, density)

    np.testing.assert_allclose(tau, expected, rtol=1e-12, atol=1e-14)
