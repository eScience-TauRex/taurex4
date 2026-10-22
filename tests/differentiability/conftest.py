"""Fixtures for the differentiability tests.

The fixtures build a small self-contained retrieval: synthetic opacity and CIA
tables in a temporary directory, a small observed spectrum and a transmission
model with absorption, Rayleigh, collision induced absorption and a flat cloud
deck. Keeping the grids small makes the tests run in about a second while still
going through the real opacity, chemistry and contribution code paths.
"""

import pickle  # noqa: S403

import numpy as np
import pytest


WAVENUMBER_GRID = np.linspace(2200.0, 2600.0, 120)
TEMPERATURE_GRID = np.linspace(300.0, 1500.0, 6)
PRESSURE_GRID = np.logspace(6.0, 0.0, 5)


def cross_sections() -> np.ndarray:
    """Temperature and pressure dependent cross-sections in cm^2."""
    temperature = TEMPERATURE_GRID[None, :, None]
    pressure = PRESSURE_GRID[:, None, None]
    wavenumber = WAVENUMBER_GRID[None, None, :]

    band = np.exp(-(((wavenumber - 2350.0) / 60.0) ** 2))
    scale = 1e-18 * band + 1e-21
    # The temperature and pressure dependence is not separable, which is what
    # gives every fitting parameter a distinct spectral signature.
    return scale * (1.0 + 500.0 / temperature) * (1.0 + 1e-6 * pressure)


@pytest.fixture
def data_path(tmp_path):
    """Write synthetic opacity and CIA tables and point taurex at them."""
    import h5py

    from taurex.cache import CIACache
    from taurex.cache import GlobalCache
    from taurex.cache import OpacityCache

    opacity_file = tmp_path / "1H2-16O__synth.R1000_2-5mu.xsec.TauREx.h5"
    with h5py.File(opacity_file, "w") as handle:
        handle.create_dataset("bin_edges", data=WAVENUMBER_GRID)
        handle.create_dataset("t", data=TEMPERATURE_GRID)
        pressure = handle.create_dataset("p", data=PRESSURE_GRID)
        pressure.attrs["units"] = "Pa"
        handle.create_dataset("xsecarr", data=cross_sections())
        handle.create_dataset("mol_name", data=np.array([b"H2O"]))

    cia_table = (
        np.outer(
            1.0 + 200.0 / TEMPERATURE_GRID,
            np.exp(-(((WAVENUMBER_GRID - 2400.0) / 150.0) ** 2)),
        )
        * 1e-40
    )
    # The CIA cache looks for .db files and takes the pair name from the stem.
    with open(tmp_path / "H2-H2.db", "wb") as handle:
        pickle.dump(
            {
                "t": TEMPERATURE_GRID,
                "wno": WAVENUMBER_GRID,
                "xsecarr": cia_table,
                "name": "H2-H2",
            },
            handle,
        )

    GlobalCache()["mpi_use_shared"] = False
    # The differentiable model only implements cross-section opacities, and the
    # opacity method is global state that an earlier test can leave on ktables,
    # so it is pinned here rather than inherited from whatever ran before.
    GlobalCache()["opacity_method"] = "xsec"
    opacity_cache = OpacityCache()
    opacity_cache.clear_cache()
    opacity_cache.set_opacity_path(str(tmp_path))
    cia_cache = CIACache()
    cia_cache.set_cia_path(str(tmp_path))

    yield tmp_path

    opacity_cache.clear_cache()
    GlobalCache()["xsec_path"] = None
    GlobalCache()["cia_path"] = None


@pytest.fixture
def observation():
    """A small observed spectrum of twenty bins."""
    from taurex.data.spectrum import ArraySpectrum

    wavelength = np.linspace(4.0, 4.5, 20)
    data = np.full(wavelength.shape, 0.015)
    error = np.full(wavelength.shape, 1e-4)
    width = np.full(wavelength.shape, 0.02)
    return ArraySpectrum(np.vstack([wavelength, data, error, width]).T)


def build_model(
    temperature_profile=None,
    h2o_mix_ratio=1e-4,
    cia=True,
    flat_mie=True,
):
    """Build the small transmission model.

    The temperature profile has to be chosen before the model is built, because
    the fitting parameters of a profile are compiled into the model when it is
    built and swapping the profile afterwards leaves that list stale.

    Parameters
    ----------
    temperature_profile:
        Temperature profile to use, isothermal at 900 K by default

    h2o_mix_ratio:
        Mixing ratio of the absorbing gas. The synthetic cross-sections make
        the default model optically thick everywhere; a test that needs a
        transmission it can invert has to lower it.

    cia:
        Whether to add collision induced absorption, which dominates the
        optical depth of the default model

    flat_mie:
        Whether to add the flat cloud deck

    Returns
    -------
    :class:`taurex.model.TransmissionModel`
        The built model

    """
    from taurex.contributions import AbsorptionContribution
    from taurex.contributions import CIAContribution
    from taurex.contributions import FlatMieContribution
    from taurex.contributions import RayleighContribution
    from taurex.data import Planet
    from taurex.data.profiles.chemistry import ConstantGas
    from taurex.data.profiles.chemistry import TaurexChemistry
    from taurex.data.profiles.pressure import SimplePressureProfile
    from taurex.data.profiles.temperature import Isothermal
    from taurex.data.stellar import BlackbodyStar
    from taurex.model import TransmissionModel

    chemistry = TaurexChemistry(fill_gases=["H2", "He"], ratio=0.172)
    chemistry.addGas(ConstantGas("H2O", mix_ratio=h2o_mix_ratio))

    if temperature_profile is None:
        temperature_profile = Isothermal(T=900.0)

    model = TransmissionModel(
        planet=Planet(planet_mass=0.74, planet_radius=1.38),
        star=BlackbodyStar(temperature=6000.0, radius=1.16),
        pressure_profile=SimplePressureProfile(
            nlayers=30, atm_min_pressure=1e-2, atm_max_pressure=1e6
        ),
        temperature_profile=temperature_profile,
        chemistry=chemistry,
    )
    model.add_contribution(AbsorptionContribution())
    model.add_contribution(RayleighContribution())
    if cia:
        model.add_contribution(CIAContribution(cia_pairs=["H2-H2"]))
    if flat_mie:
        model.add_contribution(
            FlatMieContribution(flat_mix_ratio=1e-24, flat_topP=1e2, flat_bottomP=-1)
        )
    model.model()
    return model


@pytest.fixture
def taurex_model(data_path, observation):
    """A small transmission model using the synthetic opacities."""
    return build_model()


def build_thin_model():
    """A model whose transmission is not saturated by the synthetic tables.

    The default fixtures are optically thick at every layer (tau of order 1e18),
    so exp(-tau) underflows and a cloud deck could be added or removed without
    changing the spectrum. Dropping the CIA term and the absorbing gas leaves a
    transmission that the cloud actually moves.
    """
    return build_model(h2o_mix_ratio=1e-30, cia=False, flat_mie=False)


def layer_midpoints(model):
    """Two pressures, each midway between consecutive layer centres.

    A boundary that sits on a layer centre is exactly where a soft window and a
    step function disagree. Midway between two centres each sigmoid splits one
    pair of layers, and the symmetric pair carries the column the step function
    selects.

    Parameters
    ----------
    model:
        The model the pressures are read from

    Returns
    -------
    bottom, top:
        The two boundary pressures in Pa

    """
    centres = np.log10(np.asarray(model.pressureProfile))
    return 10.0 ** (0.5 * (centres[3] + centres[4])), 10.0 ** (
        0.5 * (centres[18] + centres[19])
    )


def add_leemie_deck(model, bottom, top, mix_ratio=1e12):
    """Add a LeeMie contribution and collect its fitting parameters.

    Parameters
    ----------
    model:
        The model to add the contribution to

    bottom, top:
        Boundary pressures in Pa; a negative value leaves that end open

    mix_ratio:
        Particle number density in m^-3

    """
    from taurex.contributions import LeeMieContribution

    model.add_contribution(
        LeeMieContribution(
            lee_mie_radius=0.05,
            lee_mie_q=40.0,
            lee_mie_mix_ratio=mix_ratio,
            lee_mie_bottomP=bottom,
            lee_mie_topP=top,
        )
    )
    # The model was built before the deck was added, so its fitting parameters
    # have to be collected again; an input file adds its contributions first.
    model.build()


def compile_fit_params(model, observation, names):
    """Compile the named fitting parameters of a model and observation."""
    from taurex.optimizer.optimizer import compile_params

    model_params, _ = compile_params(
        model.fittingParameters, model.derivedParameters, {}
    )
    observation_params, _ = compile_params(
        observation.fittingParameters, observation.derivedParameters, {}
    )
    available = {param.name: param for param in model_params + observation_params}
    return [available[name] for name in names]
