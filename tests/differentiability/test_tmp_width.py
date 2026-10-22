"""Temporary width experiment for the LeeMie sigmoid."""

import numpy as np

from taurex.differentiability import Atmosphere
from taurex.differentiability.model import LeeMiePlan

from .conftest import build_model
from .test_model import _add_leemie_deck
from .test_model import _layer_midpoints


def test_width(data_path, observation):
    import jax.numpy as jnp

    for width in (0.5, 0.25, 0.1, 0.05):
        LeeMiePlan.width_in_layers = width
        model = build_model(h2o_mix_ratio=1e-30, cia=False, flat_mie=False)
        bottom, top = _layer_midpoints(model)
        _add_leemie_deck(model, bottom, top)
        atmosphere = Atmosphere(model, observation, fit_params=[])
        depth = np.asarray(atmosphere.native_spectrum(jnp.zeros(0))[1])
        _, expected, _, _ = model.model(wngrid=observation.wavenumberGrid)
        expected = np.asarray(expected)
        relative = np.abs(depth - expected) / expected
        print(f"\nwidth {width}: spectrum max rel {relative.max():.3e}")
    LeeMiePlan.width_in_layers = 0.25
