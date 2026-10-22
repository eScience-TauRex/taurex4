"""Temporary diagnostics for the LeeMie plan."""

import numpy as np

from taurex.differentiability import Atmosphere

from .conftest import build_model


def test_debug(data_path, observation):
    import jax.numpy as jnp

    model = build_model()
    atmosphere = Atmosphere(model, observation, fit_params=[])
    _, tau, _ = atmosphere.forward(jnp.zeros(0))
    tau = np.asarray(tau)
    print("\nbase tau min/max", tau.min(), tau.max())
    print("base tau[:, 0]", tau[:, 0])
    print("plan names", [type(p).__name__ for p in atmosphere._plans])
    density = model.pressureProfile / (1.380649e-23 * model.temperatureProfile)
    print("density min/max", density.min(), density.max())
