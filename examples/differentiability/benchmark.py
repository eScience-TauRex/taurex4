"""Time the differentiable forward model and one end to end retrieval.

The workload is synthetic and self contained - 80 layers, 4669 native
wavenumber points, 488 observation bins - so it needs no opacity data and runs
in seconds. It is deliberately the same size as the retrieval the
differentiability reports quote, so the numbers of the numpy model, of the torch
implementation and of this one can be put side by side.

The script works on both branches of the differentiability work, because the
package it imports has the same name and the same interface in each:

    python examples/differentiability/benchmark.py --backend jax
    python examples/differentiability/benchmark.py --backend torch
    python examples/differentiability/benchmark.py --backend jax --fit

Only ``--fit`` needs the synthetic data at the injected parameters, and it is
the part that reports the optimiser's own timings.
"""

import argparse
import pickle  # noqa: S403
from functools import partial
import tempfile
import time
from pathlib import Path

import numpy as np


WAVENUMBER_GRID = np.linspace(5900.0, 9100.0, 4669)
TEMPERATURE_GRID = np.linspace(300.0, 1500.0, 6)
PRESSURE_GRID = np.logspace(6.0, 0.0, 5)
N_LAYERS = 80
N_BINS = 488
TRUTH = np.array([1.0, 800.0, -5.0, -5.0])
START = np.array([1.05, 700.0, -4.0, -6.0])
FIT_NAMES = ("planet_radius", "T", "H2O", "CO2")


def cross_sections(centre, width):
    """Synthetic cross-sections with a distinguishable band per molecule.

    Parameters
    ----------
    centre:
        Band centre in cm^-1

    width:
        Band width in cm^-1

    Returns
    -------
    :obj:`numpy.ndarray`
        Cross-sections in cm^2, shape ``(np, nt, nwn)``

    """
    temperature = TEMPERATURE_GRID[None, :, None]
    pressure = PRESSURE_GRID[:, None, None]
    wavenumber = WAVENUMBER_GRID[None, None, :]
    band = np.exp(-((wavenumber - centre) / width) ** 2)
    return (1e-18 * band + 1e-22) * (1.0 + 3000.0 / temperature) * (
        1.0 + 1e-6 * pressure
    )


def write_tables(tmp_path):
    """Write the synthetic opacity and CIA tables and point taurex at them.

    Parameters
    ----------
    tmp_path:
        Directory to write the tables into

    """
    import h5py

    from taurex.cache import CIACache, GlobalCache, OpacityCache

    tables = (
        ("H2O", "1H2-16O", 6800.0, 250.0),
        ("CO2", "12C-16O2", 8600.0, 120.0),
    )
    for molecule, tag, centre, width in tables:
        with h5py.File(
            tmp_path / f"{tag}__synth.R1000_2-5mu.xsec.TauREx.h5", "w"
        ) as handle:
            handle.create_dataset("bin_edges", data=WAVENUMBER_GRID)
            handle.create_dataset("t", data=TEMPERATURE_GRID)
            pressure = handle.create_dataset("p", data=PRESSURE_GRID)
            pressure.attrs["units"] = "Pa"
            handle.create_dataset("xsecarr", data=cross_sections(centre, width))
            handle.create_dataset("mol_name", data=np.array([molecule.encode()]))

    cia_table = np.outer(
        1.0 + 200.0 / TEMPERATURE_GRID,
        np.exp(-((WAVENUMBER_GRID - 7500.0) / 600.0) ** 2),
    ) * 1e-40
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
    GlobalCache()["opacity_method"] = "xsec"
    opacity_cache = OpacityCache()
    opacity_cache.clear_cache()
    opacity_cache.set_opacity_path(str(tmp_path))
    CIACache().set_cia_path(str(tmp_path))


def build_model(observation):
    """Build the transmission model the benchmark runs.

    Parameters
    ----------
    observation:
        Observation the model is evaluated against

    Returns
    -------
    :class:`taurex.model.TransmissionModel`
        A built model with absorption, Rayleigh, CIA and a flat cloud deck

    """
    from taurex.contributions import (
        AbsorptionContribution,
        CIAContribution,
        FlatMieContribution,
        RayleighContribution,
    )
    from taurex.data import Planet
    from taurex.data.profiles.chemistry import ConstantGas, TaurexChemistry
    from taurex.data.profiles.pressure import SimplePressureProfile
    from taurex.data.profiles.temperature import Isothermal
    from taurex.data.stellar import BlackbodyStar
    from taurex.model import TransmissionModel

    chemistry = TaurexChemistry(fill_gases=["H2", "He"], ratio=0.172)
    chemistry.addGas(ConstantGas("H2O", mix_ratio=1e-5))
    chemistry.addGas(ConstantGas("CO2", mix_ratio=1e-5))

    model = TransmissionModel(
        planet=Planet(planet_mass=0.12, planet_radius=0.82),
        star=BlackbodyStar(temperature=6400.0, radius=0.676),
        pressure_profile=SimplePressureProfile(
            nlayers=N_LAYERS, atm_min_pressure=1e-2, atm_max_pressure=1e6
        ),
        temperature_profile=Isothermal(T=800.0),
        chemistry=chemistry,
    )
    model.add_contribution(AbsorptionContribution())
    model.add_contribution(RayleighContribution())
    model.add_contribution(CIAContribution(cia_pairs=["H2-H2"]))
    model.add_contribution(
        FlatMieContribution(flat_mix_ratio=1e-24, flat_topP=1e2, flat_bottomP=-1)
    )
    model.model(wngrid=observation.wavenumberGrid)
    return model


def build_observation():
    """Build the 488 bin observation used for the binning operator.

    Returns
    -------
    :class:`taurex.data.spectrum.ArraySpectrum`
        The observation

    """
    from taurex.data.spectrum import ArraySpectrum

    centres = np.linspace(6000.0, 9000.0, N_BINS)
    wavelength = 1e4 / centres
    data = np.full(wavelength.shape, 0.015)
    error = np.full(wavelength.shape, 1e-4)
    width = np.full(wavelength.shape, 0.9 / N_BINS)
    return ArraySpectrum(np.vstack([wavelength, data, error, width]).T)


def fit_parameters(model, observation):
    """Compile the four fitting parameters the benchmark varies.

    Parameters
    ----------
    model:
        The forward model

    observation:
        The observation

    Returns
    -------
    :obj:`list` of :class:`taurex.optimizer.optimizer.FitParam`
        The fitting parameters

    """
    from taurex.optimizer.optimizer import compile_params

    model_params, _ = compile_params(
        model.fittingParameters, model.derivedParameters, {}
    )
    observation_params, _ = compile_params(
        observation.fittingParameters, observation.derivedParameters, {}
    )
    available = {param.name: param for param in model_params + observation_params}
    return [available[name] for name in FIT_NAMES]


def torch_value_and_gradient(torch, function, point):
    """Return the value and gradient of ``function`` at ``point``.

    Parameters
    ----------
    torch:
        The torch module

    function:
        Scalar function to differentiate

    point:
        Point to evaluate at

    Returns
    -------
    value, gradient:
        The value and its derivative at the point

    """
    point = point.clone().requires_grad_(True)
    value = function(point)
    value.backward()
    return value, point.grad


def time_it(function, repeats=10):
    """Time a callable, taking the best of several runs.

    Parameters
    ----------
    function:
        Callable to time

    repeats:
        Number of times to run it

    Returns
    -------
    float
        Best wall clock time in milliseconds

    """
    function()
    best = np.inf
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - started)
    return best * 1e3


def run_micro_benchmark(backend, model, observation, atmosphere):
    """Time the numpy forward pass and the differentiable one.

    Parameters
    ----------
    backend:
        Name of the backend being benchmarked

    model:
        The taurex forward model

    observation:
        The observation

    atmosphere:
        The differentiable atmosphere

    Returns
    -------
    forward:
        Time of one backward-capable forward pass in milliseconds

    """
    theta = backend.array(TRUTH)

    numpy_time = time_it(lambda: model.model(wngrid=observation.wavenumberGrid))
    print(f"numpy forward              {numpy_time:8.2f} ms")

    spectrum = backend.compile(atmosphere.spectrum)
    started = time.perf_counter()
    backend.run(spectrum, theta)
    compile_time = time.perf_counter() - started
    forward = time_it(lambda: backend.run(spectrum, theta))
    print(
        f"{backend.name:5s} forward              {forward:8.2f} ms   "
        f"(compile {compile_time:.2f} s)"
    )

    objective = backend.compile(backend.value_and_grad(atmosphere.chi_squared))
    started = time.perf_counter()
    backend.run(objective, theta)
    objective_compile = time.perf_counter() - started
    both = time_it(lambda: backend.run(objective, theta))
    print(
        f"{backend.name:5s} forward + backward  {both:8.2f} ms   "
        f"({both / forward:.2f} x forward, compile {objective_compile:.2f} s)"
    )

    frozen = backend.Atmosphere(model, observation, fit_params=[])
    reference = model.model(wngrid=observation.wavenumberGrid)[1]
    value = backend.to_numpy(frozen.native_spectrum(backend.zeros(0))[1])
    print(
        "native spectrum max relative difference against numpy "
        f"{np.max(np.abs(value - reference) / np.abs(reference)):.3e}"
    )
    return forward


def run_fit(backend, model, observation, atmosphere):
    """Run one retrieval and print what it cost.

    The observation is replaced by the model evaluated at ``TRUTH``, so the
    fit has a known answer to converge to and the two backends can be compared
    on what they recover as well as on how long they took.

    Parameters
    ----------
    backend:
        Name of the backend being benchmarked

    model:
        The taurex forward model

    observation:
        The observation

    atmosphere:
        The differentiable atmosphere used to generate the data

    """
    from taurex.differentiability import LaplaceOptimizer

    data = backend.to_numpy(atmosphere.spectrum(backend.array(TRUTH)))
    observation._obs_spectrum[:, 1] = data

    optimizer = LaplaceOptimizer(num_samples=2000, max_iterations=200)
    optimizer.set_model(model)
    optimizer.set_observed(observation)
    for name in FIT_NAMES:
        optimizer.enable_fit(name)
    optimizer.update_model(START)

    started = time.perf_counter()
    optimizer.compute_fit()
    total = time.perf_counter() - started

    print(f"fit {total:.2f} s total")
    print(f"    start  {np.array2string(START, precision=4)}")
    print(f"    map    {np.array2string(optimizer._map, precision=4)}")
    print(f"    truth  {np.array2string(TRUTH, precision=4)}")
    print(
        f"    sigma  "
        f"{np.array2string(optimizer.get_samples(0).std(axis=0), precision=4)}"
    )
    chi_squared = backend.to_numpy(
        optimizer.atmosphere.chi_squared(backend.array(optimizer._map))
    )
    print(f"    chi-squared at the map {float(chi_squared):.6g}")
    for name in (
        "iterations",
        "function_evaluations",
        "gradient_evaluations",
        "compile_time",
        "fit_time",
        "laplace_time",
    ):
        value = getattr(optimizer, name, None)
        if value is not None:
            print(f"    {name}: {value}")


class JaxBackend:
    """The handful of operations the benchmark needs, in JAX."""

    name = "jax"

    def __init__(self):
        """Import JAX and the differentiable model."""
        import jax
        import jax.numpy as jnp

        from taurex.differentiability import Atmosphere

        self.jax = jax
        self.jnp = jnp
        self.Atmosphere = Atmosphere

    def array(self, values):
        """Return the values as a JAX array."""
        return self.jnp.asarray(values, dtype=self.jnp.float64)

    def zeros(self, size):
        """Return a zero JAX array of the given size."""
        return self.jnp.zeros(size, dtype=self.jnp.float64)

    def compile(self, function):
        """Compile a function and return it, ready to be called."""
        return self.jax.jit(function)

    def run(self, function, *args):
        """Call a compiled function and wait for the result."""
        return self.jax.block_until_ready(function(*args))

    def value_and_grad(self, function):
        """Return the value and gradient function of ``function``."""
        return self.jax.value_and_grad(function)

    def to_numpy(self, value):
        """Return a numpy copy of a JAX array."""
        return np.asarray(value)


class TorchBackend:
    """The handful of operations the benchmark needs, in torch."""

    name = "torch"

    def __init__(self):
        """Import torch and the differentiable model."""
        import torch

        from taurex.differentiability import Atmosphere

        self.torch = torch
        self.Atmosphere = Atmosphere

    def array(self, values):
        """Return the values as a torch tensor."""
        return self.torch.tensor(values, dtype=self.torch.float64)

    def zeros(self, size):
        """Return a zero torch tensor of the given size."""
        return self.torch.zeros(size, dtype=self.torch.float64)

    def compile(self, function):
        """Return the function unchanged; torch runs it eagerly."""
        return function

    def run(self, function, *args):
        """Call a function with gradients switched off."""
        with self.torch.no_grad():
            return function(*args)

    def value_and_grad(self, function):
        """Return a value and gradient function built from ``backward``."""
        return partial(torch_value_and_gradient, self.torch, function)

    def to_numpy(self, value):
        """Return a numpy copy of a torch tensor."""
        return value.detach().numpy().copy()


def main():
    """Run the benchmark requested on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=("jax", "torch"), default="jax", help="backend to time"
    )
    parser.add_argument(
        "--fit", action="store_true", help="also run one end to end retrieval"
    )
    args = parser.parse_args()

    backend = JaxBackend() if args.backend == "jax" else TorchBackend()

    tmp_path = Path(tempfile.mkdtemp())
    write_tables(tmp_path)
    observation = build_observation()
    model = build_model(observation)
    atmosphere = backend.Atmosphere(
        model, observation, fit_params=fit_parameters(model, observation)
    )
    print(
        f"native points {atmosphere.wngrid.size}, "
        f"layers {atmosphere._n_layers}, backend {backend.name}"
    )

    run_micro_benchmark(backend, model, observation, atmosphere)
    if args.fit:
        run_fit(backend, model, observation, atmosphere)


if __name__ == "__main__":
    main()
