"""Differentiable PyTorch port of the TauREx transmission forward model.

The module mirrors the numpy pipeline that
:class:`taurex.model.transmission.TransmissionModel` implements, but every
stage is expressed as torch tensor operations, so a single ``backward()``
returns the gradient of the spectrum with respect to all fitting parameters.

The taurex objects remain the source of truth: profiles, gases, contributions,
priors and bounds all come from the object graph the input file built, and the
opacity tables are the very same arrays the numpy model uses. Only the
arithmetic is re-expressed in torch.
"""

import typing as t

import numpy as np
import torch

from taurex.contributions import AbsorptionContribution
from taurex.contributions import CIAContribution
from taurex.contributions import FlatMieContribution
from taurex.contributions import RayleighContribution
from taurex.core.priors import PriorMode
from taurex.types import get_float_dtype

from . import physics


FloatTensor = torch.Tensor


class GridResampler:
    """Static linear resampling operator along the last axis.

    Reproduces the wavenumber interpolation that
    :meth:`taurex.opacity.opacity.Opacity.opacity` and
    :meth:`taurex.cia.cia.CIA.cia` apply after the temperature and pressure
    interpolation. Both steps are linear, so a table is resampled once at
    construction time and the hot loop only has to interpolate in ``(T, log P)``.

    Parameters
    ----------
    source:
        Native grid of the table

    target:
        Grid to resample onto

    hold:
        When True, targets outside the source range take the nearest endpoint
        value, which is what :func:`numpy.interp` does by default. When False
        they are set to zero, which is the ``opacity_hold`` default of
        :meth:`taurex.opacity.opacity.Opacity.opacity`.

    dtype:
        Floating point dtype of the working tensors

    device:
        Device of the working tensors

    subset:
        Optional index array selecting the part of ``source`` that the caller
        interpolates over, as ``Opacity.opacity`` does before calling
        :func:`numpy.interp`.

    """

    def __init__(
        self,
        source: np.ndarray,
        target: np.ndarray,
        hold: bool,
        dtype: torch.dtype,
        device: torch.device,
        subset: t.Optional[np.ndarray] = None,
    ) -> None:
        """Initialise the resampler.

        Parameters
        ----------
        source:
            Native grid of the table

        target:
            Grid to resample onto

        hold:
            Edge behaviour for targets outside the source range

        dtype:
            Floating point dtype of the working tensors

        device:
            Device of the working tensors

        subset:
            Optional index array selecting part of ``source``

        """
        if subset is not None:
            source = source[subset]
        if source.shape[0] == 0:
            raise ValueError(
                "Cannot resample onto the working wavenumber grid: no table "
                "points fall inside it"
            )

        self.identity = np.array_equal(source, target)
        self.hold = hold
        if self.identity:
            return
        if source.shape[0] < 2:
            raise ValueError("Cannot interpolate from a single table point")

        upper = np.searchsorted(source, target, side="left")
        upper = np.clip(upper, 1, source.shape[0] - 1)
        lower = upper - 1
        weight = (target - source[lower]) / (source[upper] - source[lower])

        self._lower = torch.as_tensor(lower, dtype=torch.int64, device=device)
        self._upper = torch.as_tensor(upper, dtype=torch.int64, device=device)
        self._weight = torch.as_tensor(weight, dtype=dtype, device=device)
        self._below = torch.as_tensor(target < source[0], device=device)
        self._above = torch.as_tensor(target > source[-1], device=device)

    def apply(self, values: FloatTensor) -> FloatTensor:
        """Resample ``values`` along its last axis.

        Parameters
        ----------
        values:
            Array whose last axis is the source grid

        Returns
        -------
        :obj:`torch.Tensor`
            Array whose last axis is the target grid

        """
        if self.identity:
            return values

        weight = self._weight.reshape((1,) * (values.dim() - 1) + (-1,))
        lower = values.index_select(-1, self._lower)
        upper = values.index_select(-1, self._upper)
        result = lower + weight * (upper - lower)

        shape = (1,) * (values.dim() - 1) + (-1,)
        if self.hold:
            low_fill = values[..., :1].expand_as(result)
            high_fill = values[..., -1:].expand_as(result)
        else:
            low_fill = torch.zeros_like(result)
            high_fill = torch.zeros_like(result)

        result = torch.where(self._below.reshape(shape), low_fill, result)
        return torch.where(self._above.reshape(shape), high_fill, result)


def interp_temperature(
    x_low: FloatTensor,
    x_high: FloatTensor,
    temperature: FloatTensor,
    temperature_min: FloatTensor,
    temperature_max: FloatTensor,
    mode: str,
) -> FloatTensor:
    """Interpolate the temperature axis with the table's own interpolation mode.

    Parameters
    ----------
    x_low, x_high:
        Values at ``temperature_min`` and ``temperature_max``

    temperature:
        Coordinate to interpolate to

    temperature_min, temperature_max:
        Bounding temperature grid points

    mode:
        ``linear`` or ``exp``

    Returns
    -------
    :obj:`torch.Tensor`
        Interpolated values

    """
    if mode == "linear":
        return physics.interp_lin(
            x_low, x_high, temperature, temperature_min, temperature_max
        )
    if mode == "exp":
        return physics.interp_exp_only(
            x_low, x_high, temperature, temperature_min, temperature_max
        )
    raise ValueError(f"Unknown interpolation mode {mode}")


class OpacityInterpolator:
    """Differentiable bilinear interpolation of an opacity table.

    Port of
    :meth:`taurex.opacity.interpolateopacity.InterpolatingOpacity.interp_bilinear_grid`
    for every layer at once. The ``searchsorted`` indices that select the
    bracketing grid points are integer valued and carry no gradient, so they
    are detached and the gradient flows through the interpolation weights
    instead.

    Parameters
    ----------
    table:
        Cross-section table in cm^2, shape ``(np, nt, nwn)``, already resampled
        onto the working wavenumber grid

    temperature_grid:
        Temperature grid in K

    pressure_grid:
        Pressure grid in Pa

    mode:
        ``linear`` or ``exp`` temperature interpolation direction

    """

    def __init__(
        self,
        table: FloatTensor,
        temperature_grid: FloatTensor,
        pressure_grid: FloatTensor,
        mode: str,
    ) -> None:
        """Initialise the interpolator.

        Parameters
        ----------
        table:
            Cross-section table in cm^2, already resampled onto the working grid

        temperature_grid:
            Temperature grid in K

        pressure_grid:
            Pressure grid in Pa

        mode:
            ``linear`` or ``exp`` temperature interpolation

        """
        self.table = table
        self.temperature_grid = temperature_grid
        self.log_pressure_grid = torch.log10(pressure_grid)
        self.mode = mode

    def __call__(
        self, temperature: FloatTensor, log_pressure: FloatTensor
    ) -> FloatTensor:
        """Interpolate to ``temperature`` and ``log_pressure`` per layer.

        Parameters
        ----------
        temperature:
            Temperature at each layer in K, shape ``(nlayers,)``

        log_pressure:
            Base-10 logarithm of the pressure at each layer, shape
            ``(nlayers,)``

        Returns
        -------
        :obj:`torch.Tensor`
            Cross-sections in m^2, shape ``(nlayers, nwn)``

        """
        t_low, t_high = physics.find_closest_pair(self.temperature_grid, temperature)
        p_low, p_high = physics.find_closest_pair(self.log_pressure_grid, log_pressure)

        q11 = self.table[p_low, t_low]
        q12 = self.table[p_low, t_high]
        q21 = self.table[p_high, t_low]
        q22 = self.table[p_high, t_high]

        t_min = self.temperature_grid[t_low][:, None]
        t_max = self.temperature_grid[t_high][:, None]
        p_min = self.log_pressure_grid[p_low][:, None]
        p_max = self.log_pressure_grid[p_high][:, None]
        t_col = temperature[:, None]
        p_col = log_pressure[:, None]

        if self.mode == "linear":
            bilinear = physics.interp_bilin(
                q11, q12, q21, q22, t_col, t_min, t_max, p_col, p_min, p_max
            )
        elif self.mode == "exp":
            bilinear = physics.interp_exp_and_lin(
                q11, q12, q21, q22, t_col, t_min, t_max, p_col, p_min, p_max
            )
        else:
            raise ValueError(f"Unknown interpolation mode {self.mode}")

        over_pressure = log_pressure >= self.log_pressure_grid[-1]
        over_temperature = temperature >= self.temperature_grid[-1]
        under_pressure = log_pressure < self.log_pressure_grid[0]
        under_temperature = temperature < self.temperature_grid[0]

        # Past the edge of the grid one axis is pinned to the last or first
        # entry and the other keeps its dependence, exactly as
        # interp_temp_only / interp_pressure_only do.
        def temperature_branch(pressure_index: int) -> FloatTensor:
            return interp_temperature(
                self.table[pressure_index, t_low],
                self.table[pressure_index, t_high],
                t_col,
                t_min,
                t_max,
                self.mode,
            )

        def pressure_branch(temperature_index: int) -> FloatTensor:
            return physics.interp_lin(
                self.table[p_low, temperature_index],
                self.table[p_high, temperature_index],
                p_col,
                p_min,
                p_max,
            )

        over_pressure_value = temperature_branch(-1)
        over_temperature_value = pressure_branch(-1)
        under_pressure_value = temperature_branch(0)
        under_temperature_value = pressure_branch(0)

        def mask(flag: FloatTensor) -> FloatTensor:
            return flag[:, None]

        result = torch.where(
            mask(over_pressure & over_temperature), self.table[-1, -1], bilinear
        )
        result = torch.where(
            mask(under_pressure & under_temperature),
            torch.zeros_like(bilinear),
            result,
        )
        result = torch.where(mask(over_pressure), over_pressure_value, result)
        result = torch.where(mask(over_temperature), over_temperature_value, result)
        result = torch.where(mask(under_pressure), under_pressure_value, result)
        result = torch.where(mask(under_temperature), under_temperature_value, result)
        return result / 10000.0


class ThermalState:
    """Layer-by-layer quantities that the contributions depend on.

    Parameters
    ----------
    temperature:
        Temperature at each layer in K

    pressure:
        Pressure at each layer in Pa

    density:
        Number density at each layer in m^-3

    mix:
        Mixing ratio profile of each molecule

    n_wavenumbers:
        Size of the working wavenumber grid

    """

    def __init__(
        self,
        temperature: FloatTensor,
        pressure: FloatTensor,
        density: FloatTensor,
        mix: t.Dict[str, FloatTensor],
        n_wavenumbers: int,
    ) -> None:
        """Initialise the layer state.

        Parameters
        ----------
        temperature:
            Temperature at each layer in K

        pressure:
            Pressure at each layer in Pa

        density:
            Number density at each layer in m^-3

        mix:
            Mixing ratio profile of each molecule

        n_wavenumbers:
            Size of the working wavenumber grid

        """
        self.temperature = temperature
        self.pressure = pressure
        self.log_pressure = torch.log10(pressure)
        self.density = density
        self.mix = mix
        self.n_wavenumbers = n_wavenumbers


class ContributionPlan:
    """Base class for a differentiable contribution."""

    name: str = "Contribution"
    density_power: int = 1

    def sigma(
        self,
        values: t.Dict[str, FloatTensor],
        thermal: ThermalState,
    ) -> t.Optional[FloatTensor]:
        """Cross-section of this contribution, before the density weighting.

        Parameters
        ----------
        values:
            Physical parameter values

        thermal:
            Layer quantities

        Returns
        -------
        :obj:`torch.Tensor` or None
            Cross-sections in m^2, shape ``(nlayers, nwn)``, or None when the
            contribution is inactive

        """
        raise NotImplementedError


class AbsorptionPlan(ContributionPlan):
    """Molecular absorption from cross-section opacities."""

    name = "Absorption"

    def __init__(self, terms: t.List[t.Tuple[str, OpacityInterpolator]]) -> None:
        """Initialise the plan.

        Parameters
        ----------
        terms:
            Molecule name and its opacity interpolator, per active gas

        """
        self._terms = terms

    def sigma(self, values, thermal):
        """Sum the absorption cross-section of every active gas."""
        sigma = None
        for gas, interpolator in self._terms:
            gas_sigma = interpolator(thermal.temperature, thermal.log_pressure)
            gas_sigma = gas_sigma * thermal.mix[gas][:, None]
            sigma = gas_sigma if sigma is None else sigma + gas_sigma
        return sigma


class RayleighPlan(ContributionPlan):
    """Rayleigh scattering, whose cross-section only depends on wavenumber."""

    name = "Rayleigh"

    def __init__(self, terms: t.List[t.Tuple[str, FloatTensor]]) -> None:
        """Initialise the plan.

        Parameters
        ----------
        terms:
            Molecule name and its wavenumber dependent cross-section

        """
        self._terms = terms

    def sigma(self, values, thermal):
        """Sum the abundance weighted Rayleigh cross-section of each gas."""
        sigma = None
        for gas, gas_sigma in self._terms:
            term = gas_sigma[None, :] * thermal.mix[gas][:, None]
            sigma = term if sigma is None else sigma + term
        return sigma


class CiaPlan(ContributionPlan):
    """Collision induced absorption, weighted by the product of pair abundances."""

    name = "CIA"
    density_power = 2

    def __init__(
        self, terms: t.List[t.Tuple[str, str, FloatTensor, FloatTensor]]
    ) -> None:
        """Initialise the plan.

        Parameters
        ----------
        terms:
            Molecule pair and its temperature dependent cross-section table

        """
        self._terms = terms

    def sigma(self, values, thermal):
        """Sum the pair density weighted CIA cross-section of every pair."""
        sigma = None
        for pair_one, pair_two, temperature_grid, table in self._terms:
            low, high = physics.find_closest_pair(temperature_grid, thermal.temperature)
            sigma_cia = physics.interp_lin(
                table[low],
                table[high],
                thermal.temperature[:, None],
                temperature_grid[low][:, None],
                temperature_grid[high][:, None],
            )
            # HitranCIA.interp_linear_grid pins the cross-section to the end of
            # the temperature grid instead of extrapolating.
            over = (thermal.temperature > temperature_grid[-1])[:, None]
            under = (thermal.temperature < temperature_grid[0])[:, None]
            sigma_cia = torch.where(over, table[-1], sigma_cia)
            sigma_cia = torch.where(under, table[0], sigma_cia)

            factor = (thermal.mix[pair_one] * thermal.mix[pair_two])[:, None]
            term = sigma_cia * factor
            sigma = term if sigma is None else sigma + term
        return sigma


class FlatMiePlan(ContributionPlan):
    """Flat, grey absorption between two pressures.

    The layer window is chosen with ``searchsorted`` on the static pressure
    grid, exactly as
    :meth:`taurex.contributions.flatmie.FlatMieContribution.prepare_each` does,
    but the ramp inside the window keeps its dependence on the two pressures,
    so a fitted cloud top still has a gradient.

    Parameters
    ----------
    atmosphere:
        The atmosphere the plan belongs to, used to read parameter values

    pressure_levels:
        Pressure level boundaries in Pa

    """

    name = "FlatMie"

    def __init__(self, atmosphere: "Atmosphere", pressure_levels: np.ndarray) -> None:
        """Initialise the plan.

        Parameters
        ----------
        atmosphere:
            The atmosphere the plan belongs to, used to read parameter values

        pressure_levels:
            Pressure level boundaries in Pa

        """
        self._atmosphere = atmosphere
        # Flipped once here: flipping in the forward pass would hand torch a
        # negative-stride view, which it refuses to convert.
        self._log_levels = np.log10(
            np.asarray(pressure_levels, dtype=get_float_dtype())
        )[::-1].copy()

    def sigma(self, values, thermal):
        """Cross-section of the cloud deck."""
        read = self._atmosphere._value
        # A negative boundary means "not set", in which case the deck reaches
        # the surface or the top of the atmosphere.
        top = self._boundary(values, "flat_topP", self._log_levels.min())
        bottom = self._boundary(values, "flat_bottomP", self._log_levels.max())
        low = torch.minimum(top, bottom)
        high = torch.maximum(top, bottom)

        p_left = self._log_levels[:-1]
        p_right = self._log_levels[1:]

        save_start = int(np.searchsorted(p_right, float(low.detach()), side="right"))
        save_stop = int(
            np.searchsorted(p_left[1:], float(high.detach()), side="right")
        )

        left = physics.as_tensor(
            p_left[save_start : save_stop + 1],
            dtype=thermal.temperature.dtype,
            device=thermal.temperature.device,
        )
        right = physics.as_tensor(
            p_right[save_start : save_stop + 1],
            dtype=thermal.temperature.dtype,
            device=thermal.temperature.device,
        )
        window = torch.clamp(right, max=high) - torch.clamp(left, min=low)
        window = window / window.max()

        sigma = torch.zeros(
            thermal.temperature.shape[0],
            thermal.n_wavenumbers,
            dtype=thermal.temperature.dtype,
            device=thermal.temperature.device,
        )
        sigma[save_start : save_stop + 1] = (
            window[:, None] * read(values, "flat_mix_ratio")
        )
        return sigma.flip(0)

    def _boundary(self, values, name: str, fallback: float) -> FloatTensor:
        """Log10 pressure of a cloud deck boundary.

        Parameters
        ----------
        values:
            Physical parameter values

        name:
            Name of the boundary pressure parameter

        fallback:
            Log10 pressure to use when the parameter is unset

        Returns
        -------
        :obj:`torch.Tensor`
            Log10 boundary pressure

        """
        current = self._atmosphere._all_params[name][2]()
        if current is None or current < 0:
            return physics.as_tensor(
                fallback,
                dtype=self._atmosphere.dtype,
                device=self._atmosphere.device,
            )
        return torch.log10(self._atmosphere._value(values, name))


class Atmosphere:
    """Differentiable forward pass of a 1D transmission atmosphere."""

    def __init__(
        self,
        model,
        observed=None,
        fit_params: t.Optional[t.Sequence[t.Any]] = None,
        device: t.Optional[t.Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        """Initialise the differentiable atmosphere.

        Parameters
        ----------
        model:
            A taurex forward model; it is built and initialised by this
            constructor, so it does not need to be prepared beforehand

        observed:
            Observation used for the binning and the chi-squared

        fit_params:
            Fitting parameters that will be varied. Each needs a ``name``, a
            ``fit_prior`` and ``bounds``, which
            :class:`taurex.optimizer.optimizer.FitParam` provides. When None,
            every parameter flagged ``to_fit`` in the model and observation is
            used.

        device:
            Torch device to run on

        dtype:
            Floating point dtype to use

        """
        self.device = torch.device(device or "cpu")
        self.dtype = dtype

        self.model = model
        self.observed = observed

        if not model.built:
            model.build()
        model.initialize_profiles()

        self._collect_parameters(model, observed, fit_params)
        self._build_geometry()
        self._build_observation()
        self._build_opacities()
        self._build_contributions()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _collect_parameters(self, model, observed, fit_params) -> None:
        """Record every fitting parameter and which ones will be varied."""
        self._all_params: t.Dict[str, t.Any] = {}
        for source in (model, observed):
            if source is None:
                continue
            for name, param in source.fittingParameters.items():
                self._all_params[name] = param

        if fit_params is None:
            self.parameter_names = [
                name for name, param in self._all_params.items() if param[5]
            ]
        else:
            self.parameter_names = [p.name for p in fit_params]

        self._index = {name: i for i, name in enumerate(self.parameter_names)}

        self._priors: t.Dict[str, t.Any] = {}
        if fit_params is not None:
            for param in fit_params:
                self._priors[param.name] = param.fit_prior

    def _frozen(self, name: str) -> FloatTensor:
        """Current value of a parameter that is not being fitted."""
        return physics.as_tensor(
            self._all_params[name][2](), dtype=self.dtype, device=self.device
        )

    def _prior(self, name: str):
        """Prior object of a fitted parameter."""
        from taurex.core.priors import LogUniform, Uniform

        if name in self._priors:
            return self._priors[name]
        mode = self._all_params[name][4]
        bounds = self._all_params[name][6]
        if mode == "log":
            return LogUniform(lin_bounds=bounds)
        return Uniform(bounds=bounds)

    def _parameters(self, theta: FloatTensor) -> t.Dict[str, FloatTensor]:
        """Map the fitted vector onto named physical parameters.

        Parameters
        ----------
        theta:
            Fitted parameters in prior space: linear for linear parameters and
            base-10 logarithm for parameters whose mode is ``log``

        Returns
        -------
        :obj:`dict` of str to :obj:`torch.Tensor`
            Physical parameter values

        """
        values: t.Dict[str, FloatTensor] = {}
        for name, index in self._index.items():
            raw = theta[index]
            prior = self._prior(name)
            mode = getattr(prior, "priorMode", PriorMode.LINEAR)
            values[name] = 10**raw if mode is PriorMode.LOG else raw
        return values

    def _value(
        self, values: t.Dict[str, FloatTensor], name: str
    ) -> FloatTensor:
        """Value of a parameter, fitted or frozen."""
        if name in values:
            return values[name]
        return self._frozen(name)

    def _node_pressure(
        self, values: t.Dict[str, FloatTensor], name: str, fallback: float
    ) -> FloatTensor:
        """Pressure of a temperature profile node.

        The node pressures are optional: when the input file leaves them out
        the profile falls back to the first and last layer centre, which is
        what :meth:`taurex.data.profiles.temperature.npoint.NPoint.profile`
        does. A negative value also means "not set".

        Parameters
        ----------
        values:
            Physical parameter values

        name:
            Name of the node pressure parameter

        fallback:
            Layer centre pressure to use when the node is unset

        Returns
        -------
        :obj:`torch.Tensor`
            Node pressure in Pa

        """
        if self._has(name):
            current = self._all_params[name][2]()
            if current is not None and current >= 0:
                return self._value(values, name)
        return physics.as_tensor(fallback, dtype=self.dtype, device=self.device)

    def _build_geometry(self) -> None:
        """Cache the pressure grid, planet constants and the working grid."""
        from taurex.constants import MJUP, RJUP
        from taurex.util import clip_native_to_wngrid

        model = self.model
        pressure = model.pressure

        self._pressure_levels = np.asarray(
            pressure.pressure_profile_levels, dtype=get_float_dtype()
        )
        self._n_layers = model.nLayers
        self._p_levels = physics.as_tensor(
            self._pressure_levels, dtype=self.dtype, device=self.device
        )
        self._p_centres = physics.as_tensor(
            np.asarray(pressure.pressure_profile, dtype=get_float_dtype()),
            dtype=self.dtype,
            device=self.device,
        )

        self._r_jup = RJUP
        self._m_jup = MJUP
        self._star_radius = model.star.radius

        native_grid = np.asarray(model.nativeWavenumberGrid)
        if self.observed is not None:
            native_grid = clip_native_to_wngrid(native_grid, self.observed.wavenumberGrid)

        self.wngrid = native_grid
        self._n_wavenumbers = native_grid.shape[0]

        chemistry = model.chemistry
        self._mix_names = set(chemistry.gases)

    def _build_observation(self) -> None:
        """Build the binning operator and cache the observed spectrum."""
        if self.observed is None:
            self.binning = None
            self._binning_matrix = None
            self._data = None
            self._error = None
            return

        from taurex.binning import FluxBinner
        from taurex.binning import NativeBinner

        binner = self.observed.create_binner()
        supported = (FluxBinner, NativeBinner)
        if not isinstance(binner, supported):
            raise NotImplementedError(
                "The differentiable model can only bin with "
                + ", ".join(cls.__name__ for cls in supported)
                + f", got {type(binner).__name__}"
            )
        self.binning = binner

        self._data = physics.as_tensor(
            self.observed.spectrum.ravel(), dtype=self.dtype, device=self.device
        )
        self._error = physics.as_tensor(
            self.observed.errorBar.ravel(), dtype=self.dtype, device=self.device
        )

        if isinstance(binner, NativeBinner):
            self._binning_matrix = None
        else:
            self._binning_matrix = physics.as_tensor(
                self._overlap_matrix(binner), dtype=self.dtype, device=self.device
            )

    def _overlap_matrix(self, binner) -> np.ndarray:
        """Weights of the linear binning operator.

        Reproduces the overlap integral of
        :meth:`taurex.binning.fluxbinner.FluxBinner.bindown` as one matrix. It
        is linear in the input spectrum with weights that only depend on the
        static grids, which is what lets a differentiable fit run at the
        observation resolution instead of the native resolution.

        Parameters
        ----------
        binner:
            Binner to reproduce

        Returns
        -------
        :obj:`numpy.ndarray`
            Matrix of shape ``(nbins, nwn)``

        """
        from taurex.util import compute_bin_edges

        centres = np.asarray(self.observed.wavenumberGrid, dtype=get_float_dtype())
        widths = np.asarray(self.observed.binWidths, dtype=get_float_dtype())

        order = np.argsort(centres)
        centres = centres[order]
        widths = widths[order]

        bin_min = centres - widths / 2
        bin_max = centres + widths / 2

        native_edges = compute_bin_edges(self.wngrid)[0]
        native_min = native_edges[:-1]
        native_max = native_edges[1:]

        overlap = np.minimum(bin_max[:, None], native_max[None, :]) - np.maximum(
            bin_min[:, None], native_min[None, :]
        )
        overlap = np.clip(overlap, 0.0, None)
        total = overlap.sum(axis=1, keepdims=True)
        return np.divide(overlap, total, out=np.zeros_like(overlap), where=total > 0)

    def _resample_table(
        self,
        table: FloatTensor,
        source: np.ndarray,
        hold: t.Optional[bool] = None,
        subset: t.Optional[np.ndarray] = None,
    ) -> FloatTensor:
        """Resample a table onto the working wavenumber grid.

        Parameters
        ----------
        table:
            Table whose last axis is ``source``

        source:
            Native wavenumber grid of the table

        hold:
            Edge behaviour, see :class:`GridResampler`

        subset:
            Optional index array restricting ``source`` before interpolating

        Returns
        -------
        :obj:`torch.Tensor`
            Table resampled onto :attr:`wngrid`

        """
        resampler = GridResampler(
            source,
            self.wngrid,
            hold=self._hold if hold is None else hold,
            dtype=self.dtype,
            device=self.device,
            subset=subset,
        )
        if subset is None:
            return resampler.apply(table)
        return resampler.apply(table[..., subset])

    def _build_opacities(self) -> None:
        """Load the cross-section tables and resample them once."""
        from taurex.cache import GlobalCache
        from taurex.cache import OpacityCache

        if GlobalCache()["opacity_method"] == "ktables":
            raise NotImplementedError(
                "The differentiable model only implements cross-section "
                "opacities. Remove opacity_method = ktables from the input, or "
                "stay with the numpy model."
            )

        self._hold = bool(GlobalCache()["opacity_hold"])
        cache = OpacityCache()
        self._interpolators: t.Dict[str, OpacityInterpolator] = {}

        for gas in self.model.chemistry.activeGases:
            opacity = cache[gas]
            source = np.asarray(opacity.wavenumberGrid, dtype=get_float_dtype())
            # Opacity.opacity() first keeps only the table points that fall
            # inside the requested grid and then interpolates over that subset.
            subset = np.where(
                (source >= self.wngrid.min()) & (source <= self.wngrid.max())
            )[0]
            table = physics.as_tensor(
                np.asarray(opacity.xsecGrid, dtype=get_float_dtype()),
                dtype=self.dtype,
                device=self.device,
            )
            self._interpolators[gas] = OpacityInterpolator(
                self._resample_table(table, source, subset=subset),
                physics.as_tensor(
                    np.asarray(opacity.temperatureGrid, dtype=get_float_dtype()),
                    dtype=self.dtype,
                    device=self.device,
                ),
                physics.as_tensor(
                    np.asarray(opacity.pressureGrid, dtype=get_float_dtype()),
                    dtype=self.dtype,
                    device=self.device,
                ),
                opacity._interp_mode,
            )

    def _build_contributions(self) -> None:
        """Translate each taurex contribution into a differentiable plan."""
        self._plans: t.List[ContributionPlan] = []
        for contribution in sorted(
            self.model.contribution_list, key=lambda item: item.order
        ):
            if isinstance(contribution, AbsorptionContribution):
                self._plans.append(self._absorption_plan())
            elif isinstance(contribution, RayleighContribution):
                self._plans.append(self._rayleigh_plan())
            elif isinstance(contribution, CIAContribution):
                self._plans.append(self._cia_plan(contribution))
            elif isinstance(contribution, FlatMieContribution):
                self._plans.append(FlatMiePlan(self, self._pressure_levels))
            else:
                raise NotImplementedError(
                    "The differentiable model does not implement "
                    f"{type(contribution).__name__}; it supports Absorption, "
                    "Rayleigh, CIA and FlatMie."
                )

    def _absorption_plan(self) -> AbsorptionPlan:
        """Build the molecular absorption plan."""
        return AbsorptionPlan(
            [
                (gas, self._interpolators[gas])
                for gas in self.model.chemistry.activeGases
            ]
        )

    def _rayleigh_plan(self) -> RayleighPlan:
        """Build the Rayleigh scattering plan."""
        from taurex.util.scattering import rayleigh_sigma_from_name

        chemistry = self.model.chemistry
        terms = []
        for gas in list(chemistry.activeGases) + list(chemistry.inactiveGases):
            if gas not in self._mix_names:
                continue
            sigma = rayleigh_sigma_from_name(gas, self.wngrid)
            if sigma is None:
                continue
            terms.append(
                (gas, physics.as_tensor(sigma, dtype=self.dtype, device=self.device))
            )
        return RayleighPlan(terms)

    def _cia_plan(self, contribution: CIAContribution) -> CiaPlan:
        """Build the collision induced absorption plan."""
        from taurex.cache import CIACache

        cache = CIACache()
        terms = []
        for pair_name in contribution.ciaPairs:
            cia = cache[pair_name]
            source = np.asarray(cia.wavenumberGrid, dtype=get_float_dtype())
            table = physics.as_tensor(
                np.asarray(cia._xsec_grid, dtype=get_float_dtype()),
                dtype=self.dtype,
                device=self.device,
            )
            terms.append(
                (
                    cia.pairOne,
                    cia.pairTwo,
                    physics.as_tensor(
                        np.asarray(cia.temperatureGrid, dtype=get_float_dtype()),
                        dtype=self.dtype,
                        device=self.device,
                    ),
                    # CIA.cia() applies np.interp, which clamps at the edges.
                    self._resample_table(table, source, hold=True),
                )
            )
        return CiaPlan(terms)

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------
    def _temperature(self, values) -> FloatTensor:
        """Temperature at each layer centre in K."""
        from taurex.data.profiles.temperature import Isothermal
        from taurex.data.profiles.temperature import NPoint

        profile = self.model.temperature
        if isinstance(profile, Isothermal):
            return self._value(values, "T") * torch.ones(
                self._n_layers, dtype=self.dtype, device=self.device
            )

        if isinstance(profile, NPoint):
            surface = self._node_pressure(
                values, "P_surface", profile.pressure_profile[0]
            )
            top = self._node_pressure(values, "P_top", profile.pressure_profile[-1])
            n_points = profile._p_points.shape[0]
            pressure_nodes = torch.stack(
                [
                    surface,
                    *[
                        self._value(values, f"P_point{i + 1}")
                        for i in range(n_points)
                    ],
                    top,
                ]
            )
            temperature_nodes = torch.stack(
                [
                    self._value(values, "T_surface"),
                    *[
                        self._value(values, f"T_point{i + 1}")
                        for i in range(n_points)
                    ],
                    self._value(values, "T_top"),
                ]
            )
            return physics.npoint_temperature(
                self._p_centres,
                pressure_nodes,
                temperature_nodes,
                profile._smooth_window,
            )

        raise NotImplementedError(
            "The differentiable model implements the Isothermal and NPoint "
            f"temperature profiles, got {type(profile).__name__}"
        )

    def _chemistry(self, values) -> t.Tuple[t.Dict[str, FloatTensor], FloatTensor]:
        """Mixing ratio profiles and mean molecular weight per layer.

        Parameters
        ----------
        values:
            Physical parameter values

        Returns
        -------
        mix, mu:
            Per-molecule mixing ratios and the mean molecular weight in kg

        """
        from taurex.data.profiles.chemistry import TaurexChemistry

        chemistry = self.model.chemistry
        if not isinstance(chemistry, TaurexChemistry):
            raise NotImplementedError(
                "The differentiable model implements TaurexChemistry, got "
                f"{type(chemistry).__name__}"
            )

        for gas in chemistry._gases:
            if type(gas).__name__ != "ConstantGas":
                raise NotImplementedError(
                    "The differentiable model implements ConstantGas mixing "
                    f"profiles, got {type(gas).__name__} for {gas.molecule}"
                )

        ones = torch.ones(self._n_layers, dtype=self.dtype, device=self.device)

        mix: t.Dict[str, FloatTensor] = {}
        total = torch.zeros(self._n_layers, dtype=self.dtype, device=self.device)
        for gas in chemistry._gases:
            profile = self._value(values, gas.molecule) * ones
            mix[gas.molecule] = profile
            total = total + profile

        remainder = 1.0 - total
        fill_gases = list(chemistry._fill_gases)
        if len(fill_gases) == 1:
            mix[fill_gases[0]] = remainder
        else:
            main = fill_gases[0]
            ratios = [self._value(values, f"{gas}_{main}") for gas in fill_gases[1:]]
            main_share = remainder / (1.0 + sum(ratios))
            mix[main] = main_share
            for gas, ratio in zip(fill_gases[1:], ratios, strict=True):
                mix[gas] = ratio * main_share

        mu = torch.zeros(self._n_layers, dtype=self.dtype, device=self.device)
        for gas, profile in mix.items():
            mass = physics.as_tensor(
                chemistry.get_molecular_mass(gas), dtype=self.dtype, device=self.device
            )
            mu = mu + mass * profile

        return mix, mu

    def _planet(self, values) -> t.Tuple[FloatTensor, FloatTensor]:
        """Planet radius and mass in SI units."""
        planet = self.model.planet
        if self._has("planet_radius"):
            radius = self._value(values, "planet_radius") * self._r_jup
        else:
            radius = physics.as_tensor(
                planet.get_planet_radius(unit="m"),
                dtype=self.dtype,
                device=self.device,
            )
        if self._has("planet_mass"):
            mass = self._value(values, "planet_mass") * self._m_jup
        else:
            mass = physics.as_tensor(
                planet.get_planet_mass(unit="kg"),
                dtype=self.dtype,
                device=self.device,
            )
        return radius, mass

    def _has(self, name: str) -> bool:
        """Whether a fitting parameter exists on the model or observation."""
        return name in self._all_params

    def forward(
        self, theta: FloatTensor
    ) -> t.Tuple[FloatTensor, FloatTensor, FloatTensor]:
        """Run the atmosphere and return everything needed downstream.

        Parameters
        ----------
        theta:
            Fitted parameters in prior space

        Returns
        -------
        depth:
            Native transit depth

        tau:
            Optical depth, shape ``(nlayers, nwn)``

        temperature:
            Temperature at each layer in K

        """
        values = self._parameters(theta)

        planet_radius, planet_mass = self._planet(values)
        temperature = self._temperature(values)
        mix, mu = self._chemistry(values)
        density = self._p_centres / (self._boltzmann() * temperature)

        altitude, _, _, deltaz = physics.altitude_gravity_scaleheight(
            planet_radius, planet_mass, temperature, mu, self._p_levels
        )
        path = physics.path_matrix(altitude[:-1], deltaz, planet_radius)

        thermal = ThermalState(
            temperature=temperature,
            pressure=self._p_centres,
            density=density,
            mix=mix,
            n_wavenumbers=self._n_wavenumbers,
        )

        weighted_sigma = torch.zeros(
            self._n_layers, self._n_wavenumbers, dtype=self.dtype, device=self.device
        )
        weight_cache: t.Dict[int, t.Optional[FloatTensor]] = {}
        for plan in self._plans:
            sigma = plan.sigma(values, thermal)
            if sigma is None:
                continue
            power = plan.density_power
            if power not in weight_cache:
                weight_cache[power] = (
                    None if power == 0 else density if power == 1 else density**power
                )
            weight = weight_cache[power]
            if weight is not None:
                sigma = sigma * weight[:, None]
            weighted_sigma = weighted_sigma + sigma

        tau = path @ weighted_sigma
        transmission = torch.exp(-tau)

        # TransmissionModel.compute_absorption overwrites tau with exp(-tau)
        # and then integrates (1 - tau), so the darkening is 1 - transmission.
        integral = torch.sum(
            (planet_radius + altitude[:-1])[:, None]
            * (1.0 - transmission)
            * deltaz[:, None]
            * 2.0,
            dim=0,
        )
        depth = (planet_radius**2 + integral) / self._star_radius**2
        return depth, tau, temperature

    @staticmethod
    def _boltzmann() -> float:
        """Boltzmann constant in J/K."""
        from taurex.constants import KBOLTZ

        return KBOLTZ

    def native_spectrum(self, theta: FloatTensor) -> t.Tuple[FloatTensor, FloatTensor]:
        """Unbinned transmission spectrum.

        Parameters
        ----------
        theta:
            Fitted parameters in prior space

        Returns
        -------
        wngrid, depth:
            Wavenumber grid and transit depth

        """
        depth, _, _ = self.forward(theta)
        return self.wngrid, depth

    def spectrum(self, theta: FloatTensor) -> FloatTensor:
        """Binned model spectrum for a set of fitted parameters.

        Parameters
        ----------
        theta:
            Fitted parameters in prior space

        Returns
        -------
        :obj:`torch.Tensor`
            Model flux on the observation grid

        """
        depth, _, _ = self.forward(theta)
        if self._binning_matrix is None:
            return depth
        return self._binning_matrix @ depth

    def chi_squared(self, theta: FloatTensor) -> FloatTensor:
        """Chi-squared against the observation.

        Parameters
        ----------
        theta:
            Fitted parameters in prior space

        Returns
        -------
        :obj:`torch.Tensor`
            Scalar chi-squared

        """
        residual = (self._data - self.spectrum(theta)) / self._error
        return torch.sum(residual * residual)

    def log_likelihood(self, theta: FloatTensor) -> FloatTensor:
        """Log likelihood, matching the convention of the taurex optimizers.

        Parameters
        ----------
        theta:
            Fitted parameters in prior space

        Returns
        -------
        :obj:`torch.Tensor`
            Scalar log likelihood

        """
        normalisation = torch.sum(torch.log(self._error)) + self._error.shape[0] * 0.5 * float(
            np.log(2.0 * np.pi)
        )
        return -normalisation - 0.5 * self.chi_squared(theta)
