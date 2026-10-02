"""Differentiable building blocks of the TauREx transmission forward model.

Every routine works on ``jax`` arrays so that one call to :func:`jax.grad`
walks the entire atmosphere, and every routine is traceable so the forward model
can be compiled as a whole with :func:`jax.jit`. The routines mirror the
corresponding numpy implementations in :mod:`taurex.util` and
:class:`taurex.model.transmission.TransmissionModel` closely enough to
reproduce their output, but the index lookups that ``searchsorted`` performs
are detached so only the interpolation weights carry gradients.

JAX arrays are immutable, so nothing is written in place: the single place
where the numpy code fills a buffer uses a functional ``.at[].set``, and the
hydrostatic recursion - which the numpy code runs as a python loop - is a
:func:`jax.lax.scan`, so the compiled program contains a loop rather than
``nlayers`` unrolled copies of it.
"""

import typing as t

import jax
import jax.numpy as jnp


Array = jax.Array


def as_array(value: t.Any, dtype: jnp.dtype = jnp.float64) -> Array:
    """Convert a python/array value to an array of the working dtype.

    Parameters
    ----------
    value:
        Value to convert

    dtype:
        Floating point dtype to use

    Returns
    -------
    :obj:`jax.Array`
        Converted array

    """
    return jnp.asarray(value, dtype=dtype)


def find_closest_pair(x: Array, value: Array) -> t.Tuple[Array, Array]:
    """Find the indices either side of ``value`` in the sorted array ``x``.

    Port of :func:`taurex.util.find_closest_pair`. The result is clipped so
    that both indices are always valid, which is what the numpy version does
    for values outside the grid.

    Both indices are integer valued, so they can only ever contribute a zero
    derivative; ``stop_gradient`` states that explicitly and keeps the
    interpolation weights the only differentiable path.

    Parameters
    ----------
    x:
        Sorted array to search

    value:
        Value to locate

    Returns
    -------
    left, right:
        Indices with ``x[left] <= value <= x[right]``

    """
    right = jnp.searchsorted(x, jax.lax.stop_gradient(value))
    right = jnp.clip(right, 1, x.shape[0] - 1)
    left = jnp.clip(right - 1, 0, x.shape[0] - 1)
    return left, right


def _safe_ratio(numerator: Array, denominator: Array) -> Array:
    """Divide, replacing a zero denominator with one."""
    return numerator / jnp.where(
        denominator == 0,
        jnp.ones_like(denominator),
        denominator,
    )


def interp_bilin(
    x11: Array,
    x12: Array,
    x21: Array,
    x22: Array,
    temperature: Array,
    temperature_min: Array,
    temperature_max: Array,
    pressure: Array,
    pressure_min: Array,
    pressure_max: Array,
) -> Array:
    """Bilinear interpolation, matching :func:`taurex.util.math.intepr_bilin`.

    Parameters
    ----------
    x11, x12, x21, x22:
        Corner values for ``(p_min, t_min)``, ``(p_min, t_max)``,
        ``(p_max, t_min)`` and ``(p_max, t_max)``

    temperature:
        Coordinate to interpolate to

    temperature_min, temperature_max:
        Bounding temperature grid points

    pressure:
        Coordinate to interpolate to

    pressure_min, pressure_max:
        Bounding pressure grid points

    Returns
    -------
    :obj:`jax.Array`
        Interpolated values

    """
    pressure_diff = pressure_max - pressure_min
    temperature_diff = temperature_max - temperature_min
    pressure_scale = _safe_ratio(pressure - pressure_min, pressure_diff)
    temperature_scale = _safe_ratio(temperature - temperature_min, temperature_diff)

    return (
        x11
        - pressure_scale * (x11 - x21)
        - pressure_scale * temperature_scale * (x21 - x11 + x12 - x22)
        - temperature_scale * (x11 - x12)
    )


def interp_lin(
    x11: Array,
    x12: Array,
    pressure: Array,
    pressure_min: Array,
    pressure_max: Array,
) -> Array:
    """Linear pressure interpolation, matching :func:`taurex.util.math.interp_lin_only`."""
    scale = _safe_ratio(pressure - pressure_min, pressure_max - pressure_min)
    return x11 - scale * (x11 - x12)


def interp_exp_only(
    x11: Array,
    x12: Array,
    temperature: Array,
    temperature_min: Array,
    temperature_max: Array,
) -> Array:
    """Exponential temperature interpolation.

    Matches :func:`taurex.util.math.interp_exp_only`.

    Parameters
    ----------
    x11, x12:
        Values at ``temperature_min`` and ``temperature_max``

    temperature:
        Coordinate to interpolate to

    temperature_min, temperature_max:
        Bounding temperature grid points

    Returns
    -------
    :obj:`jax.Array`
        Interpolated values

    """
    return x11 * jnp.exp(
        _safe_ratio(
            temperature_max * (temperature_min - temperature) * jnp.log(x11 / x12),
            temperature * (temperature_max - temperature_min),
        )
    )


def interp_exp_and_lin(
    x11: Array,
    x12: Array,
    x21: Array,
    x22: Array,
    temperature: Array,
    temperature_min: Array,
    temperature_max: Array,
    pressure: Array,
    pressure_min: Array,
    pressure_max: Array,
) -> Array:
    """Exp-in-temperature, linear-in-pressure interpolation.

    Matches :func:`taurex.util.math.interp_exp_and_lin`.

    Parameters
    ----------
    x11, x12, x21, x22:
        Corner values for ``(p_min, t_min)``, ``(p_min, t_max)``,
        ``(p_max, t_min)`` and ``(p_max, t_max)``

    temperature:
        Coordinate to interpolate to

    temperature_min, temperature_max:
        Bounding temperature grid points

    pressure:
        Coordinate to interpolate to

    pressure_min, pressure_max:
        Bounding pressure grid points

    Returns
    -------
    :obj:`jax.Array`
        Interpolated values

    """
    pressure_diff = pressure_max - pressure_min
    low = x11 * pressure_diff - (pressure - pressure_min) * (x11 - x21)
    high = x12 * pressure_diff - (pressure - pressure_min) * (x12 - x22)
    return (
        low
        * jnp.exp(
            _safe_ratio(
                temperature_max
                * (temperature_min - temperature)
                * jnp.log(_safe_ratio(low, high)),
                temperature * (temperature_max - temperature_min),
            )
        )
        / pressure_diff
    )


def boxcar(a: Array, n: int) -> Array:
    """Moving average with window ``n``.

    Port of :func:`taurex.util.movingaverage` that keeps the autodiff graph
    intact instead of writing into the cumulative sum in place.

    Parameters
    ----------
    a:
        Array to smooth

    n:
        Window size, must be at least one

    Returns
    -------
    :obj:`jax.Array`
        Smoothed array of length ``len(a) - n + 1``

    """
    if n < 1:
        raise ValueError(f"Window size must be at least 1, got {n}")
    zero = jnp.zeros(1, dtype=a.dtype)
    cumulative = jnp.concatenate([zero, jnp.cumsum(a, axis=0)])
    return (cumulative[n:] - cumulative[:-n]) / n


def linear_interp_nd(
    x: Array,
    xp: Array,
    fp: Array,
) -> Array:
    """Piecewise linear interpolation with edge clamping.

    Reproduces :func:`numpy.interp` for ``x`` against nodes ``xp`` with values
    ``fp``. The knot positions are only used to pick the bracketing pair and
    to form the interpolation weight, so ``fp`` may carry a gradient.

    Parameters
    ----------
    x:
        Sample points, shape ``(n,)``

    xp:
        Increasing node positions, shape ``(m,)``

    fp:
        Node values, shape ``(m,)`` or ``(m, k)``

    Returns
    -------
    :obj:`jax.Array`
        Interpolated values, shape ``(n,)`` or ``(n, k)``

    """
    n_nodes = xp.shape[0]
    upper = jnp.clip(
        jnp.searchsorted(
            jax.lax.stop_gradient(xp), jax.lax.stop_gradient(x), side="right"
        ),
        1,
        n_nodes - 1,
    )
    lower = upper - 1

    x_lo = xp[lower]
    x_hi = xp[upper]
    weight = _safe_ratio(x - x_lo, x_hi - x_lo).reshape(
        x.shape + (1,) * (fp.ndim - 1)
    )

    y_lo = fp[lower]
    y_hi = fp[upper]
    result = y_lo + weight * (y_hi - y_lo)

    shape = x.shape + (1,) * (fp.ndim - 1)
    result = jnp.where((x <= xp[0]).reshape(shape), fp[0], result)
    result = jnp.where((x >= xp[-1]).reshape(shape), fp[-1], result)
    return result


def npoint_temperature(
    pressure_profile: Array,
    pressure_nodes: Array,
    temperature_nodes: Array,
    smooth_window: int,
) -> Array:
    """Temperature profile from user points, smoothed.

    Port of :meth:`taurex.data.profiles.temperature.npoint.NPoint.profile`.
    The pressure nodes and temperature nodes are ordered from the surface to
    the top of the atmosphere, as they are in the numpy implementation.

    The numpy version returns a constant profile when every node temperature is
    the same. That test looks at the values themselves rather than at their
    shapes, so under tracing it is a :func:`jax.numpy.where` rather than a
    python ``if``; the interpolation it guards against is written with
    :func:`_safe_ratio` so that the discarded branch is finite and its
    gradient stays finite too.

    Parameters
    ----------
    pressure_profile:
        Layer centre pressures in Pa, shape ``(nlayers,)``

    pressure_nodes:
        Node pressures in Pa, shape ``(nnodes,)``

    temperature_nodes:
        Node temperatures in K, shape ``(nnodes,)``

    smooth_window:
        Smoothing window as a percentage of the layer count

    Returns
    -------
    :obj:`jax.Array`
        Temperature at each layer centre in K

    """
    isothermal = jnp.all(temperature_nodes == temperature_nodes[0])

    log_pressure = jnp.log10(pressure_profile[::-1])
    log_nodes = jnp.log10(pressure_nodes[::-1])
    nodes = temperature_nodes[::-1]

    profile = linear_interp_nd(log_pressure, log_nodes, nodes)

    n_layers = pressure_profile.shape[0]
    window = int(n_layers * (smooth_window / 100.0))
    if window % 2 == 0:
        window += 1

    smoothed = boxcar(profile, window)
    border = int((profile.shape[0] - smoothed.shape[0]) / 2)

    reversed_profile = profile[::-1]
    if smoothed.shape[0] == reversed_profile.shape[0]:
        flat = smoothed[::-1]
    else:
        flat = reversed_profile.at[border : reversed_profile.shape[0] - border].set(
            smoothed[::-1]
        )

    return jnp.where(
        isothermal,
        jnp.ones_like(pressure_profile) * temperature_nodes[0],
        flat,
    )


def altitude_gravity_scaleheight(
    planet_radius: Array,
    planet_mass: Array,
    temperature: Array,
    mu: Array,
    pressure_levels: Array,
) -> t.Tuple[Array, Array, Array, Array]:
    r"""Solve the hydrostatic profile for altitude, gravity and layer thickness.

    Port of :meth:`taurex.data.planet.BasePlanet.calculate_scale_properties`.
    The layer thickness at each boundary depends on the scale height of the
    layer below it, which in turn depends on the altitude below it, so the
    recursion cannot be vectorised without a scan. ``nlayers`` is of order one
    hundred, so a :func:`jax.lax.scan` is used: the compiled program holds one
    copy of the step rather than ``nlayers`` of them.

    Parameters
    ----------
    planet_radius:
        Planet radius at the surface in m

    planet_mass:
        Planet mass in kg

    temperature:
        Temperature at each layer centre in K

    mu:
        Mean molecular weight at each layer in kg

    pressure_levels:
        Pressure at each layer boundary in Pa, ordered from the surface up

    Returns
    -------
    altitude:
        Altitude of each layer boundary in m, shape ``(nlayers + 1,)``

    scaleheight:
        Scale height of each layer in m, shape ``(nlayers,)``

    gravity:
        Gravity at each layer in m/s^2, shape ``(nlayers,)``

    deltaz:
        Thickness of each layer in m, shape ``(nlayers,)``

    """
    from taurex.constants import G, KBOLTZ

    n_layers = temperature.shape[0]
    zero = jnp.zeros((), dtype=temperature.dtype)

    surface_gravity = (G * planet_mass) / planet_radius**2
    first_scaleheight = (KBOLTZ * temperature[0]) / (mu[0] * surface_gravity)

    def step(carry, index):
        """Advance the recursion by one layer."""
        altitude_below, scaleheight_below = carry
        deltaz = -scaleheight_below * jnp.log(
            pressure_levels[index] / pressure_levels[index - 1]
        )
        altitude = altitude_below + deltaz
        # There is no layer above the last boundary, so the gravity and scale
        # height computed for it are dropped when the outputs are assembled.
        within = jnp.clip(index, 0, n_layers - 1)
        gravity = (G * planet_mass) / (planet_radius + altitude) ** 2
        scaleheight = (KBOLTZ * temperature[within]) / (mu[within] * gravity)
        return (altitude, scaleheight), (deltaz, altitude, gravity, scaleheight)

    _, (deltaz, upper_altitude, upper_gravity, upper_scaleheight) = jax.lax.scan(
        step, (zero, first_scaleheight), jnp.arange(1, n_layers + 1)
    )

    return (
        jnp.concatenate([jnp.zeros((1,), dtype=temperature.dtype), upper_altitude]),
        jnp.concatenate([first_scaleheight[None], upper_scaleheight[:-1]]),
        jnp.concatenate([surface_gravity[None], upper_gravity[:-1]]),
        deltaz,
    )


def path_matrix(
    altitude: Array,
    deltaz: Array,
    planet_radius: Array,
) -> Array:
    r"""Chord length through every layer for every tangent ray.

    Port of :meth:`taurex.model.transmission.TransmissionModel.compute_path_matrix`.
    Entry ``(layer, m)`` is the path length through the segment between
    altitude boundaries ``m`` and ``m + 1`` for a ray whose tangent layer is
    ``layer``, and zero where ``m < layer``.

    Parameters
    ----------
    altitude:
        Altitude at each layer centre in m, shape ``(nlayers,)``

    deltaz:
        Layer thicknesses in m, shape ``(nlayers,)``

    planet_radius:
        Planet radius in m

    Returns
    -------
    :obj:`jax.Array`
        Path-length matrix, shape ``(nlayers, nlayers)``

    """
    n_layers = altitude.shape[0]
    deltaz = deltaz[:n_layers]

    c0 = planet_radius + deltaz[0] / 2.0

    # u[q] = z[q] + dz[q] / 2, padded so the broadcast index layer + j stays
    # inside the array.
    u = jnp.concatenate(
        [
            altitude + deltaz / 2.0,
            jnp.zeros(n_layers - 1, dtype=altitude.dtype),
        ]
    )

    layer = jnp.arange(n_layers)[:, None]
    index = layer + jnp.arange(n_layers)[None, :]

    defined = index < n_layers
    p = (c0 + altitude) ** 2
    argument = jnp.where(
        defined, (c0 + u[index]) ** 2 - p[:, None], jnp.ones_like(p[:, None])
    )
    b = jnp.where(defined, jnp.sqrt(argument), jnp.zeros_like(argument))

    padded = jnp.concatenate([jnp.zeros((n_layers, 1), dtype=b.dtype), b], axis=1)

    # The segment between boundaries m and m + 1 contributes
    # 2 * (b[layer, m - layer] - b[layer, m - layer - 1]), and padded[:, k] is
    # b[:, k - 1], so one gather of each array at the shifted index gives both
    # terms.
    m = jnp.arange(n_layers)[None, :]
    offset = jnp.clip(m - layer, 0, n_layers - 1)
    current = jnp.take_along_axis(b, offset, axis=1)
    previous = jnp.take_along_axis(padded, offset, axis=1)
    return jnp.where(m >= layer, 2.0 * (current - previous), jnp.zeros_like(b))
