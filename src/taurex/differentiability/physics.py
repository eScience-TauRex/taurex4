"""Differentiable building blocks of the TauREx transmission forward model.

Every routine here works on ``torch`` tensors so that one call to
``backward()`` walks the entire atmosphere. The routines mirror the
corresponding numpy implementations in :mod:`taurex.util` and
:class:`taurex.model.transmission.TransmissionModel` closely enough to
reproduce their output, but the index lookups that ``searchsorted`` performs
are detached so only the interpolation weights carry gradients.
"""

import typing as t

import torch


FloatTensor = torch.Tensor


def as_tensor(
    value: t.Any,
    dtype: torch.dtype = torch.float64,
    device: t.Optional[torch.device] = None,
) -> FloatTensor:
    """Convert a python/array value to a tensor of the working dtype.

    Parameters
    ----------
    value:
        Value to convert

    dtype:
        Floating point dtype to use

    device:
        Device to place the tensor on

    Returns
    -------
    :obj:`torch.Tensor`
        Converted tensor

    """
    return torch.as_tensor(value, dtype=dtype, device=device)


def find_closest_pair(x: FloatTensor, value: FloatTensor) -> t.Tuple[FloatTensor, FloatTensor]:
    """Find the indices either side of ``value`` in the sorted array ``x``.

    Port of :func:`taurex.util.find_closest_pair`. The result is clipped so
    that both indices are always valid, which is what the numpy version does
    for values outside the grid.

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
    right = torch.searchsorted(x.detach(), value.detach())
    right = right.clamp(1, x.shape[0] - 1)
    left = (right - 1).clamp(min=0)
    return left, right


def _safe_ratio(numerator: FloatTensor, denominator: FloatTensor) -> FloatTensor:
    """Divide, replacing a zero denominator with one."""
    return numerator / torch.where(
        denominator == 0,
        torch.ones_like(denominator),
        denominator,
    )


def interp_bilin(
    x11: FloatTensor,
    x12: FloatTensor,
    x21: FloatTensor,
    x22: FloatTensor,
    temperature: FloatTensor,
    temperature_min: FloatTensor,
    temperature_max: FloatTensor,
    pressure: FloatTensor,
    pressure_min: FloatTensor,
    pressure_max: FloatTensor,
) -> FloatTensor:
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
    :obj:`torch.Tensor`
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
    x11: FloatTensor,
    x12: FloatTensor,
    pressure: FloatTensor,
    pressure_min: FloatTensor,
    pressure_max: FloatTensor,
) -> FloatTensor:
    """Linear pressure interpolation, matching :func:`taurex.util.math.interp_lin_only`."""
    scale = _safe_ratio(pressure - pressure_min, pressure_max - pressure_min)
    return x11 - scale * (x11 - x12)


def interp_exp_only(
    x11: FloatTensor,
    x12: FloatTensor,
    temperature: FloatTensor,
    temperature_min: FloatTensor,
    temperature_max: FloatTensor,
) -> FloatTensor:
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
    :obj:`torch.Tensor`
        Interpolated values

    """
    return x11 * torch.exp(
        _safe_ratio(
            temperature_max * (temperature_min - temperature) * torch.log(x11 / x12),
            temperature * (temperature_max - temperature_min),
        )
    )


def interp_exp_and_lin(
    x11: FloatTensor,
    x12: FloatTensor,
    x21: FloatTensor,
    x22: FloatTensor,
    temperature: FloatTensor,
    temperature_min: FloatTensor,
    temperature_max: FloatTensor,
    pressure: FloatTensor,
    pressure_min: FloatTensor,
    pressure_max: FloatTensor,
) -> FloatTensor:
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
    :obj:`torch.Tensor`
        Interpolated values

    """
    pressure_diff = pressure_max - pressure_min
    low = x11 * pressure_diff - (pressure - pressure_min) * (x11 - x21)
    high = x12 * pressure_diff - (pressure - pressure_min) * (x12 - x22)
    return (
        low
        * torch.exp(
            _safe_ratio(
                temperature_max
                * (temperature_min - temperature)
                * torch.log(_safe_ratio(low, high)),
                temperature * (temperature_max - temperature_min),
            )
        )
        / pressure_diff
    )


def boxcar(a: FloatTensor, n: int) -> FloatTensor:
    """Moving average with window ``n``.

    Port of :func:`taurex.util.movingaverage` that keeps the autograd graph
    intact instead of writing into the cumulative sum in place.

    Parameters
    ----------
    a:
        Array to smooth

    n:
        Window size, must be at least one

    Returns
    -------
    :obj:`torch.Tensor`
        Smoothed array of length ``len(a) - n + 1``

    """
    if n < 1:
        raise ValueError(f"Window size must be at least 1, got {n}")
    zero = torch.zeros(1, dtype=a.dtype, device=a.device)
    cumulative = torch.cat([zero, torch.cumsum(a, dim=0)])
    return (cumulative[n:] - cumulative[:-n]) / n


def linear_interp_nd(
    x: FloatTensor,
    xp: FloatTensor,
    fp: FloatTensor,
) -> FloatTensor:
    """Piecewise linear interpolation with edge clamping.

    Reproduces :func:`numpy.interp` for ``x`` against nodes ``xp`` with values
    ``fp``. The knot positions are only used to pick the bracketing pair and
    to form the interpolation weight, so ``fp`` may require grad.

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
    :obj:`torch.Tensor`
        Interpolated values, shape ``(n,)`` or ``(n, k)``

    """
    n_nodes = xp.shape[0]
    upper = torch.searchsorted(xp.detach(), x.detach(), right=True).clamp(1, n_nodes - 1)
    lower = upper - 1

    x_lo = xp[lower]
    x_hi = xp[upper]
    weight = _safe_ratio(x - x_lo, x_hi - x_lo).reshape(
        x.shape + (1,) * (fp.dim() - 1)
    )

    y_lo = fp[lower]
    y_hi = fp[upper]
    result = y_lo + weight * (y_hi - y_lo)

    result = torch.where((x <= xp[0]).reshape(x.shape + (1,) * (fp.dim() - 1)), fp[0], result)
    result = torch.where(
        (x >= xp[-1]).reshape(x.shape + (1,) * (fp.dim() - 1)), fp[-1], result
    )
    return result


def npoint_temperature(
    pressure_profile: FloatTensor,
    pressure_nodes: FloatTensor,
    temperature_nodes: FloatTensor,
    smooth_window: int,
) -> FloatTensor:
    """Temperature profile from user points, smoothed.

    Port of :meth:`taurex.data.profiles.temperature.npoint.NPoint.profile`.
    The pressure nodes and temperature nodes are ordered from the surface to
    the top of the atmosphere, as they are in the numpy implementation.

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
    :obj:`torch.Tensor`
        Temperature at each layer centre in K

    """
    if torch.all(temperature_nodes == temperature_nodes[0]):
        return torch.ones_like(pressure_profile) * temperature_nodes[0]

    log_pressure = torch.log10(pressure_profile.flip(0))
    log_nodes = torch.log10(pressure_nodes.flip(0))
    nodes = temperature_nodes.flip(0)

    profile = linear_interp_nd(log_pressure, log_nodes, nodes)

    n_layers = pressure_profile.shape[0]
    window = int(n_layers * (smooth_window / 100.0))
    if window % 2 == 0:
        window += 1

    smoothed = boxcar(profile, window)
    border = int((profile.shape[0] - smoothed.shape[0]) / 2)

    reversed_profile = profile.flip(0)
    if smoothed.shape[0] == reversed_profile.shape[0]:
        return smoothed.flip(0)

    result = reversed_profile.clone()
    result[border : result.shape[0] - border] = smoothed.flip(0)
    return result


def altitude_gravity_scaleheight(
    planet_radius: FloatTensor,
    planet_mass: FloatTensor,
    temperature: FloatTensor,
    mu: FloatTensor,
    pressure_levels: FloatTensor,
) -> FloatTensor:
    r"""Solve the hydrostatic profile for altitude, gravity and layer thickness.

    Port of :meth:`taurex.data.planet.BasePlanet.calculate_scale_properties`.
    The layer thickness at each boundary depends on the scale height of the
    layer below it, which in turn depends on the altitude below it, so the
    recursion cannot be vectorised without a scan. ``nlayers`` is of order one
    hundred, so the python loop is left in place.

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
    zero = torch.zeros((), dtype=temperature.dtype, device=temperature.device)

    surface_gravity = (G * planet_mass) / planet_radius**2
    scaleheight = [(KBOLTZ * temperature[0]) / (mu[0] * surface_gravity)]
    gravity = [surface_gravity]
    altitude = [zero]
    deltaz = [zero]

    # Written with lists rather than in-place assignment into preallocated
    # tensors: the recursion reads altitude[i - 1] and writes altitude[i], and
    # autograd refuses in-place updates to a tensor it still needs.
    for i in range(1, n_layers + 1):
        step = -scaleheight[i - 1] * torch.log(
            pressure_levels[i] / pressure_levels[i - 1]
        )
        deltaz.append(step)
        altitude.append(altitude[i - 1] + step)
        if i < n_layers:
            gravity.append((G * planet_mass) / (planet_radius + altitude[i]) ** 2)
            scaleheight.append((KBOLTZ * temperature[i]) / (mu[i] * gravity[i]))

    return (
        torch.stack(altitude),
        torch.stack(scaleheight),
        torch.stack(gravity),
        torch.stack(deltaz[1:]),
    )


def path_matrix(
    altitude: FloatTensor,
    deltaz: FloatTensor,
    planet_radius: FloatTensor,
) -> FloatTensor:
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
    :obj:`torch.Tensor`
        Path-length matrix, shape ``(nlayers, nlayers)``

    """
    n_layers = altitude.shape[0]
    deltaz = deltaz[:n_layers]

    c0 = planet_radius + deltaz[0] / 2.0

    # u[q] = z[q] + dz[q] / 2, padded so the broadcast index layer + j stays
    # inside the array.
    u = torch.cat(
        [
            altitude + deltaz / 2.0,
            torch.zeros(n_layers - 1, dtype=altitude.dtype, device=altitude.device),
        ]
    )

    layer = torch.arange(n_layers, device=altitude.device)[:, None]
    index = layer + torch.arange(n_layers, device=altitude.device)[None, :]

    defined = index < n_layers
    p = (c0 + altitude) ** 2
    argument = torch.where(
        defined, (c0 + u[index]) ** 2 - p[:, None], torch.ones_like(p[:, None])
    )
    b = torch.where(defined, torch.sqrt(argument), torch.zeros_like(argument))

    padded = torch.cat(
        [torch.zeros(n_layers, 1, dtype=b.dtype, device=b.device), b], dim=1
    )

    # The segment between boundaries m and m + 1 contributes
    # 2 * (b[layer, m - layer] - b[layer, m - layer - 1]), and padded[:, k] is
    # b[:, k - 1], so one gather of each array at the shifted index gives both
    # terms.
    m = torch.arange(n_layers, device=altitude.device)[None, :]
    offset = (m - layer).clamp(0, n_layers - 1)
    current = torch.gather(b, 1, offset)
    previous = torch.gather(padded, 1, offset)
    return torch.where(m >= layer, 2.0 * (current - previous), torch.zeros_like(b))
