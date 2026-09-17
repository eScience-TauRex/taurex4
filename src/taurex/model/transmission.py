"""Transit forward model."""

import typing as t

import numpy as np
import numpy.typing as npt

from taurex.chemistry import Chemistry
from taurex.planet import Planet
from taurex.pressure import PressureProfile
from taurex.stellar import Star
from taurex.temperature import TemperatureProfile
from taurex.types import get_float_dtype

from .simplemodel import OneDForwardModel


if t.TYPE_CHECKING:
    from taurex.contributions import Contribution
else:
    Contribution = object


class TransmissionModel(OneDForwardModel):
    """A forward model for transits."""

    def __init__(
        self,
        planet: t.Optional[Planet] = None,
        star: t.Optional[Star] = None,
        pressure_profile: t.Optional[PressureProfile] = None,
        temperature_profile: t.Optional[TemperatureProfile] = None,
        chemistry: t.Optional[Chemistry] = None,
        nlayers: t.Optional[int] = 100,
        atm_min_pressure: t.Optional[float] = 1e-4,
        atm_max_pressure: t.Optional[float] = 1e6,
        contributions: t.Optional[t.List[Contribution]] = None,
        new_path_method: t.Optional[bool] = False,
    ) -> None:
        """Initialize transit forward model.

        Parameters
        ----------
        name: str
            Name to use in logging

        planet:
            Planet model, default planet is Jupiter

        star:
            Star model, default star is Sun-like

        pressure_profile:
            Pressure model, alternative is to set ``nlayers``, ``atm_min_pressure``
            and ``atm_max_pressure``

        temperature_profile:
            Temperature model, default is an
            :class:`~taurex.data.profiles.temperature.isothermal.Isothermal`
            profile at 1500 K

        chemistry:
            Chemistry model, default is
            :class:`~taurex.data.profiles.chemistry.taurexchemistry.TaurexChemistry`
            with ``H2O`` and ``CH4``

        nlayers: int, optional
            Number of layers. Used if ``pressure_profile`` is not defined.

        atm_min_pressure: float, optional
            Pressure at TOA. Used if ``pressure_profile`` is not defined.

        atm_max_pressure: float, optional
            Pressure at BOA. Used if ``pressure_profile`` is not defined.

        contributions: list, optional
            List of contributions to include

        new_path_method: bool, optional
            Use new path length computation method

        """
        super().__init__(
            self.__class__.__name__,
            planet,
            star,
            pressure_profile,
            temperature_profile,
            chemistry,
            nlayers,
            atm_min_pressure,
            atm_max_pressure,
            contributions,
        )
        self.new_method = new_path_method

    def compute_path_length_old(
        self, dz: npt.NDArray[np.float64]
    ) -> t.List[npt.NDArray[np.float64]]:
        """Compute path length for each layer."""
        dl = []

        planet_radius = self._planet.fullRadius
        total_layers = self.nLayers

        z = self.altitudeProfile
        self.debug("Computing path_length: \n z=%s \n dz=%s", z, dz)

        # Pre-allocate max-size k array, reuse with views to avoid
        # repeated allocation in the loop.
        _k_buf = np.empty(total_layers, dtype=get_float_dtype())

        for layer in range(0, total_layers):
            p = (planet_radius + dz[0] / 2 + z[layer]) ** 2
            n_remaining = total_layers - layer
            k = _k_buf[:n_remaining]
            k[0] = np.sqrt(
                (planet_radius + dz[0] / 2.0 + z[layer] + dz[layer] / 2.0) ** 2 - p
            )

            k[1:] = np.sqrt(
                (planet_radius + dz[0] / 2 + z[layer + 1 :] + dz[layer + 1 :] / 2) ** 2
                - p
            )

            k[1:] -= np.sqrt(
                (
                    planet_radius
                    + dz[0] / 2
                    + z[layer : self.nLayers - 1]
                    + dz[layer : self.nLayers - 1] / 2
                )
                ** 2
                - p
            )

            dl.append(k * 2.0)
        return dl

    def compute_path_matrix(
        self, dz: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Compute the path-length matrix.

        Rectangular form of :func:`compute_path_length_old`. Entry
        ``(layer, m)`` is the path length through the segment between altitude
        boundaries ``m`` and ``m + 1`` for a ray whose tangent layer is
        ``layer``, and zero where ``m < layer``.

        Parameters
        ----------
        dz:
            Layer thicknesses

        Returns
        -------
        :obj:`array`
            Path length matrix, ``(nlayers, nlayers)``

        """
        planet_radius = self._planet.fullRadius
        total_layers = self.nLayers

        z = self.altitudeProfile
        dz = np.asarray(dz, dtype=get_float_dtype())[:total_layers]

        self.debug("Computing path_matrix: \n z=%s \n dz=%s", z, dz)

        c0 = planet_radius + dz[0] / 2.0

        # u[q] = z[q] + dz[q]/2, padded to 2 * nlayers - 1 so that the
        # broadcast index layer + j never runs past the end.
        u = np.zeros(2 * total_layers - 1, dtype=get_float_dtype())
        u[:total_layers] = z + dz / 2.0

        layer = np.arange(total_layers)[:, None]
        j = np.arange(total_layers)[None, :]
        index = layer + j

        # Only j <= nlayers - 1 - layer is defined. Elsewhere the argument is
        # replaced by 1 before the square root, so the padding in u cannot
        # produce a negative radicand.
        defined = index < total_layers
        p = (c0 + z) ** 2
        argument = np.where(defined, (c0 + u[index]) ** 2 - p[:, None], 1.0)
        b = np.where(defined, np.sqrt(argument), 0.0)

        # b[layer, -1] is taken as zero
        previous = np.concatenate(
            [np.zeros((total_layers, 1), dtype=get_float_dtype()), b[:, :-1]],
            axis=1,
        )

        # The segment between boundaries m and m + 1 contributes
        # 2 * (b[layer, m - layer] - b[layer, m - layer - 1])
        m = np.arange(total_layers)[None, :]
        offset = np.clip(m - layer, 0, total_layers - 1)
        path_matrix = np.where(
            m >= layer, 2.0 * (b[layer, offset] - previous[layer, offset]), 0.0
        )

        return path_matrix

    def compute_path_length(self) -> t.List[npt.NDArray[np.float64]]:
        """Compute path length for each layer, new method."""
        from taurex.util.geometry import parallel_vector

        altitude_boundaries = self.altitude_boundaries
        radius = self.planet.fullRadius

        # Generate our line of sight paths
        viewer, tangent = parallel_vector(
            radius, self.altitude_profile + self.deltaz / 2, altitude_boundaries.max()
        )

        path_lengths = self.planet.compute_path_length(
            altitude_boundaries, viewer, tangent
        )

        return [l for _, l in path_lengths]

    def path_integral(
        self, wngrid: npt.NDArray[np.float64], return_contrib: t.Optional[bool] = False
    ) -> t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Compute path integral.

        Calculates the absorption and optical depth for each layer assuming
        hemispherical geometry.

        """
        dz = self.deltaz

        total_layers = self.nLayers

        wngrid_size = wngrid.shape[0]

        density_profile = self.densityProfile

        tau_dtype = get_float_dtype()

        if self.new_method:
            # compute_path_length() returns a filtered ragged list that does
            # not map onto the rectangular path matrix, so this keeps the loop.
            path_length = self.compute_path_length()
            self.path_length = path_length

            tau = np.zeros(shape=(total_layers, wngrid_size), dtype=tau_dtype)

            # Memory-efficient: prepare each contribution just before use,
            # then clean up its sigma_xsec immediately after all layers.
            for contrib in self.contribution_list:
                if contrib.sigma_xsec is None:
                    contrib.prepare(self, wngrid)

                for layer in range(total_layers):
                    self.debug("Computing layer %s", layer)
                    contrib.contribute(
                        self,
                        0,
                        total_layers - layer,
                        layer,
                        layer,
                        density_profile,
                        tau,
                        path_length=path_length[layer],
                    )
                # Free the large sigma_xsec array for this contribution
                del contrib.sigma_xsec
                contrib.sigma_xsec = None
        else:
            path_matrix = self.compute_path_matrix(dz)
            self.path_length = path_matrix

            tau = self.accumulate_path_matrix(wngrid, path_matrix, density_profile)

        self.debug("tau %s %s", tau, tau.shape)

        absorption, tau = self.compute_absorption(tau, dz)
        return absorption, tau

    def accumulate_path_matrix(
        self,
        wngrid: npt.NDArray[np.float64],
        path_matrix: npt.NDArray[np.float64],
        density_profile: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute the optical depth with the path-matrix product.

        Same result as the per-layer loop over contributions, but the
        cross-sections are summed before the product because tau is linear in
        them. The data-dependent early exit that used to skip optically thick
        layers is deliberately not reproduced: it made tau depend on the order
        of ``contribution_list``, and the terms it dropped were already below
        ``exp(-10)``.

        Parameters
        ----------
        wngrid:
            Wavenumber grid

        path_matrix:
            Path length through layer ``m`` for a ray tangent at layer ``l``,
            zero where ``m < l``

        density_profile:
            Atmospheric density in m-3

        Returns
        -------
        :obj:`array`
            Optical depth for each layer and wavelength

        """
        from taurex.contributions.contribution import accumulate_ktau_matrix

        nlayers = self.nLayers
        dtype = get_float_dtype()

        tau = np.zeros(shape=(nlayers, wngrid.shape[0]), dtype=dtype)

        # Cross-sections of the contributions that integrate along the line of
        # sight, combined before the product because tau is linear in them.
        # Everything below works in place against one reused scratch buffer, so
        # the whole accumulation allocates at most two extra (nlayers, ngrid)
        # arrays: the accumulator and the scratch.
        weighted_sigma = None
        scratch = None
        # Per-layer weights keyed by density power, built once per power and
        # shared between contributions. ``None`` means no weighting is needed.
        weights: t.Dict[int, t.Optional[npt.NDArray[np.float64]]] = {}
        # Contributions whose cross-section is already the optical depth.
        direct_tau = None
        # Only built if some contribution brings its own integration rule.
        legacy_path = None
        # Whether anything has already written into tau, which decides if the
        # product can be written straight into it.
        tau_written = False

        # Memory-efficient: prepare each contribution just before use,
        # then clean up its sigma_xsec immediately after use.
        for contrib in self.contribution_list:
            if contrib.sigma_xsec is None:
                contrib.prepare(self, wngrid)

            sigma = contrib.sigma_xsec

            if not contrib.uses_path_matrix:
                # Overrides contribute() with its own integration rule, so it
                # keeps the per-layer loop. It only adds to tau, so it still
                # composes with the product below.
                if legacy_path is None:
                    legacy_path = self.compute_path_length_old(self.deltaz)

                for layer in range(nlayers):
                    contrib.contribute(
                        self,
                        0,
                        nlayers - layer,
                        layer,
                        layer,
                        density_profile,
                        tau,
                        path_length=legacy_path[layer],
                    )
                tau_written = True
            elif contrib.direct_accumulation:
                if direct_tau is None:
                    direct_tau = np.zeros_like(tau)
                direct_tau += sigma
            else:
                weight = self._density_weight(
                    weights, density_profile, contrib.density_power
                )

                if contrib.has_gauss_axis:
                    # Correlated-k: the g-point reduction sits inside the sum
                    # over contributions, so it cannot be folded into the
                    # product and each one pays for its own.
                    tau += accumulate_ktau_matrix(
                        sigma if weight is None else sigma * weight[:, None, None],
                        path_matrix,
                        contrib.weights,
                    )
                    tau_written = True
                elif weight is None:
                    # No weighting, the cross-section is summed as it is.
                    if weighted_sigma is None:
                        weighted_sigma = np.array(sigma, dtype=dtype, copy=True)
                    else:
                        weighted_sigma += sigma
                elif weighted_sigma is None:
                    # The first weighted cross-section becomes the accumulator,
                    # so there is no zero-fill and no separate add pass.
                    weighted_sigma = sigma * weight[:, None]
                else:
                    if scratch is None:
                        scratch = np.empty(sigma.shape, dtype=dtype)
                    np.multiply(sigma, weight[:, None], out=scratch)
                    np.add(weighted_sigma, scratch, out=weighted_sigma)

            # Free the large sigma_xsec array for this contribution
            del contrib.sigma_xsec
            contrib.sigma_xsec = None

        if weighted_sigma is not None:
            if tau_written:
                tau += path_matrix @ weighted_sigma
            else:
                np.matmul(path_matrix, weighted_sigma, out=tau)
        if direct_tau is not None:
            tau += direct_tau

        return tau

    @staticmethod
    def _density_weight(
        cache: t.Dict[int, t.Optional[npt.NDArray[np.float64]]],
        density_profile: npt.NDArray[np.float64],
        power: int,
    ) -> t.Optional[npt.NDArray[np.float64]]:
        """Per-layer weight for a power of the density, computed at most once.

        The weight is a property of the atmosphere, not of the contribution, so
        every contribution that shares a power reuses the same array.

        Parameters
        ----------
        cache:
            Weights already computed, keyed by power

        density_profile:
            Atmospheric density in m-3

        power:
            Power of the density multiplying the cross-section

        Returns
        -------
        :obj:`array` or None
            Weight for each layer, or ``None`` when no weighting is needed

        """
        if power not in cache:
            if power == 0:
                cache[power] = None
            elif power == 1:
                cache[power] = density_profile
            elif power == 2:
                cache[power] = density_profile * density_profile
            else:
                cache[power] = density_profile**power
        return cache[power]

    def compute_absorption(
        self, tau: npt.NDArray[np.float64], dz: npt.NDArray[np.float64]
    ) -> t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Compute final absorption and optical depth."""
        # In-place exp to avoid temporary array allocation
        np.exp(-tau, out=tau)
        ap = self.altitudeProfile[:, None]
        pradius = self._planet.fullRadius
        sradius = self._star.radius
        _dz = dz[:, None]

        integral = np.sum((pradius + ap) * (1.0 - tau) * _dz * 2.0, axis=0)
        return ((pradius**2.0) + integral) / (sradius**2), tau

    @classmethod
    def input_keywords(cls) -> t.Tuple[str, ...]:
        """Input keywords for this class."""
        return (
            "transmission",
            "transit",
        )
