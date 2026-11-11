"""Wavevector grids and shared optical quantities."""

import warnings

import jax.numpy as jnp


def k_axes(shape, spacing):
    """Return (kz, ky, kx), in rad/length and unshifted FFT order.

    ``shape=(nz, ny, nx)`` and ``spacing=(dz, dy, dx)`` refer to the sample,
    not camera pixels. All lengths, including vacuum wavelength, must use
    the same unit. JAX's x64 configuration sets the grid dtype.
    """
    return tuple(2 * jnp.pi * jnp.fft.fftfreq(n, d=d) for n, d in zip(shape, spacing))


def circular_pupil(ky, kx, *, wavelength, na):
    """Return a complex (ny, nx) pupil: ky**2+kx**2 <= (2*pi*NA/lambda0)**2.

    ``wavelength`` is the vacuum wavelength. Supply 1D FFT frequency axes
    in rad/length, sampled finely enough to contain the pupil cutoff.
    """
    ky, kx = jnp.asarray(ky), jnp.asarray(kx)
    cutoff = 2 * jnp.pi * na / wavelength
    mask = ky[:, None] ** 2 + kx[None, :] ** 2 <= cutoff**2
    return mask.astype(jnp.result_type(ky, kx, 1j))


def _prepare(shape, spacing, wavelength, n_medium, source, pupil):
    """Compute background, dispersion, and active modes for both methods."""
    source = jnp.asarray(source)
    pupil = jnp.asarray(pupil) + 0j
    detected = pupil != 0
    if bool(jnp.any((source > 0) & ~detected)):
        warnings.warn(
            "Source modes outside the detection pupil were discarded; "
            "they have no linear brightfield contribution.",
            UserWarning, stacklevel=3,
        )
    source = jnp.where(detected, source, 0)
    background = jnp.sum(source * jnp.abs(pupil) ** 2)
    if not bool(background > 0):
        raise ValueError("nonzero brightfield background I0=sum(source*abs(pupil)**2) required")

    _, ky, kx = k_axes(shape, spacing)
    radial2 = ky[:, None] ** 2 + kx[None, :] ** 2
    km = 2 * jnp.pi * n_medium / wavelength
    active = (source > 0) | detected
    if bool(jnp.any(active & (radial2 >= km**2))):
        raise ValueError("source and pupil must contain only strictly propagating modes")
    # Inactive frequencies may be evanescent; their pupil/source are zero.
    kz = jnp.sqrt(jnp.maximum(km**2 - radial2, 0))
    source_indices = jnp.nonzero(source > 0)
    detector_indices = jnp.nonzero(detected)

    # All physical v-s AND s-v frequencies must fit, before modular indexing.
    for axis, n, si, vi in zip(
        ("y", "x"), shape[1:], source_indices, detector_indices
    ):
        bins = jnp.rint(jnp.fft.fftfreq(n) * n).astype(jnp.int32)
        s, v = bins[si], bins[vi]
        largest = jnp.maximum(jnp.max(v) - jnp.min(s), jnp.max(s) - jnp.min(v))
        if int(largest) > (n - 1) // 2:
            raise ValueError(f"transverse {axis} sampling aliases a pupil/source difference; reduce d{axis}")
    ks = kz[source_indices]
    kv = kz[detector_indices]
    largest_axial = jnp.maximum(jnp.max(kv) - jnp.min(ks), jnp.max(ks) - jnp.min(kv))
    if float(largest_axial) >= jnp.pi / spacing[0]:
        raise ValueError("axial sampling aliases a cap frequency; reduce dz")
    return source, pupil, kz, background, source_indices, detector_indices
