"""Frequency grids, precision, and shared optical quantities."""

import jax
import jax.numpy as jnp


def _precision(dtype):
    if dtype == "float64":
        return jnp.float64, jnp.complex128
    if dtype == "float32":
        return jnp.float32, jnp.complex64
    raise ValueError('dtype must be "float64" or "float32"')


def frequency_axes(shape, spacing, *, dtype="float64"):
    """Return (fz, fy, fx), in cycles/length and unshifted FFT order.

    ``shape=(nz, ny, nx)`` and ``spacing=(dz, dy, dx)`` refer to the sample,
    not camera pixels. All lengths, including vacuum wavelength, must use
    the same unit. Returned arrays are real JAX arrays of the selected dtype.
    """
    real, _ = _precision(dtype)
    with jax.enable_x64(True):
        return tuple(
            jnp.fft.fftfreq(n, d=jnp.asarray(d, real), dtype=real)
            for n, d in zip(shape, spacing)
        )


def circular_pupil(fy, fx, *, wavelength, na, dtype="float64"):
    """Return a complex (ny, nx) pupil: fy**2 + fx**2 <= (NA/lambda0)**2.

    ``wavelength`` is the vacuum wavelength. Supply 1D FFT frequency axes
    in cycles/length, sampled finely enough to contain the pupil cutoff.
    """
    real, complex_ = _precision(dtype)
    with jax.enable_x64(True):
        fy, fx = jnp.asarray(fy, real), jnp.asarray(fx, real)
        cutoff = jnp.asarray(na / wavelength, real)
        return (fy[:, None] ** 2 + fx[None, :] ** 2 <= cutoff**2).astype(complex_)


def _prepare(shape, spacing, wavelength, n_medium, source, pupil, dtype):
    """Compute background, dispersion, and active modes for both methods."""
    real, complex_ = _precision(dtype)
    source = jnp.asarray(source, real)
    pupil = jnp.asarray(pupil, complex_)
    background = jnp.sum(source * jnp.abs(pupil) ** 2)
    if not bool(background > 0):
        raise ValueError("nonzero brightfield background I0=sum(source*abs(pupil)**2) required")

    _, fy, fx = frequency_axes(shape, spacing, dtype=dtype)
    radial2 = fy[:, None] ** 2 + fx[None, :] ** 2
    fm = jnp.asarray(n_medium / wavelength, real)
    active = (source > 0) | (jnp.abs(pupil) > 0)
    if bool(jnp.any(active & (radial2 >= fm**2))):
        raise ValueError("source and pupil must contain only strictly propagating modes")
    # Inactive frequencies may be evanescent; their pupil/source are zero.
    longitudinal = jnp.sqrt(jnp.maximum(fm**2 - radial2, 0))
    source_indices = jnp.nonzero((source > 0) & (jnp.abs(pupil) > 0))
    detector_indices = jnp.nonzero(jnp.abs(pupil) > 0)

    # All physical v-s AND s-v frequencies must fit, before modular indexing.
    for axis, n, si, vi in zip(
        ("y", "x"), shape[1:], source_indices, detector_indices
    ):
        bins = jnp.rint(jnp.fft.fftfreq(n, dtype=real) * n).astype(jnp.int32)
        s, v = bins[si], bins[vi]
        largest = jnp.maximum(jnp.max(v) - jnp.min(s), jnp.max(s) - jnp.min(v))
        if int(largest) > (n - 1) // 2:
            raise ValueError(f"transverse {axis} sampling aliases a pupil/source difference; reduce d{axis}")
    ws = longitudinal[source_indices]
    wv = longitudinal[detector_indices]
    largest_axial = jnp.maximum(jnp.max(wv) - jnp.min(ws), jnp.max(ws) - jnp.min(wv))
    if float(largest_axial) >= 0.5 / spacing[0]:
        raise ValueError("axial sampling aliases a cap frequency; reduce dz")
    return source, pupil, longitudinal, background, source_indices, detector_indices
