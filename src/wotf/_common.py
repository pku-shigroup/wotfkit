"""Wavevector grids, precision, and shared optical quantities."""

from contextlib import contextmanager
from contextvars import ContextVar
import warnings

import jax
import jax.numpy as jnp


_default_dtype = ContextVar("wotf_dtype", default="float64")


def _precision(dtype=None):
    dtype = _default_dtype.get() if dtype is None else dtype
    if dtype == "float64":
        return jnp.float64, jnp.complex128
    if dtype == "float32":
        return jnp.float32, jnp.complex64
    raise ValueError('dtype must be "float64" or "float32"')


@contextmanager
def precision(dtype):
    """Select float32 or float64 for a block of WOTF and JAX calculations.

    The default outside a block is float64. An explicit function ``dtype``
    overrides the block; nested blocks and exceptions restore prior settings.
    JAX's local x64 setting also controls arrays created in the block.
    """
    real, _ = _precision(dtype)
    token = _default_dtype.set(jnp.dtype(real).name)
    try:
        with jax.enable_x64(real == jnp.float64):
            yield
    finally:
        _default_dtype.reset(token)


def k_axes(shape, spacing, *, dtype=None):
    """Return (kz, ky, kx), in rad/length and unshifted FFT order.

    ``shape=(nz, ny, nx)`` and ``spacing=(dz, dy, dx)`` refer to the sample,
    not camera pixels. All lengths, including vacuum wavelength, must use
    the same unit. Returned arrays are real JAX arrays of the selected dtype.
    """
    real, _ = _precision(dtype)
    with jax.enable_x64(real == jnp.float64):
        return tuple(
            2 * jnp.pi * jnp.fft.fftfreq(n, d=jnp.asarray(d, real), dtype=real)
            for n, d in zip(shape, spacing)
        )


def circular_pupil(ky, kx, *, wavelength, na, dtype=None):
    """Return a complex (ny, nx) pupil: ky**2+kx**2 <= (2*pi*NA/lambda0)**2.

    ``wavelength`` is the vacuum wavelength. Supply 1D FFT frequency axes
    in rad/length, sampled finely enough to contain the pupil cutoff.
    """
    real, complex_ = _precision(dtype)
    with jax.enable_x64(real == jnp.float64):
        ky, kx = jnp.asarray(ky, real), jnp.asarray(kx, real)
        cutoff = jnp.asarray(2 * jnp.pi * na / wavelength, real)
        return (ky[:, None] ** 2 + kx[None, :] ** 2 <= cutoff**2).astype(complex_)


def _prepare(shape, spacing, wavelength, n_medium, source, pupil, dtype):
    """Compute background, dispersion, and active modes for both methods."""
    real, complex_ = _precision(dtype)
    source = jnp.asarray(source, real)
    pupil = jnp.asarray(pupil, complex_)
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

    _, ky, kx = k_axes(shape, spacing, dtype=dtype)
    radial2 = ky[:, None] ** 2 + kx[None, :] ** 2
    km = jnp.asarray(2 * jnp.pi * n_medium / wavelength, real)
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
        bins = jnp.rint(jnp.fft.fftfreq(n, dtype=real) * n).astype(jnp.int32)
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
