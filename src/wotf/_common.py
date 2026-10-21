"""Grid conventions, explicit precision, and input checks (no global settings)."""

import math
from numbers import Integral

import jax
import jax.numpy as jnp


def _precision(dtype):
    if dtype == "float64":
        return jnp.float64, jnp.complex128
    if dtype == "float32":
        return jnp.float32, jnp.complex64
    raise ValueError('dtype must be "float64" or "float32"')


def _positive(value, name):
    if not math.isfinite(float(value)) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _grid_parameters(shape, spacing):
    if len(shape) != 3 or any(
        isinstance(n, bool) or not isinstance(n, Integral) or n < 1 for n in shape
    ):
        raise ValueError("shape must contain three positive integers (nz, ny, nx)")
    if len(spacing) != 3:
        raise ValueError("spacing must contain (dz, dy, dx)")
    for d in spacing:
        _positive(d, "spacing")
    return tuple(int(n) for n in shape), tuple(float(d) for d in spacing)


def frequency_axes(shape, spacing, *, dtype="float64"):
    """Return (fz, fy, fx), in cycles/length and unshifted FFT order.

    ``shape=(nz, ny, nx)`` and ``spacing=(dz, dy, dx)`` refer to the sample,
    not camera pixels. All lengths, including vacuum wavelength, must use
    the same unit. Returned arrays are real JAX arrays of the selected dtype.
    """
    real, _ = _precision(dtype)
    shape, spacing = _grid_parameters(shape, spacing)
    with jax.enable_x64(True):
        return tuple(
            jnp.fft.fftfreq(n, d=jnp.asarray(d, real), dtype=real)
            for n, d in zip(shape, spacing)
        )


def circular_pupil(fy, fx, *, wavelength, na, dtype="float64"):
    """Return a complex (ny, nx) pupil: fy**2 + fx**2 <= (NA/lambda0)**2.

    ``wavelength`` is the vacuum wavelength. Axes must be unshifted uniform
    FFT axes with at least two entries. The cutoff must fit below their
    Nyquist limits. The WOTF functions also check propagating active modes.
    """
    real, complex_ = _precision(dtype)
    _positive(wavelength, "wavelength")
    _positive(na, "na")
    with jax.enable_x64(True):
        fy, fx = jnp.asarray(fy, real), jnp.asarray(fx, real)
        if fy.ndim != 1 or fx.ndim != 1 or fy.size == 0 or fx.size == 0:
            raise ValueError("fy and fx must be nonempty 1D frequency axes")
        if not bool(jnp.all(jnp.isfinite(fy))) or not bool(jnp.all(jnp.isfinite(fx))):
            raise ValueError("frequency axes must be finite")
        cutoff = jnp.asarray(na / wavelength, real)
        for axis in (fy, fx):
            if axis.size < 2:
                raise ValueError("pupil frequency axes need at least two entries")
            df = jnp.abs(axis[1])
            if not bool(df > 0):
                raise ValueError("pupil frequency axes must be unshifted FFT axes")
            expected = jnp.fft.fftfreq(axis.size, 1 / (axis.size * df), dtype=real)
            tolerance = 32 * jnp.finfo(real).eps * jnp.max(jnp.abs(axis))
            if not bool(jnp.all(jnp.abs(axis - expected) <= tolerance)):
                raise ValueError("pupil frequency axes must be unshifted uniform FFT axes")
            if bool(cutoff >= axis.size * df / 2):
                raise ValueError("pupil cutoff reaches transverse Nyquist; reduce pixel spacing")
        return (fy[:, None] ** 2 + fx[None, :] ** 2 <= cutoff**2).astype(complex_)


def _prepare(shape, spacing, wavelength, n_medium, source, pupil, dtype):
    """Validate concrete inputs before entering fixed-shape JIT kernels."""
    real, complex_ = _precision(dtype)
    _positive(wavelength, "wavelength")
    _positive(n_medium, "n_medium")
    if jnp.iscomplexobj(source):
        raise ValueError("source must contain real intensity weights")
    source = jnp.asarray(source, real)
    pupil = jnp.asarray(pupil, complex_)
    if source.shape != shape[1:] or pupil.shape != shape[1:]:
        raise ValueError("source and pupil must have shape (ny, nx)")
    if not bool(jnp.all(jnp.isfinite(source) & (source >= 0))):
        raise ValueError("source must contain finite nonnegative intensity weights")
    if not bool(jnp.all(jnp.isfinite(pupil))):
        raise ValueError("pupil must be finite")
    background = jnp.sum(source * jnp.abs(pupil) ** 2)
    if not bool(jnp.isfinite(background) & (background > 0)):
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
