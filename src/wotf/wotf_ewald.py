"""WOTFs by translated Ewald caps with finite or linear axial sampling."""

import jax
import jax.numpy as jnp

from ._common import _precision, _prepare, k_axes


def _conjugate_reverse(spectrum):
    """conj(C(-k)), with exact modular reflection for odd and even grids."""
    for axis, n in enumerate(spectrum.shape):
        spectrum = jnp.take(spectrum, (-jnp.arange(n)) % n, axis=axis)
    return jnp.conj(spectrum)


def _axial_kernel(delta_kz, kz, dz):
    """DFT of exp(i delta_kz z_lag), including dz; no axial FFT used."""
    nz = kz.size
    u = (delta_kz[:, None] - kz[None, :]) * dz / (2 * jnp.pi)
    u = u - jnp.floor(u + 0.5)  # exact period-one sum; avoid removable poles
    # Centered even grids include -nz/2 but not +nz/2: retain its phase.
    return (
        nz * dz * jnp.sinc(nz * u) / jnp.sinc(u)
        * jnp.exp(-1j * jnp.pi * u * (1 - nz % 2))
    )


@jax.jit(static_argnames=("shape", "axial"))
def _ewald_kernel(
    source_weights, source_pupil, source_kz, source_y, source_x,
    detector_pupil, detector_kz, detector_y, detector_x,
    kz, dz, background, *, shape, axial,
):
    nz, ny, nx = shape

    def accumulate(i, cross):
        qy = (detector_y - source_y[i]) % ny
        qx = (detector_x - source_x[i]) % nx
        delta_kz = detector_kz - source_kz[i]
        # source_pupil is the detection pupil at the incident mode: it
        # weights the detected reference field, not an illumination pupil.
        coefficient = (
            source_weights[i] * jnp.conj(source_pupil[i]) * detector_pupil
            / (2 * detector_kz * background)
        )
        if axial == "finite":
            values = coefficient[:, None] * _axial_kernel(delta_kz, kz, dz)
            return cross.at[:, qy, qx].add(values.T)
        dkz = 2 * jnp.pi / (nz * dz)
        position = delta_kz / dkz
        lo = jnp.floor(position).astype(jnp.int32)
        fraction = position - lo
        weight = coefficient * (nz * dz)  # 2*pi*delta(kz-delta_kz): 2*pi/dkz
        cross = cross.at[lo % nz, qy, qx].add(weight * (1 - fraction))
        return cross.at[(lo + 1) % nz, qy, qx].add(weight * fraction)

    cross = jax.lax.fori_loop(
        0, source_weights.size, accumulate,
        jnp.zeros(shape, dtype=detector_pupil.dtype),
    )
    reverse = _conjugate_reverse(cross)
    H_real = -1j * (cross - reverse)
    H_imag = cross + reverse
    return H_real, H_imag


def wotf_ewald(
    shape, spacing, *, wavelength, n_medium, source, pupil,
    axial="finite", dtype=None,
):
    """Return (H_real, H_imag) by translating and accumulating Ewald caps.

    Parameters, units and ordering match :func:`wotf_fft`. Source is a 2D
    array of incoherent intensity weights; pupil is the 2D complex detection
    amplitude. Source modes where pupil == 0 are discarded with a warning.

    ``axial='finite'`` uses the exact finite axial DFT kernel (the same
    sampled impulse response as FFT correlation, up to floating-point error).
    ``axial='linear'`` deposits each continuous cap into two adjacent bins
    with total axial-kernel weight 2*pi/dkz. The latter is an approximation
    whose error must be checked by increasing axial extent.
    No illumination direction is rounded: source modes are grid cells.
    """
    real, _ = _precision(dtype)
    if axial not in ("finite", "linear"):
        raise ValueError('axial must be "finite" or "linear"')
    with jax.enable_x64(real == jnp.float64):
        source, pupil, kz_modes, background, si, vi = _prepare(
            shape, spacing, wavelength, n_medium, source, pupil, dtype
        )
        kz, _, _ = k_axes(shape, spacing, dtype=dtype)
        return _ewald_kernel(
            source[si], pupil[si], kz_modes[si], *si,
            pupil[vi], kz_modes[vi], *vi,
            kz, jnp.asarray(spacing[0], real), background,
            shape=tuple(shape), axial=axial,
        )
