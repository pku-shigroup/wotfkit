"""FFT correlation and translated Ewald caps, under one Born convention."""

from functools import partial

import jax
import jax.numpy as jnp

from ._common import _grid_parameters, _precision, _prepare, frequency_axes


def _conjugate_reverse(spectrum):
    """conj(C(-f)), with exact modular reflection for odd and even grids."""
    for axis, n in enumerate(spectrum.shape):
        spectrum = jnp.take(spectrum, (-jnp.arange(n)) % n, axis=axis)
    return jnp.conj(spectrum)


def _channels(cross):
    reverse = _conjugate_reverse(cross)
    return -1j * (cross - reverse), cross + reverse


@jax.jit
def _fft_kernel(source, pupil, longitudinal, z_lag, dz, fm, background):
    # Removing the common carrier exp(2 pi i fm z) is exact and reduces
    # cancellation in single precision. It cancels in conj(a(s))*b(s+q).
    phase = jnp.exp(2j * jnp.pi * z_lag[:, None, None] * (longitudinal - fm))
    a = source * pupil * phase
    b = pupil * phase / (4 * jnp.pi * jnp.where(longitudinal > 0, longitudinal, 1))
    correlation = jnp.fft.ifft2(jnp.conj(jnp.fft.fft2(a)) * jnp.fft.fft2(b))
    cross = dz * jnp.fft.fft(correlation, axis=0) / background
    return _channels(cross)


def wotf_fft(
    shape, spacing, *, wavelength, n_medium, source, pupil, dtype="float64"
):
    """Return (H_real, H_imag) by FFT correlation, both in FFT order.

    Arrays have shape (nz, ny, nx). With F=k0**2*(n_medium**2-n**2),
    c=(I-I0)/I0 obeys c_hat=H_real*F_real_hat+H_imag*F_imag_hat.
    H has units length**2; F has units length**-2. ``source`` contains
    nonnegative, mutually incoherent intensity weights on the transverse
    FFT grid, not a continuous density. ``pupil`` may be complex.

    This is a finite, unwindowed axial impulse response. Multiplication by
    H implements circular convolution; pad z and crop when modeling a
    finite specimen. See the independent Born example for a guarded ROI.
    """
    real, _ = _precision(dtype)
    shape, spacing = _grid_parameters(shape, spacing)
    with jax.enable_x64(True):
        source, pupil, w, background, _, _ = _prepare(
            shape, spacing, wavelength, n_medium, source, pupil, dtype
        )
        dz = jnp.asarray(spacing[0], real)
        z_lag = jnp.fft.ifftshift((jnp.arange(shape[0], dtype=real) - shape[0] // 2) * dz)
        return _fft_kernel(
            source, pupil, w, z_lag, dz, jnp.asarray(n_medium / wavelength, real), background
        )


def _axial_kernel(eta, fz, dz):
    """DFT of exp(2 pi i eta z_lag), including dz; no axial FFT used."""
    nz = fz.size
    u = (eta[:, None] - fz[None, :]) * dz
    u = u - jnp.floor(u + 0.5)  # exact period-one sum; avoid removable poles
    # Centered even grids include -nz/2 but not +nz/2: retain its phase.
    return (
        nz * dz * jnp.sinc(nz * u) / jnp.sinc(u)
        * jnp.exp(-1j * jnp.pi * u * (1 - nz % 2))
    )


@partial(jax.jit, static_argnames=("shape", "axial"))
def _ewald_kernel(
    source_weights, source_pupil, source_w, source_y, source_x,
    detector_pupil, detector_w, detector_y, detector_x,
    fz, dz, background, *, shape, axial,
):
    nz, ny, nx = shape

    def accumulate(i, cross):
        qy = (detector_y - source_y[i]) % ny
        qx = (detector_x - source_x[i]) % nx
        eta = detector_w - source_w[i]
        coefficient = (
            source_weights[i] * jnp.conj(source_pupil[i]) * detector_pupil
            / (4 * jnp.pi * detector_w * background)
        )
        if axial == "finite":
            values = coefficient[:, None] * _axial_kernel(eta, fz, dz)
            return cross.at[:, qy, qx].add(values.T)
        position = eta * (nz * dz)
        lo = jnp.floor(position).astype(jnp.int32)
        fraction = position - lo
        weight = coefficient * (nz * dz)  # delta(fz-eta) -> weights / dfz
        cross = cross.at[lo % nz, qy, qx].add(weight * (1 - fraction))
        return cross.at[(lo + 1) % nz, qy, qx].add(weight * fraction)

    cross = jax.lax.fori_loop(
        0, source_weights.size, accumulate,
        jnp.zeros(shape, dtype=detector_pupil.dtype),
    )
    return _channels(cross)


def wotf_ewald(
    shape, spacing, *, wavelength, n_medium, source, pupil,
    axial="finite", dtype="float64",
):
    """Return (H_real, H_imag) by translating and accumulating Ewald caps.

    Parameters, units and ordering match :func:`wotf_fft`. ``axial='finite'``
    uses the exact finite axial DFT kernel (the same sampled impulse response
    as FFT correlation). ``axial='linear'`` deposits each continuous cap into
    two adjacent bins with total delta weight 1/dfz. The latter is an
    approximation whose error must be checked by increasing axial extent.
    No illumination direction is rounded: source modes are grid cells.
    """
    real, _ = _precision(dtype)
    shape, spacing = _grid_parameters(shape, spacing)
    if axial not in ("finite", "linear"):
        raise ValueError('axial must be "finite" or "linear"')
    with jax.enable_x64(True):
        source, pupil, w, background, si, vi = _prepare(
            shape, spacing, wavelength, n_medium, source, pupil, dtype
        )
        fz, _, _ = frequency_axes(shape, spacing, dtype=dtype)
        return _ewald_kernel(
            source[si], pupil[si], w[si], *si,
            pupil[vi], w[vi], *vi,
            fz, jnp.asarray(spacing[0], real), background,
            shape=shape, axial=axial,
        )


@jax.jit
def _predict_kernel(real_potential, imag_potential, h_real, h_imag):
    spectrum = (
        h_real * jnp.fft.fftn(jnp.fft.ifftshift(real_potential))
        + h_imag * jnp.fft.fftn(jnp.fft.ifftshift(imag_potential))
    )
    return jnp.fft.fftshift(jnp.fft.ifftn(spectrum)).real


def predict_contrast(F_real, F_imag, H_real, H_imag, *, dtype="float64"):
    """Predict centered real c=(I-I0)/I0 with circular 3D convolution.

    All four arrays must share shape (nz, ny, nx). Spatial potentials are
    centered; transfer functions are in unshifted FFT order. Voxel factors
    in the forward and inverse continuous Fourier approximations cancel:
    do not multiply the result by dx*dy*dz. H already includes the required
    axial integration dz. Absolute intensity is I0*(1+c).
    """
    real, complex_ = _precision(dtype)
    with jax.enable_x64(True):
        if jnp.iscomplexobj(F_real) or jnp.iscomplexobj(F_imag):
            raise ValueError("F_real and F_imag must each be real spatial arrays")
        fr, fi = jnp.asarray(F_real, real), jnp.asarray(F_imag, real)
        hr, hi = jnp.asarray(H_real, complex_), jnp.asarray(H_imag, complex_)
        if fr.ndim != 3 or not (fr.shape == fi.shape == hr.shape == hi.shape):
            raise ValueError("potentials and transfer functions must share a 3D shape")
        for array in (fr, fi, hr, hi):
            if not bool(jnp.all(jnp.isfinite(array))):
                raise ValueError("potentials and transfer functions must be finite")
        for name, h in (("H_real", hr), ("H_imag", hi)):
            tolerance = 64 * jnp.finfo(real).eps * jnp.max(jnp.abs(h))
            if not bool(jnp.all(jnp.abs(h - _conjugate_reverse(h)) <= tolerance)):
                raise ValueError(f"{name} must be Hermitian to predict a real intensity")
        return _predict_kernel(fr, fi, hr, hi)
