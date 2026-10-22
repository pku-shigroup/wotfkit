"""Finite-sampling WOTFs by FFT correlation, following analytical A/B terms."""

import jax
import jax.numpy as jnp

from ._common import _precision, _prepare


@jax.jit
def _fft_kernel(source, pupil, kz, z_lag, dz, km, background):
    # Removing the common carrier exp(i km z) is exact and reduces
    # cancellation in single precision. It cancels in both correlations.
    phase = jnp.exp(1j * z_lag[:, None, None] * (kz - km))
    # The detection pupil filters the unscattered reference at the source mode.
    illumination_arm = source * pupil * phase
    # analytical.tex: outgoing Born factor 1/(2 i k_z).
    scattering_arm = pupil * phase / (2j * jnp.where(kz > 0, kz, 1))
    fft_illumination = jnp.fft.fft2(illumination_arm)
    fft_scattering = jnp.fft.fft2(scattering_arm)
    correlation_spectrum = jnp.conj(fft_illumination) * fft_scattering

    # Match analytical.tex: A = corr(illumination_arm, scattering_arm),
    # B = corr(scattering_arm, illumination_arm). Both include the Born 1/i.
    A = jnp.fft.ifft2(correlation_spectrum)
    B = jnp.fft.ifft2(jnp.conj(correlation_spectrum))
    h_real = A + B
    h_imag = 1j * (A - B)
    H_real = dz * jnp.fft.fft(h_real, axis=0) / background
    H_imag = dz * jnp.fft.fft(h_imag, axis=0) / background
    return H_real, H_imag


def wotf_fft(
    shape, spacing, *, wavelength, n_medium, source, pupil, dtype=None
):
    """Return (H_real, H_imag) by FFT correlation, both in FFT order.

    Arrays have shape (nz, ny, nx). With F=k0**2*(n_medium**2-n**2),
    c=(I-I0)/I0 obeys c_hat=H_real*F_real_hat+H_imag*F_imag_hat.
    H has units length**2; F has units length**-2. ``source`` contains
    nonnegative, mutually incoherent intensity weights on the transverse
    FFT grid, not a continuous density. ``pupil`` is the complex detection
    amplitude, including its aperture and aberrations. Source modes where
    pupil == 0 are discarded with a warning: they have no reference field.

    This is a finite, unwindowed axial impulse response. Multiplication by
    H implements circular convolution; pad z and crop when modeling a
    finite specimen. See the independent Born example for a guarded ROI.
    """
    real, _ = _precision(dtype)
    with jax.enable_x64(real == jnp.float64):
        source, pupil, kz, background, _, _ = _prepare(
            shape, spacing, wavelength, n_medium, source, pupil, dtype
        )
        dz = jnp.asarray(spacing[0], real)
        z_lag = jnp.fft.ifftshift((jnp.arange(shape[0], dtype=real) - shape[0] // 2) * dz)
        return _fft_kernel(
            source, pupil, kz, z_lag, dz,
            jnp.asarray(2 * jnp.pi * n_medium / wavelength, real), background,
        )
