"""Select a WOTF method and predict linear intensity contrast."""

import jax
import jax.numpy as jnp

from ._common import _precision
from .wotf_ewald import wotf_ewald
from .wotf_fft import wotf_fft


def compute_wotf(
    shape, spacing, *, wavelength, n_medium, source, pupil,
    method="fft", axial="finite", dtype=None,
):
    """Return (H_real, H_imag) using 'fft' (default) or 'ewald'.

    All optical parameters follow :func:`wotf_fft`. ``axial`` selects
    'finite' or 'linear' for Ewald only; it is ignored for FFT.
    Precision follows the current precision() block unless dtype is given.
    """
    optics = dict(
        wavelength=wavelength, n_medium=n_medium,
        source=source, pupil=pupil, dtype=dtype,
    )
    if method == "fft":
        return wotf_fft(shape, spacing, **optics)
    if method == "ewald":
        return wotf_ewald(shape, spacing, **optics, axial=axial)
    raise ValueError('method must be "fft" or "ewald"')


@jax.jit
def _predict_kernel(real_potential, imag_potential, h_real, h_imag):
    spectrum = (
        h_real * jnp.fft.fftn(jnp.fft.ifftshift(real_potential))
        + h_imag * jnp.fft.fftn(jnp.fft.ifftshift(imag_potential))
    )
    return jnp.fft.fftshift(jnp.fft.ifftn(spectrum)).real


def predict_contrast(F_real, F_imag, H_real, H_imag, *, dtype=None):
    """Predict centered real c=(I-I0)/I0 with circular 3D convolution.

    All four arrays must share shape (nz, ny, nx). Spatial potentials are
    centered; transfer functions are in unshifted FFT order. Voxel factors
    in the forward and inverse continuous Fourier approximations cancel:
    do not multiply the result by dx*dy*dz. H already includes the required
    axial integration dz. Absolute intensity is I0*(1+c).
    """
    real, complex_ = _precision(dtype)
    with jax.enable_x64(real == jnp.float64):
        fr, fi = jnp.asarray(F_real, real), jnp.asarray(F_imag, real)
        hr, hi = jnp.asarray(H_real, complex_), jnp.asarray(H_imag, complex_)
        return _predict_kernel(fr, fi, hr, hi)
