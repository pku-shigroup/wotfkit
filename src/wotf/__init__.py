"""3D WOTFs for the real and imaginary parts of a scattering potential.

Arrays use (z, y, x); spatial arrays are centered, spectra are in FFT order.
Wavevectors are in rad/length. Numeric types follow JAX configuration and
input promotion; this package does not change JAX's x64 setting.
See the README and docs/theory_audit.md for conventions and model limits.
"""

from ._common import circular_pupil, k_axes
from .forward import compute_wotf, predict_contrast
from .wotf_ewald import wotf_ewald
from .wotf_fft import wotf_fft

__all__ = [
    "k_axes",
    "circular_pupil",
    "compute_wotf",
    "wotf_fft",
    "wotf_ewald",
    "predict_contrast",
]
