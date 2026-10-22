"""3D WOTFs for the real and imaginary parts of a scattering potential.

Arrays use (z, y, x); spatial arrays are centered, spectra are in FFT order.
Wavevectors are in rad/length. Use precision("float32") for a single-precision
block; numerical functions default to double precision outside a block.
See the README and docs/theory_audit.md for conventions and model limits.
"""

from ._common import circular_pupil, precision, k_axes
from .forward import compute_wotf, predict_contrast
from .wotf_ewald import wotf_ewald
from .wotf_fft import wotf_fft

__all__ = [
    "precision",
    "k_axes",
    "circular_pupil",
    "compute_wotf",
    "wotf_fft",
    "wotf_ewald",
    "predict_contrast",
]
