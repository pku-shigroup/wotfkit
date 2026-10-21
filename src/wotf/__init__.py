"""3D WOTFs for the real and imaginary parts of a scattering potential.

Arrays use (z, y, x); spatial arrays are centered, spectra are in FFT order.
Wavevectors are in rad/length. Use precision("float32") for a single-precision
block; numerical functions default to double precision outside a block.
See the README and docs/theory_audit.md for conventions and model limits.
"""

from ._common import circular_pupil, precision, wavevector_axes
from .wotf import predict_contrast, wotf_ewald, wotf_fft

__all__ = [
    "precision",
    "wavevector_axes",
    "circular_pupil",
    "wotf_fft",
    "wotf_ewald",
    "predict_contrast",
]
