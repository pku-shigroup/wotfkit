"""3D WOTFs for the real and imaginary parts of a scattering potential.

Arrays use (z, y, x); spatial arrays are centered, spectra are in FFT order.
See the README and docs/theory_audit.md for conventions and model limits.
"""

from ._common import circular_pupil, frequency_axes
from .transfer import predict_contrast, wotf_ewald, wotf_fft

__all__ = [
    "frequency_axes",
    "circular_pupil",
    "wotf_fft",
    "wotf_ewald",
    "predict_contrast",
]
