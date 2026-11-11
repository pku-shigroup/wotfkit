"""Independent double-precision Born fields versus WOTF predictions."""

import jax
import jax.numpy as jnp
import numpy as np

from examples.validate_born import born_stack, make_case
from wotf import compute_wotf, predict_contrast


def test_against_scalar_born():
    """Five occupied Gaussian layers, nine focus planes inside padded lags."""
    # Only the test controls reference precision; neither model does so.
    with jax.enable_x64(True):
        fr, fi, spacing, optics = make_case()
        zero = jnp.zeros_like(fr)
        focus = jnp.arange(-4, 5) * spacing[0]
        cases = {"real": (np.asarray(fr), np.asarray(zero)),
                 "imag": (np.asarray(zero), np.asarray(fi)),
                 "mixed": (np.asarray(fr), np.asarray(fi))}
        references = {
            name: np.asarray(born_stack(real, imag, spacing, **optics, z_focus=focus)["linear_contrast"])
            for name, (real, imag) in cases.items()
        }
        optics = optics | {"source": np.asarray(optics["source"]), "pupil": np.asarray(optics["pupil"])}

    for enabled, tolerance in ((False, 1e-4), (True, 1e-10)):
        with jax.enable_x64(enabled):
            for channel, (real, imag) in cases.items():
                for method in ("fft", "ewald"):
                    h = compute_wotf(fr.shape, spacing, **optics, method=method, axial="finite")
                    prediction = np.asarray(predict_contrast(real, imag, *h))[4:13]
                    error = np.linalg.norm(prediction - references[channel]) / np.linalg.norm(references[channel])
                    assert error < tolerance, (channel, enabled, method, error)
                    print(f"Born {channel}, x64={enabled}, {method}: relative L2={error:.6e}")
