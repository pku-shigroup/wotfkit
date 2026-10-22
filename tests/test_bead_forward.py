"""Qualitative forward prediction of a weak spherical bead."""

import jax.numpy as jnp
import numpy as np

from examples.forward_bead import simulate_bead
from wotf import precision, predict_contrast


def test_bead_forward():
    """Qualitative WOTF demonstration: contrast, symmetry and absorption sign.

    This checks the forward workflow, not agreement with exact sphere optics.
    """
    with precision("float64"):
        for method, axial in (("fft", "finite"), ("ewald", "finite"), ("ewald", "linear")):
            result = simulate_bead((33, 25, 25), method=method, axial=axial)
            fr, fi = result["F_real"], result["F_imag"]
            h = result["H_real"], result["H_imag"]
            phase = np.asarray(predict_contrast(fr, jnp.zeros_like(fi), *h))
            absorption = np.asarray(predict_contrast(jnp.zeros_like(fr), fi, *h))
            contrast = np.asarray(result["contrast"])
            assert contrast.shape == fr.shape
            assert np.isfinite(contrast).all() and np.max(np.abs(contrast)) > 1e-6
            assert np.min(np.asarray(result["intensity"])) > 0
            np.testing.assert_allclose(contrast, phase + absorption, rtol=1e-11, atol=1e-13)
            middle = fr.shape[0] // 2
            # Centered sphere + upper half-disk: phase contrast is odd in y
            # at focus; absorption is even in y and reduces mean intensity.
            np.testing.assert_allclose(phase[middle], -phase[middle, ::-1], atol=1e-12)
            np.testing.assert_allclose(absorption[middle], absorption[middle, ::-1], atol=1e-12)
            assert absorption.mean() < 0
