"""Independent scalar Born fields versus FFT and finite Ewald predictions."""

import jax.numpy as jnp
import numpy as np

from examples.validate_born import born_stack, make_case
from wotf import compute_wotf, precision, predict_contrast


def test_against_scalar_born():
    """Separate weak Gaussian F_real/F_imag; independent slice-integrated fields.

    Periodic xy, five occupied specimen layers, nine central focus planes.
    All focus-minus-specimen lags fit inside the padded 17-plane response.
    Linear Ewald is approximate and is not subjected to the exact threshold.
    """
    with precision("float64"):
        fr, fi, spacing, optics = make_case()
        zero = jnp.zeros_like(fr)
        focus = jnp.arange(-4, 5) * spacing[0]
        cases = {"real": (fr, zero), "imag": (zero, fi), "mixed": (fr, fi)}
        for channel, (real, imag) in cases.items():
            # born_stack builds grids and propagates fields independently;
            # it never calls a WOTF or its internal helpers.
            reference = np.asarray(born_stack(real, imag, spacing, **optics, z_focus=focus)["linear_contrast"])
            for dtype, tolerance in (("float64", 1e-10), ("float32", 1e-4)):
                with precision(dtype):
                    for method in ("fft", "ewald"):
                        h = compute_wotf(fr.shape, spacing, **optics, method=method, axial="finite")
                        prediction = np.asarray(predict_contrast(real, imag, *h))[4:13]
                        error = np.linalg.norm(prediction - reference) / np.linalg.norm(reference)
                        assert error < tolerance, (channel, dtype, method, error)
                        print(f"Born {channel}, {dtype}, {method}: relative L2={error:.6e}")
