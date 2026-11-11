"""Accumulated Ewald pairs and their scattering-potential channels."""

import jax
import numpy as np
import pytest

from examples.validate_born import make_case
from wotf import compute_wotf, precision, predict_contrast, wotf_ewald, wotf_fft


def test_ewald_pair():
    for shape in ((4, 15, 17), (17, 14, 18)):
        fr, fi, spacing, optics = make_case(shape)  # asymmetric source, complex pupil
        if shape[0] == 4:
            spacing = (1.75, *spacing[1:])  # valid caps interpolate into the Nyquist bin
        for dtype, complex_dtype in (("float64", "complex128"), ("float32", "complex64")):
            with precision(dtype):
                tolerance = 1e-12 if dtype == "float64" else 1e-4
                for axial in ("finite", "linear"):
                    pair = wotf_ewald(shape, spacing, **optics, axial=axial, output="pair")
                    via_entry = compute_wotf(shape, spacing, **optics, method="ewald", axial=axial, output="pair")
                    default = wotf_ewald(shape, spacing, **optics, axial=axial)
                    explicit = compute_wotf(shape, spacing, **optics, method="ewald", axial=axial, output="tfs")
                    for direct, routed in zip(pair, via_entry):
                        assert isinstance(direct, jax.Array) and direct.shape == shape
                        assert str(direct.dtype) == complex_dtype
                        np.testing.assert_array_equal(direct, routed)

                    cross, reverse = pair
                    if shape[0] == 4:
                        assert np.max(np.abs(np.asarray(cross)[shape[0] // 2])) > 0
                    negative = np.ix_(*[(-np.arange(n)) % n for n in shape])
                    np.testing.assert_array_equal(reverse, np.conj(np.asarray(cross)[negative]))
                    rebuilt = -1j * (cross - reverse), cross + reverse
                    for actual, requested, reconstructed in zip(default, explicit, rebuilt):
                        np.testing.assert_array_equal(actual, requested)
                        np.testing.assert_allclose(actual, reconstructed, rtol=tolerance, atol=tolerance)
                    np.testing.assert_allclose(
                        predict_contrast(fr, fi, *rebuilt), predict_contrast(fr, fi, *default),
                        rtol=tolerance, atol=tolerance,
                    )

                    if axial == "finite":
                        # Independent FFT correlation determines C through
                        # C = (H_imag + 1j*H_real)/2, including scale and sign.
                        hr, hi = wotf_fft(shape, spacing, **optics)
                        expected = np.asarray((hi + 1j * hr) / 2)
                        error = np.linalg.norm(np.asarray(cross) - expected) / np.linalg.norm(expected)
                        assert error < tolerance, (shape, dtype, axial, error)

    with pytest.raises(ValueError, match='method="ewald"'):
        compute_wotf(shape, spacing, **optics, method="fft", output="pair")
    with pytest.raises(ValueError, match="output"):
        compute_wotf(shape, spacing, **optics, method="ewald", output="invalid")
