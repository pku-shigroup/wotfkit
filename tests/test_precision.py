"""Precision selection and scoped JAX x64 settings."""

import jax

from examples.validate_born import make_case
from wotf import circular_pupil, compute_wotf, k_axes, precision, predict_contrast


def test_precision_switch():
    """Default double, block switching, explicit override and x64 restoration."""
    fr, fi, spacing, optics = make_case()
    with jax.enable_x64(False):
        default = compute_wotf(fr.shape, spacing, **optics)
        assert all(str(h.dtype) == "complex128" for h in default)
        assert not jax.config.x64_enabled
        for dtype, complex_dtype in (("float32", "complex64"), ("float64", "complex128")):
            with precision(dtype):
                assert jax.config.x64_enabled == (dtype == "float64")
                axes = k_axes(fr.shape, spacing)
                assert all(str(axis.dtype) == dtype for axis in axes)
                pupil = circular_pupil(*axes[1:], wavelength=0.53, na=0.6)
                assert str(pupil.dtype) == complex_dtype
                for method, axial in (("fft", "finite"), ("ewald", "finite"), ("ewald", "linear")):
                    h = compute_wotf(fr.shape, spacing, **optics, method=method, axial=axial)
                    assert all(str(array.dtype) == complex_dtype for array in h)
                    assert str(predict_contrast(fr, fi, *h).dtype) == dtype
                override = compute_wotf(fr.shape, spacing, **optics, dtype="float64")
                assert all(str(h.dtype) == "complex128" for h in override)
                assert jax.config.x64_enabled == (dtype == "float64")
            assert not jax.config.x64_enabled
        assert str(k_axes(fr.shape, spacing)[0].dtype) == "float64"
        assert not jax.config.x64_enabled
