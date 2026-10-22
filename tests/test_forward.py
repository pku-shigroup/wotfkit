import jax
import numpy as np
import pytest

from examples.validate_born import make_case
from wotf import compute_wotf, precision, predict_contrast, wotf_ewald, wotf_fft


@pytest.mark.parametrize("method,axial", [
    ("fft", "finite"), ("fft", "linear"),
    ("ewald", "finite"), ("ewald", "linear"),
])
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_unified_entry_matches_direct_call_and_preserves_precision(method, axial, dtype):
    fr, fi, spacing, optics = make_case()
    with jax.enable_x64(False):
        with precision(dtype):
            actual = compute_wotf(fr.shape, spacing, **optics, method=method, axial=axial)
            if method == "fft":
                expected = wotf_fft(fr.shape, spacing, **optics)
            else:
                expected = wotf_ewald(fr.shape, spacing, **optics, axial=axial)
            for first, second in zip(actual, expected):
                np.testing.assert_array_equal(first, second)
                assert str(first.dtype) == ("complex128" if dtype == "float64" else "complex64")
            predicted = predict_contrast(fr, fi, *actual)
            assert str(predicted.dtype) == dtype
        assert not jax.config.x64_enabled


def test_unified_entry_defaults_to_double_fft_and_allows_dtype_override():
    fr, _, spacing, optics = make_case()
    with jax.enable_x64(False):
        actual = compute_wotf(fr.shape, spacing, **optics)
        expected = wotf_fft(fr.shape, spacing, **optics)
        with precision("float32"):
            overridden = compute_wotf(fr.shape, spacing, **optics, dtype="float64")
            assert not jax.config.x64_enabled
        for first, second, third in zip(actual, expected, overridden):
            assert str(first.dtype) == "complex128"
            np.testing.assert_array_equal(first, second)
            np.testing.assert_array_equal(first, third)
        assert not jax.config.x64_enabled


def test_unified_entry_rejects_unknown_method_and_forwards_axial_selection():
    fr, _, spacing, optics = make_case()
    with pytest.raises(ValueError, match="method"):
        compute_wotf(fr.shape, spacing, **optics, method="other")
    with pytest.raises(ValueError, match="axial"):
        compute_wotf(fr.shape, spacing, **optics, method="ewald", axial="other")
