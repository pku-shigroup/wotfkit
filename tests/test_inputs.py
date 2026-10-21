import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from examples.validate_born import make_case
from wotf import circular_pupil, frequency_axes, predict_contrast, wotf_ewald, wotf_fft


def test_import_does_not_change_jax_x64_configuration():
    result = subprocess.run(
        [sys.executable, "-c", "import jax; before=jax.config.x64_enabled; import wotf; assert jax.config.x64_enabled == before"],
        check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("dtype,complex_name", [("float32", "complex64"), ("float64", "complex128")])
def test_explicit_precision_with_external_x64_disabled(dtype, complex_name):
    fr, fi, spacing, optics = make_case()
    # Pass host doubles; constructing JAX inputs outside the library with x64
    # disabled would already have lost precision, which no cast can recover.
    optics = optics | {"source": np.asarray(optics["source"]), "pupil": np.asarray(optics["pupil"])}
    with jax.enable_x64(False):
        axes = frequency_axes(fr.shape, spacing, dtype=dtype)
        assert all(str(a.dtype) == dtype for a in axes)
        assert not jax.config.x64_enabled
        pupil = circular_pupil(*axes[1:], wavelength=0.53, na=0.6, dtype=dtype)
        assert str(pupil.dtype) == complex_name
        for method in (wotf_fft, wotf_ewald):
            h = method(fr.shape, spacing, **optics, dtype=dtype)
            assert all(str(a.dtype) == complex_name for a in h)
            pred = predict_contrast(np.asarray(fr), np.asarray(fi), *h, dtype=dtype)
            assert str(pred.dtype) == dtype
            assert not jax.config.x64_enabled
        h = wotf_fft(fr.shape, spacing, **optics)  # default must really be double
        assert str(h[0].dtype) == "complex128"
        assert not jax.config.x64_enabled


@pytest.mark.parametrize("shape,spacing", [((4, 7, 9), (0.7, 0.21, 0.17)), ((5, 8, 10), (0.4, 0.2, 0.3))])
def test_frequency_axes_are_unshifted_and_ordered_zyx(shape, spacing):
    for actual, n, d in zip(frequency_axes(shape, spacing), shape, spacing):
        np.testing.assert_allclose(actual, np.fft.fftfreq(n, d), rtol=1e-15, atol=0)


@pytest.mark.parametrize("bad", ["float16", "complex128", None])
def test_reject_unsupported_precision(bad):
    with pytest.raises(ValueError, match="dtype"):
        frequency_axes((3, 5, 7), (1, 1, 1), dtype=bad)


@pytest.mark.parametrize("method", [wotf_fft, wotf_ewald])
@pytest.mark.parametrize("bad,match", [
    ("zero", "background"), ("darkfield", "background"),
    ("evanescent", "propagating"), ("transverse_alias", "transverse"),
    ("axial_alias", "axial sampling"),
])
def test_reject_unsupported_physics(method, bad, match):
    fr, _, spacing, optics = make_case()
    if bad == "zero":
        optics["source"] = jnp.zeros_like(optics["source"])
    elif bad == "darkfield":
        optics["pupil"] = jnp.zeros_like(optics["pupil"])
    elif bad == "evanescent":
        optics["pupil"] = optics["pupil"].at[7, 8].set(1)
    elif bad == "transverse_alias":
        optics["source"] = jnp.zeros_like(optics["source"]).at[0, -3].set(1)
        optics["pupil"] = jnp.zeros_like(optics["pupil"]).at[0, -3].set(1).at[0, 6].set(1)
    elif bad == "axial_alias":
        spacing = (5.0, spacing[1], spacing[2])
    with pytest.raises(ValueError, match=match):
        method(fr.shape, spacing, **optics)


def test_reject_unknown_axial_mode():
    fr, _, spacing, optics = make_case()
    with pytest.raises(ValueError, match="axial"):
        wotf_ewald(fr.shape, spacing, **optics, axial="nearest")
