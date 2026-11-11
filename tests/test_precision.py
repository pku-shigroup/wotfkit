"""JAX configuration and native promotion determine numerical types."""

import inspect
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np

import wotf
from examples.validate_born import make_case
from wotf import circular_pupil, compute_wotf, k_axes, predict_contrast, wotf_ewald, wotf_fft


def test_precision_switch():
    subprocess.run([sys.executable, "-c",
                    "import jax; before=jax.config.x64_enabled; import wotf; "
                    "assert jax.config.x64_enabled == before"], check=True)
    assert not hasattr(wotf, "precision")
    for function in (k_axes, circular_pupil, compute_wotf, wotf_fft, wotf_ewald, predict_contrast):
        assert "dtype" not in inspect.signature(function).parameters

    before = jax.config.x64_enabled
    for enabled, real_dtype, complex_dtype in ((False, "float32", "complex64"), (True, "float64", "complex128")):
        with jax.enable_x64(enabled):
            fr, fi, spacing, optics = make_case()
            axes = k_axes(fr.shape, spacing)
            assert all(str(axis.dtype) == real_dtype for axis in axes)
            pupil = circular_pupil(*axes[1:], wavelength=0.53, na=0.6)
            assert str(pupil.dtype) == complex_dtype
            for method, axial in (("fft", "finite"), ("ewald", "finite"), ("ewald", "linear")):
                h = compute_wotf(fr.shape, spacing, **optics, method=method, axial=axial)
                assert all(str(array.dtype) == complex_dtype for array in h)
                assert str(predict_contrast(fr, fi, *h).dtype) == real_dtype
                assert jax.config.x64_enabled == enabled

            # Real/float32 pupils must become complex. With x64 enabled,
            # float64 grids must promote the Ewald accumulator to complex128.
            mixed = optics | {"source": optics["source"].astype(jnp.float32),
                              "pupil": optics["pupil"].real.astype(jnp.float32)}
            shape = (16, *fr.shape[1:])
            for axial in ("finite", "linear"):
                cross, _ = wotf_ewald(shape, spacing, **mixed, axial=axial, output="pair")
                assert str(cross.dtype) == complex_dtype
                if axial == "finite":
                    assert np.max(np.abs(np.asarray(cross).imag)) > 0
            assert jax.config.x64_enabled == enabled
    assert jax.config.x64_enabled == before
