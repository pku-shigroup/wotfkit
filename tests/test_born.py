import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from examples.validate_born import born_stack, make_case
from wotf import circular_pupil, k_axes, predict_contrast, wotf_ewald, wotf_fft


@pytest.mark.parametrize("shape", [(17, 15, 17), (16, 14, 18)])
@pytest.mark.parametrize("channel", ["real", "imag", "mixed"])
@pytest.mark.parametrize("dtype,tolerance", [("float64", 1e-10), ("float32", 1e-4)])
def test_wotfs_against_independent_born(shape, channel, dtype, tolerance):
    fr, fi, spacing, optics = make_case(shape)
    if channel == "real":
        fi = jnp.zeros_like(fi)
    if channel == "imag":
        fr = jnp.zeros_like(fr)
    focus = jnp.arange(-4, 5, dtype=jnp.float64) * spacing[0]
    # Reference is double precision even when testing a single precision WOTF.
    reference = born_stack(fr, fi, spacing, **optics, z_focus=focus)
    linear = np.asarray(reference["linear_contrast"])
    start = shape[0] // 2 - 4
    for method in (wotf_fft, wotf_ewald):
        h = method(shape, spacing, **optics, dtype=dtype)
        prediction = np.asarray(predict_contrast(fr, fi, *h, dtype=dtype))[start:start + 9]
        error = np.linalg.norm(prediction - linear) / np.linalg.norm(linear)
        assert error < tolerance, (method.__name__, channel, dtype, error)


@pytest.mark.parametrize("source_kind", ["on_axis", "oblique"])
def test_single_plane_wave_and_binary_pupil(source_kind):
    fr, fi, spacing, optics = make_case(aberrated=False)
    source = jnp.zeros_like(optics["source"])
    index = (0, 0) if source_kind == "on_axis" else (1, 2)
    optics["source"] = source.at[index].set(1)
    focus = jnp.arange(-4, 5, dtype=jnp.float64) * spacing[0]
    reference = born_stack(fr, fi, spacing, **optics, z_focus=focus)
    for method in (wotf_fft, wotf_ewald):
        h = method(fr.shape, spacing, **optics)
        predicted = predict_contrast(fr, fi, *h)[4:13]
        np.testing.assert_allclose(predicted, reference["linear_contrast"], rtol=1e-10, atol=1e-16)


def test_linear_ewald_axial_gridding_converges_in_fixed_focus_roi():
    fr, fi, spacing, optics = make_case()
    focus = jnp.arange(-4, 5, dtype=jnp.float64) * spacing[0]
    reference = born_stack(fr, fi, spacing, **optics, z_focus=focus)["linear_contrast"]
    errors = []
    for nz in (17, 33, 65, 129):
        fr, fi, _, optics = make_case((nz, 15, 17))
        h = wotf_ewald(fr.shape, spacing, **optics, axial="linear")
        prediction = predict_contrast(fr, fi, *h)[nz // 2 - 4:nz // 2 + 5]
        errors.append(float(jnp.linalg.norm(prediction - reference) / jnp.linalg.norm(reference)))
    assert all(a > b for a, b in zip(errors, errors[1:])), errors
    assert errors[-1] < 0.003, errors


def test_born_quadratic_residual_and_intensity_normalization():
    fr, fi, spacing, optics = make_case()
    focus = jnp.arange(-4, 5, dtype=jnp.float64) * spacing[0]
    residuals = []
    for scale in (1.0, 0.5, 0.25):
        reference = born_stack(scale * fr, scale * fi, spacing, **optics, z_focus=focus)
        np.testing.assert_allclose(reference["linear_intensity"], reference["background"] * (1 + reference["linear_contrast"]), atol=5e-16)
        np.testing.assert_allclose(reference["born_intensity"], reference["background"] * (1 + reference["born_contrast"]), atol=5e-16)
        residual = reference["born_contrast"] - reference["linear_contrast"]
        assert float(jnp.min(residual)) >= -1e-18
        residuals.append(float(jnp.linalg.norm(residual)))
    np.testing.assert_allclose(np.array(residuals) / residuals[0], [1, 0.25, 0.0625], rtol=1e-10)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_independent_reference_precision(dtype):
    fr, fi, spacing, optics = make_case()
    data = born_stack(fr, fi, spacing, **optics, z_focus=[0], dtype=dtype)
    assert all(str(a.dtype) == dtype for a in data.values())


@pytest.mark.parametrize("shape", [(17, 15, 17), (16, 14, 18)])
@pytest.mark.parametrize("na_illumination,na_detection", [(0.3, 0.6), (0.8, 0.45)])
@pytest.mark.parametrize("dtype,tolerance", [("float64", 1e-12), ("float32", 1e-4)])
def test_unequal_apertures_match_transfer_and_independent_born(
    shape, na_illumination, na_detection, dtype, tolerance,
):
    fr, fi, spacing, _ = make_case(shape)
    _, ky, kx = k_axes(shape, spacing)
    cutoff = 2 * jnp.pi * na_detection / 0.53
    radial2 = ky[:, None] ** 2 + kx[None, :] ** 2
    source = jnp.abs(circular_pupil(ky, kx, wavelength=0.53, na=na_illumination)) ** 2
    source = source * (1 + 0.2 * kx[None, :] / (2 * jnp.pi * na_illumination / 0.53))
    pupil = circular_pupil(ky, kx, wavelength=0.53, na=na_detection)
    pupil = pupil * (1 - 0.25 * radial2 / cutoff**2) * jnp.exp(
        1j * (0.19 * kx[None, :] / cutoff + 0.31 * ky[:, None] ** 2 / cutoff**2)
    )
    optics = dict(wavelength=0.53, n_medium=1.33, source=source, pupil=pupil)
    # The reference retains all illumination. Outside-pupil modes have a
    # zero reference field, so their *linear* interference term is zero.
    focus = jnp.arange(-4, 5, dtype=jnp.float64) * spacing[0]
    reference = born_stack(fr, fi, spacing, **optics, z_focus=focus)["linear_contrast"]
    transfers = []
    for method in (wotf_fft, wotf_ewald):
        if na_illumination > na_detection:
            with pytest.warns(UserWarning, match="outside.*discarded"):
                h = method(shape, spacing, **optics, dtype=dtype)
        else:
            with warnings.catch_warnings(record=True) as quiet:
                h = method(shape, spacing, **optics, dtype=dtype)
            assert not quiet
        transfers.append(h)
        start = shape[0] // 2 - 4
        prediction = predict_contrast(fr, fi, *h, dtype=dtype)[start:start + 9]
        error = np.linalg.norm(np.asarray(prediction - reference)) / np.linalg.norm(reference)
        assert error < tolerance
    for fft, finite in zip(*transfers):
        fft, finite = np.asarray(fft), np.asarray(finite)
        norm = np.linalg.norm(fft)
        error = np.linalg.norm(fft - finite)
        if norm:
            assert error / norm < tolerance
        else:
            assert np.max(np.abs(finite)) < tolerance
