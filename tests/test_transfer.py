import jax
import jax.numpy as jnp
import numpy as np
import pytest

from examples.validate_born import make_case
from wotf import circular_pupil, frequency_axes, predict_contrast, wotf_ewald, wotf_fft
from wotf.transfer import _axial_kernel


def direct_channels(shape, spacing, optics):
    """Small explicit pair sum, with no lateral FFT correlation."""
    nz, ny, nx = shape
    dz, dy, dx = spacing
    z = np.fft.ifftshift((np.arange(nz) - nz // 2) * dz)
    fy, fx = np.fft.fftfreq(ny, dy), np.fft.fftfreq(nx, dx)
    w = np.sqrt(np.maximum((optics["n_medium"] / optics["wavelength"]) ** 2
                           - fy[:, None] ** 2 - fx[None, :] ** 2, 0))
    source, pupil = np.asarray(optics["source"]), np.asarray(optics["pupil"])
    background = np.sum(source * abs(pupil) ** 2)
    cross = np.zeros(shape, complex)
    for sy, sx in zip(*np.nonzero(source)):
        for vy, vx in zip(*np.nonzero(pupil)):
            coefficient = source[sy, sx] * pupil[sy, sx].conjugate() * pupil[vy, vx] / (4 * np.pi * w[vy, vx] * background)
            cross[:, (vy - sy) % ny, (vx - sx) % nx] += coefficient * np.exp(2j * np.pi * (w[vy, vx] - w[sy, sx]) * z)
    reverse = np.conj(cross[:, (-np.arange(ny)) % ny][:, :, (-np.arange(nx)) % nx])
    return dz * np.fft.fft(-1j * (cross - reverse), axis=0), dz * np.fft.fft(cross + reverse, axis=0)


@pytest.mark.parametrize("shape", [(17, 15, 17), (16, 14, 18)])
@pytest.mark.parametrize("dtype,tolerance", [("float64", 2e-12), ("float32", 2e-5)])
def test_both_methods_match_direct_correlation(shape, dtype, tolerance):
    _, _, spacing, optics = make_case(shape)
    expected = direct_channels(shape, spacing, optics)
    for method in (wotf_fft, wotf_ewald):
        h = method(shape, spacing, **optics, dtype=dtype)
        assert all(isinstance(array, jax.Array) for array in h)
        for actual, target in zip(h, expected):
            assert np.linalg.norm(np.asarray(actual) - target) / np.linalg.norm(target) < tolerance
            reflected = np.conj(np.asarray(actual)[np.ix_(*[(-np.arange(n)) % n for n in shape])])
            # Combining A/B before the axial FFT preserves Hermitian symmetry
            # up to transform roundoff, rather than imposing it bit for bit.
            roundoff = 8 * np.finfo(dtype).eps * np.max(np.abs(actual))
            np.testing.assert_allclose(actual, reflected, rtol=0, atol=roundoff)


def test_original_pgt_channel_signs_are_equivalent():
    fr, _, spacing, optics = make_case()
    nz, ny, nx = fr.shape
    _, fy, fx = frequency_axes(fr.shape, spacing)
    w = np.sqrt(np.maximum((optics["n_medium"] / optics["wavelength"]) ** 2
                           - np.asarray(fy)[:, None] ** 2 - np.asarray(fx)[None, :] ** 2, 0))
    z = np.fft.ifftshift((np.arange(nz) - nz // 2) * spacing[0])
    source, pupil = np.asarray(optics["source"]), np.asarray(optics["pupil"])
    phase = np.exp(2j * np.pi * z[:, None, None] * w)
    a = source * pupil * phase
    b = pupil * phase / (4 * np.pi * np.where(w > 0, w, 1))
    reversed_product = np.fft.fft2(a) * np.fft.fft2(b).conj()
    background = np.sum(source * abs(pupil) ** 2)
    old_real = 1j * spacing[0] * np.fft.fft(2j * np.fft.ifft2(reversed_product.imag), axis=0) / background
    old_imag = spacing[0] * np.fft.fft(2 * np.fft.ifft2(reversed_product.real), axis=0) / background
    for actual, expected in zip(wotf_fft(fr.shape, spacing, **optics), (old_real, old_imag)):
        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-14)


@pytest.mark.parametrize("nz", [1, 8, 9, 16, 17])
@pytest.mark.parametrize("dtype,tolerance", [("float64", 2e-13), ("float32", 2e-5)])
def test_finite_kernel_including_even_grid_phase(nz, dtype, tolerance):
    real = getattr(jnp, dtype)
    dz = jnp.asarray(0.35, real)
    fz = jnp.fft.fftfreq(nz, dz, dtype=real)
    eta = jnp.asarray([0, 0.137, -0.291, 1 / (nz * 0.35)], real)
    z = jnp.fft.ifftshift((jnp.arange(nz, dtype=real) - nz // 2) * dz)
    direct = dz * jnp.sum(jnp.exp(2j * jnp.pi * (eta[:, None, None] - fz[None, :, None]) * z), axis=-1)
    np.testing.assert_allclose(_axial_kernel(eta, fz, dz), direct, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("method", [wotf_fft, wotf_ewald])
@pytest.mark.parametrize("dtype,tolerance", [("float64", 1e-13), ("float32", 1e-8)])
def test_analytic_uniform_thin_layer_sign_and_absolute_scale(method, dtype, tolerance):
    shape, spacing = (9, 9, 11), (0.4, 0.22, 0.2)
    _, fy, fx = frequency_axes(shape, spacing)
    pupil = circular_pupil(fy, fx, wavelength=0.53, na=0.4)
    source = jnp.zeros(shape[1:]).at[0, 0].set(1)
    real = jnp.zeros(shape).at[shape[0] // 2].set(0.02)
    imag = jnp.zeros(shape).at[shape[0] // 2].set(-0.01)
    h = method(shape, spacing, wavelength=0.53, n_medium=1.33, source=source, pupil=pupil, dtype=dtype)
    phase_only = predict_contrast(real, jnp.zeros_like(imag), *h, dtype=dtype)
    absorption = predict_contrast(real, imag, *h, dtype=dtype)
    expected = spacing[0] * -0.01 / (2 * np.pi * (1.33 / 0.53))
    np.testing.assert_allclose(phase_only, 0, atol=tolerance)
    np.testing.assert_allclose(absorption, expected, rtol=0, atol=tolerance)
    assert bool(jnp.all(absorption < 0))


@pytest.mark.parametrize("method", [wotf_fft, wotf_ewald])
def test_physical_unit_scaling_and_source_weight_scaling(method):
    fr, fi, spacing, optics = make_case()
    h = method(fr.shape, spacing, **optics)
    scaled_source = method(fr.shape, spacing, **(optics | {"source": 7.5 * optics["source"]}))
    length_factor = 3.7
    rescaled = method(fr.shape, tuple(d * length_factor for d in spacing),
                      **(optics | {"wavelength": optics["wavelength"] * length_factor}))
    for first, same, longer in zip(h, scaled_source, rescaled):
        np.testing.assert_allclose(first, same, rtol=2e-12, atol=1e-14)
        np.testing.assert_allclose(longer, length_factor**2 * first, rtol=2e-12, atol=1e-13)
    np.testing.assert_allclose(
        predict_contrast(fr, fi, *h),
        predict_contrast(fr / length_factor**2, fi / length_factor**2, *rescaled),
        rtol=2e-11, atol=1e-16,
    )


def test_linear_gridding_retains_even_nyquist_and_real_prediction():
    fr, fi, _, optics = make_case((4, 15, 17))
    spacing = (1.0, 0.25, 0.22)
    h = wotf_ewald(fr.shape, spacing, **optics, axial="linear")
    assert float(jnp.max(jnp.abs(h[1][2]))) > 0  # cap interpolates into Nyquist
    spectrum = h[0] * jnp.fft.fftn(jnp.fft.ifftshift(fr)) + h[1] * jnp.fft.fftn(jnp.fft.ifftshift(fi))
    field = jnp.fft.ifftn(spectrum)
    assert float(jnp.max(jnp.abs(field.imag))) < 1e-15
    assert predict_contrast(fr, fi, *h).shape == fr.shape


def test_prediction_rejects_nonhermitian_transfer():
    fr, fi, spacing, optics = make_case()
    hr, hi = wotf_fft(fr.shape, spacing, **optics)
    with pytest.raises(ValueError, match="Hermitian"):
        predict_contrast(fr, fi, hr.at[1, 2, 3].add(1j), hi)
