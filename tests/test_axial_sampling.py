"""Reject axial sampling that cannot represent the cap frequencies."""

import jax.numpy as jnp
import numpy as np
import pytest

from examples.validate_born import make_case
from wotf import compute_wotf, precision


def test_axial_undersampling():
    """Accept dz below pi/max|delta_kz| and reject it above that limit."""
    with precision("float64"):
        fr, _, spacing, optics = make_case()
        optics["source"] = jnp.zeros_like(optics["source"]).at[0, 0].set(1)
        ny, nx = fr.shape[1:]
        ky, kx = 2 * np.pi * np.fft.fftfreq(ny, spacing[1]), 2 * np.pi * np.fft.fftfreq(nx, spacing[2])
        km = 2 * np.pi * optics["n_medium"] / optics["wavelength"]
        kz = np.sqrt(np.maximum(km**2 - ky[:, None] ** 2 - kx[None, :] ** 2, 0))
        largest = km - np.min(kz[np.asarray(optics["pupil"]) != 0])
        critical_dz = np.pi / largest
        for method, axial in (("fft", "finite"), ("ewald", "finite"), ("ewald", "linear")):
            h = compute_wotf(fr.shape, (0.9 * critical_dz, *spacing[1:]), **optics, method=method, axial=axial)
            assert all(np.isfinite(np.asarray(array)).all() for array in h)
            with pytest.raises(ValueError, match="axial sampling.*reduce dz"):
                compute_wotf(fr.shape, (1.1 * critical_dz, *spacing[1:]), **optics, method=method, axial=axial)
        print(f"Axial sampling: dz must be < {critical_dz:.6g} um for this pupil/source.")
