"""Outside-pupil illumination is discarded with a warning."""

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from examples.validate_born import make_case
from wotf import compute_wotf, precision


def test_source_outside_pupil():
    """Discard outside illumination before physics checks; warn once per call."""
    with precision("float64"):
        fr, _, spacing, optics = make_case()
        # One propagating and one evanescent mode outside the detection pupil.
        source = optics["source"].at[0, 5].set(0.7).at[7, 8].set(0.4)
        clipped = jnp.where(optics["pupil"] != 0, source, 0)
        for method, axial in (("fft", "finite"), ("ewald", "finite"), ("ewald", "linear")):
            with warnings.catch_warnings(record=True) as quiet:
                warnings.simplefilter("always")
                expected = compute_wotf(fr.shape, spacing, **(optics | {"source": clipped}), method=method, axial=axial)
            assert not quiet
            with pytest.warns(UserWarning, match="outside.*discarded") as emitted:
                actual = compute_wotf(fr.shape, spacing, **(optics | {"source": source}), method=method, axial=axial)
            assert len(emitted) == 1
            for first, second in zip(actual, expected):
                np.testing.assert_array_equal(first, second)
            outside_only = jnp.zeros_like(source).at[7, 8].set(1)
            with pytest.warns(UserWarning, match="outside.*discarded"):
                with pytest.raises(ValueError, match="background"):
                    compute_wotf(fr.shape, spacing, **(optics | {"source": outside_only}), method=method, axial=axial)
