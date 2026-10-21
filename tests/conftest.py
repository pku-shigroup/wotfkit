import jax
import pytest


@pytest.fixture(autouse=True)
def local_x64_context():
    # Tests may opt out in a nested context; never alter process-wide defaults.
    with jax.enable_x64(True):
        yield
