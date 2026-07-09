import numpy as np

ATOL_ZERO = 1e-12


def assert_zero(actual, atol=ATOL_ZERO):
    np.testing.assert_allclose(actual, 0.0, atol=atol)
