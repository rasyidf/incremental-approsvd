from svd.incremental_svd import *
import numpy as np
import pytest


def test_svd():
    mat_a1 = np.random.randn(2, 3)
    mat_a2 = np.random.randn(2, 1)
    k = 2  # same as the original
    mat_u, mat_s, mat_vt = incrementalSVD(mat_a1, mat_a2, k)
    np.testing.assert_array_almost_equal(
        np.dot(np.dot(mat_u, mat_s), mat_vt), np.hstack((mat_a1, mat_a2)))
    assert True


def test_invalid_k():
    with pytest.raises(ValueError):
        incrementalSVD(np.random.randn(2, 3), np.random.randn(2, 1), 0)

    with pytest.raises(ValueError):
        incrementalSVD(np.random.randn(2, 3), np.random.randn(2, 1), 100)


def test_different_m():
    with pytest.raises(ValueError):
        incrementalSVD(np.random.randn(2, 3), np.random.randn(3, 1), 2)
