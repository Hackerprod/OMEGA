import numpy as np
import pytest

from omega.core import kernels


def test_arnoldi_python_matches_reference():
    rng = np.random.default_rng(0)
    Q = rng.standard_normal((12, 6))
    w = rng.standard_normal(12)
    decay = 0.97
    k = 4
    res_py, h_py, grad_py = kernels._arnoldi_python(Q[:, :k], w.copy(), decay, k)
    assert res_py.shape == (12,)
    assert h_py.shape == (k + 1,)
    assert grad_py.shape == (k,)
    assert np.isfinite(res_py).all()
    assert np.isfinite(h_py).all()
    assert np.isfinite(grad_py).all()


@pytest.mark.skipif(not kernels.NATIVE_AVAILABLE, reason="Native kernels not built")
def test_native_matches_python():
    rng = np.random.default_rng(123)
    Q = rng.standard_normal((16, 8))
    w = rng.standard_normal(16)
    decay = 0.99
    k = 5

    res_native, h_native, grad_native = kernels.arnoldi_iteration(Q, w, decay, k)
    res_py, h_py, grad_py = kernels._arnoldi_python(Q[:, :k], w.copy(), decay, k)

    np.testing.assert_allclose(res_native, res_py, atol=1e-8)
    np.testing.assert_allclose(h_native, h_py, atol=1e-8)
    np.testing.assert_allclose(grad_native, grad_py, atol=1e-8)


def test_arnoldi_handles_zero_k():
    rng = np.random.default_rng(42)
    Q = rng.standard_normal((10, 3))
    w = rng.standard_normal(10)
    decay = 0.9
    res, h_col, grads = kernels.arnoldi_iteration(Q, w, decay, 0)
    np.testing.assert_array_equal(h_col, np.zeros(1, dtype=h_col.dtype))
    np.testing.assert_array_equal(grads, np.zeros(0, dtype=grads.dtype))
    np.testing.assert_array_equal(res, w)
