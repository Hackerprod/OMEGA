"""
Optional accelerated kernels for ACP/DTP routines.

The NumPy implementation is retained as a fallback so the module has no
mandatory compiled dependencies. When Numba is available, the inner Arnoldi
loop is JIT-compiled to reduce Python overhead on CPU-only workloads.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - numba is optional
    njit = None  # type: ignore[assignment]


NUMBA_AVAILABLE = njit is not None


def _arnoldi_python(Q: np.ndarray, w: np.ndarray, decay: float, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h_column = np.zeros(k + 1, dtype=w.dtype)
    gradients = np.zeros(k, dtype=w.dtype)
    for i in range(k):
        q_i = Q[:, i]
        dot_val = float(np.dot(q_i, w))
        weight = decay ** (k - 1 - i)
        h_mag = weight * abs(dot_val)
        coeff = np.sign(dot_val) * h_mag
        h_column[i] = coeff
        w = w - coeff * q_i
        exponent = k - 1 - i
        if exponent > 0:
            gradients[i] = exponent * (decay ** (exponent - 1)) * abs(dot_val)
    h_column[k] = np.linalg.norm(w)
    return w, h_column, gradients


if NUMBA_AVAILABLE:  # pragma: no cover - executed only when numba is installed

    @njit(cache=True)
    def _arnoldi_numba(Q: np.ndarray, w: np.ndarray, decay: float, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h_column = np.zeros(k + 1, dtype=w.dtype)
        gradients = np.zeros(k, dtype=w.dtype)
        for i in range(k):
            q_i = Q[:, i]
            dot_val = 0.0
            for idx in range(q_i.size):
                dot_val += q_i[idx] * w[idx]
            weight = decay ** (k - 1 - i)
            h_mag = weight * abs(dot_val)
            coeff = _sign(dot_val) * h_mag
            h_column[i] = coeff
            for idx in range(w.size):
                w[idx] -= coeff * q_i[idx]
            exponent = k - 1 - i
            if exponent > 0:
                gradients[i] = exponent * (decay ** (exponent - 1)) * abs(dot_val)
        norm_val = 0.0
        for idx in range(w.size):
            norm_val += w[idx] * w[idx]
        h_column[k] = norm_val ** 0.5
        return w, h_column, gradients

    @njit(cache=True)
    def _sign(value: float) -> float:
        if value > 0.0:
            return 1.0
        if value < 0.0:
            return -1.0
        return 0.0


def arnoldi_iteration(Q: np.ndarray, w: np.ndarray, decay: float, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Hessenberg column and monotonic gradients for the current Arnoldi step.

    Returns
    -------
    w_out : np.ndarray
        Residual vector after orthogonal projections.
    h_column : np.ndarray
        Hessenberg column (length k+1).
    gradients : np.ndarray
        Magnitudes of monotonic gradients used for diagnostics.
    """
    if k == 0:
        h_column = np.zeros(1, dtype=w.dtype)
        gradients = np.zeros(0, dtype=w.dtype)
        return w, h_column, gradients

    if NUMBA_AVAILABLE:
        return _arnoldi_numba(Q[:, :k], w.copy(), decay, k)
    return _arnoldi_python(Q[:, :k], w.copy(), decay, k)
