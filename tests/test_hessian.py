# SPDX-License-Identifier: MIT
"""Finite-difference verification of the analytic Coulomb Hessian (§9.1)."""

from __future__ import annotations

import numpy as np

from iontrap_structure import coulomb


def _finite_difference_hessian(positions: np.ndarray, charges: np.ndarray, *, h: float) -> np.ndarray:
    n = len(positions)
    flat = positions.ravel().astype(float)

    def grad_flat(x: np.ndarray) -> np.ndarray:
        return coulomb.gradient(x.reshape(n, 3), charges, k_e=1.0).ravel()

    H = np.zeros((3 * n, 3 * n))
    for k in range(3 * n):
        plus = flat.copy()
        minus = flat.copy()
        plus[k] += h
        minus[k] -= h
        H[:, k] = (grad_flat(plus) - grad_flat(minus)) / (2.0 * h)
    return 0.5 * (H + H.T)


def test_coulomb_hessian_matches_finite_difference() -> None:
    # Dimensionless units (k_e = 1, unit charges), O(1) non-overlapping config.
    positions = np.array(
        [
            [0.0, 0.0, -1.3],
            [0.2, -0.1, 0.0],
            [-0.1, 0.15, 1.4],
        ]
    )
    charges = np.ones(3)

    analytic = coulomb.hessian(positions, charges, k_e=1.0)
    numeric = _finite_difference_hessian(positions, charges, h=1e-6)

    assert np.allclose(analytic, numeric, rtol=1e-5, atol=1e-6)


def test_coulomb_hessian_is_symmetric() -> None:
    positions = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0], [0.3, 0.2, 0.0]])
    H = coulomb.hessian(positions, np.ones(3), k_e=1.0)
    assert np.allclose(H, H.T, atol=1e-12)
