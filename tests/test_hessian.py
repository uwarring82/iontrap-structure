# SPDX-License-Identifier: MIT
"""Finite-difference verification of the analytic derivatives (§9.1).

Closes the derivative chain energy → gradient → Hessian: the Coulomb gradient
against the central difference of the energy, the Coulomb Hessian against the
central difference of the gradient, and the *total* (trap + Coulomb) Hessian —
the matrix ``normal_modes`` actually diagonalises — against the central
difference of the total gradient. Checks run in dimensionless units (k_e = 1,
O(1) charges/masses) for good conditioning.
"""

from __future__ import annotations

import numpy as np

from iontrap_structure import HarmonicTrap, coulomb


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


def test_coulomb_gradient_matches_energy_finite_difference() -> None:
    """∂E/∂r (analytic) matches the central difference of the Coulomb energy."""
    positions = np.array([[0.0, 0.0, -1.3], [0.2, -0.1, 0.0], [-0.1, 0.15, 1.4]])
    charges = np.ones(3)
    n = len(positions)

    analytic = coulomb.gradient(positions, charges, k_e=1.0).ravel()
    flat = positions.ravel()
    numeric = np.zeros(3 * n)
    h = 1e-6
    for k in range(3 * n):
        plus, minus = flat.copy(), flat.copy()
        plus[k] += h
        minus[k] -= h
        e_plus = coulomb.energy(plus.reshape(n, 3), charges, k_e=1.0)
        e_minus = coulomb.energy(minus.reshape(n, 3), charges, k_e=1.0)
        numeric[k] = (e_plus - e_minus) / (2.0 * h)

    assert np.allclose(analytic, numeric, rtol=1e-6, atol=1e-7)


def test_total_hessian_matches_finite_difference() -> None:
    """The full trap + Coulomb Hessian (the matrix ``normal_modes`` diagonalises)
    matches the central difference of the full gradient, for an off-axis config.

    This exercises ``HarmonicTrap.gradient``/``.hessian`` together with the
    Coulomb kernels in the exact combination the mode solver assembles.
    """
    trap = HarmonicTrap(wx=1.3, wy=1.7, wz=1.0)  # rad·s⁻¹; only the ratios matter here
    masses = np.ones(3)
    charges = np.ones(3)
    positions = np.array([[0.1, 0.0, -1.2], [0.0, 0.15, 0.05], [-0.1, -0.1, 1.25]])
    n = len(positions)

    def total_gradient(flat: np.ndarray) -> np.ndarray:
        u = flat.reshape(n, 3)
        return (trap.gradient(u, masses) + coulomb.gradient(u, charges, k_e=1.0)).ravel()

    analytic = trap.hessian(masses) + coulomb.hessian(positions, charges, k_e=1.0)

    flat = positions.ravel()
    numeric = np.zeros((3 * n, 3 * n))
    h = 1e-6
    for k in range(3 * n):
        plus, minus = flat.copy(), flat.copy()
        plus[k] += h
        minus[k] -= h
        numeric[:, k] = (total_gradient(plus) - total_gradient(minus)) / (2.0 * h)
    numeric = 0.5 * (numeric + numeric.T)

    assert np.allclose(analytic, numeric, rtol=1e-5, atol=1e-6)
