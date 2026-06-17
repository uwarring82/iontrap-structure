# SPDX-License-Identifier: MIT
"""Equilibrium-configuration solver.

Solves the force balance ∂E/∂r = 0 for the trap + Coulomb potential. The
solve is done in **dimensionless units** (length scale ℓ = (k_e q_ref² /
(m_ref ω_z²))^{1/3}) for good conditioning, then rescaled to SI. An analytic
Jacobian (the scaled Hessian) is supplied, so the root find converges to near
machine precision — comfortably inside the §9.1 acceptance tolerances.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root

from . import coulomb
from ._constants import COULOMB_CONSTANT
from .results import EquilibriumResult
from .trap import HarmonicTrap


def length_scale(trap: HarmonicTrap, m_ref: float, q_ref: float) -> float:
    """Characteristic length ℓ = (k_e q_ref² / (m_ref ω_z²))^{1/3}, metres."""
    return float((COULOMB_CONSTANT * q_ref**2 / (m_ref * trap.wz**2)) ** (1.0 / 3.0))


def linear_chain_guess(n: int, *, spacing: float = 1.8) -> NDArray[np.float64]:
    """Dimensionless initial guess: ions on the z-axis, evenly spread."""
    u = np.zeros((n, 3))
    u[:, 2] = (np.arange(n) - (n - 1) / 2.0) * spacing
    return u


def equilibrium(
    *,
    trap: HarmonicTrap,
    masses: NDArray[np.float64],
    charges: NDArray[np.float64],
    initial_positions: NDArray[np.float64] | None = None,
) -> EquilibriumResult:
    """Find equilibrium positions for ``N`` ions in ``trap``.

    Parameters
    ----------
    trap
        Harmonic confinement.
    masses, charges
        Per-ion mass (kg) and charge (C), shape ``(N,)``.
    initial_positions
        Optional ``(N, 3)`` starting guess in metres. Defaults to an on-axis
        linear-chain guess (suitable when ω_x, ω_y ≫ ω_z).
    """
    masses = np.asarray(masses, dtype=float)
    charges = np.asarray(charges, dtype=float)
    n = len(masses)
    if charges.shape != (n,):
        raise ValueError("masses and charges must have matching shape (N,).")

    m_ref = float(masses[0])
    q_ref = float(abs(charges[0]))
    ell = length_scale(trap, m_ref, q_ref)

    mu = masses / m_ref                     # mass ratios
    qr = charges / q_ref                    # charge ratios
    w2_scaled = (trap.omega / trap.wz) ** 2  # [α_x², α_y², 1]

    if initial_positions is None:
        u0 = linear_chain_guess(n)
    else:
        u0 = np.asarray(initial_positions, dtype=float) / ell

    def grad_scaled(uflat: NDArray[np.float64]) -> NDArray[np.float64]:
        u = uflat.reshape(n, 3)
        g = mu[:, None] * w2_scaled[None, :] * u
        g += coulomb.gradient(u, qr, k_e=1.0)
        return g.ravel()

    def jac_scaled(uflat: NDArray[np.float64]) -> NDArray[np.float64]:
        u = uflat.reshape(n, 3)
        H = coulomb.hessian(u, qr, k_e=1.0)
        for i in range(n):
            H[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] += np.diag(mu[i] * w2_scaled)
        return H

    sol = root(grad_scaled, u0.ravel(), jac=jac_scaled, method="hybr", tol=1e-13)
    positions = sol.x.reshape(n, 3) * ell

    # Judge convergence by the achieved force balance, not solely by the MINPACK
    # success flag: ``hybr`` can report a false negative even at a solution
    # accurate to machine precision (e.g. the exactly-zero radial subspace of an
    # on-axis chain). The scaled residual is dimensionless and scale-free.
    scaled_residual = float(np.max(np.abs(grad_scaled(sol.x))))
    g_si = trap.gradient(positions, masses) + coulomb.gradient(positions, charges)
    residual_force = float(np.max(np.abs(g_si)))
    converged = scaled_residual < 1e-8

    return EquilibriumResult(
        positions=positions,
        masses=masses,
        charges=charges,
        trap=trap,
        converged=converged,
        residual_force=residual_force,
    )
