# SPDX-License-Identifier: MIT
"""Normal-mode analysis about an equilibrium configuration.

Builds the total Hessian (trap + Coulomb) at equilibrium, forms the
mass-symmetrised dynamical matrix D = M^{-1/2} H M^{-1/2}, and diagonalises
it. The eigenvalues give ω² (rad²·s⁻²); the eigenvectors are
Euclidean-orthonormal and so satisfy the parent's CONVENTIONS §11
normalisation Σ_i ‖b_{i,m}‖² = 1 directly.

Diagonalising the **symmetric** D — rather than the non-symmetric M⁻¹H — is
what makes the eigenvectors orthonormal in the §11 sense for mixed species
(the two coincide only for equal masses).
"""

from __future__ import annotations

import numpy as np

from . import coulomb
from .results import EquilibriumResult, ModeResult
from .trap import HarmonicTrap


def normal_modes(eq: EquilibriumResult) -> ModeResult:
    """Compute normal modes about an :class:`EquilibriumResult`."""
    trap: HarmonicTrap = eq.trap
    positions = eq.positions
    masses = eq.masses
    n = len(masses)

    H = trap.hessian(masses) + coulomb.hessian(positions, eq.charges)

    # Mass-symmetrise: D = M^{-1/2} H M^{-1/2}, with masses repeated per axis
    # in the per-ion-contiguous layout [x0,y0,z0, x1,y1,z1, …].
    m3 = np.repeat(masses, 3)
    inv_sqrt_m = 1.0 / np.sqrt(m3)
    D = H * np.outer(inv_sqrt_m, inv_sqrt_m)
    D = 0.5 * (D + D.T)  # symmetrise against round-off before eigh

    eigvals, eigvecs = np.linalg.eigh(D)
    frequencies = np.sqrt(np.clip(eigvals, 0.0, None))

    # Columns of eigvecs are the modes; reshape each to (N, 3).
    modes = np.stack([eigvecs[:, m].reshape(n, 3) for m in range(3 * n)], axis=0)

    return ModeResult(
        frequencies_rad_s=frequencies,
        eigenvectors=modes,
        positions=positions,
        masses=masses,
    )
