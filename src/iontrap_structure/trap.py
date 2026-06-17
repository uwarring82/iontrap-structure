# SPDX-License-Identifier: MIT
"""Harmonic (pseudopotential) trap model.

The first slice models the time-averaged secular pseudopotential of a
linear Paul trap as an anisotropic 3D harmonic well with angular secular
frequencies (ω_x, ω_y, ω_z) in rad·s⁻¹. The full time-dependent RF
micromotion is **not** modelled here (out of scope — origin task card §7.2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True, kw_only=True)
class HarmonicTrap:
    """Anisotropic harmonic confinement.

    Parameters
    ----------
    wx, wy, wz
        Secular angular frequencies, rad·s⁻¹. For a linear chain along z,
        use ``wx, wy >> wz`` so the ions align on the trap axis.
    """

    wx: float
    wy: float
    wz: float

    def __post_init__(self) -> None:
        if not (self.wx > 0 and self.wy > 0 and self.wz > 0):
            raise ValueError("Trap secular frequencies must be positive (rad·s⁻¹).")

    @property
    def omega(self) -> NDArray[np.float64]:
        """Secular frequencies as a 3-vector ``[ω_x, ω_y, ω_z]``."""
        return np.array([self.wx, self.wy, self.wz])

    def energy(self, positions: NDArray[np.float64], masses: NDArray[np.float64]) -> float:
        """Trap potential energy Σ_i ½ m_i (ω_x² x_i² + ω_y² y_i² + ω_z² z_i²)."""
        positions = np.asarray(positions, dtype=float)
        masses = np.asarray(masses, dtype=float)
        return 0.5 * float(np.sum(masses[:, None] * self.omega[None, :] ** 2 * positions**2))

    def gradient(self, positions: NDArray[np.float64], masses: NDArray[np.float64]) -> NDArray[np.float64]:
        """Trap gradient ∂E/∂r, shape ``(N, 3)``: m_i ω_c² r_{i,c}."""
        positions = np.asarray(positions, dtype=float)
        masses = np.asarray(masses, dtype=float)
        return masses[:, None] * self.omega[None, :] ** 2 * positions

    def hessian(self, masses: NDArray[np.float64]) -> NDArray[np.float64]:
        """Trap Hessian, shape ``(3N, 3N)``: block-diagonal diag(m_i ω_c²)."""
        masses = np.asarray(masses, dtype=float)
        n = len(masses)
        H = np.zeros((3 * n, 3 * n))
        w2 = self.omega**2
        for i in range(n):
            H[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] = np.diag(masses[i] * w2)
        return H
