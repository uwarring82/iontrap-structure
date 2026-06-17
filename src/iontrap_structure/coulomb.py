# SPDX-License-Identifier: MIT
"""Pairwise Coulomb interaction: energy, gradient, and Hessian.

All quantities are SI by default (``k_e`` = the Coulomb constant). The
``k_e`` argument is exposed so the equilibrium solver can reuse these
functions in dimensionless units (``k_e = 1`` with charge ratios).

Coordinate layout for gradients/Hessians is **per-ion contiguous**:
the 3N-vector is ``[x₀, y₀, z₀, x₁, y₁, z₁, …]``, so a ``(3N,)`` mode
eigenvector reshapes directly to ``(N, 3)`` (no axis-blocked transpose).

These are dense O(N²) kernels — correct and clear for the small-N first
slice. Large-N acceleration (FMM / neighbour lists / sparse Hessian) is
deliberately out of scope here (see the origin task card, §8).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ._constants import COULOMB_CONSTANT


def energy(positions: NDArray[np.float64], charges: NDArray[np.float64], *, k_e: float = COULOMB_CONSTANT) -> float:
    """Total Coulomb potential energy, Σ_{i<j} k_e q_i q_j / r_ij."""
    positions = np.asarray(positions, dtype=float)
    charges = np.asarray(charges, dtype=float)
    n = len(positions)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = float(np.linalg.norm(positions[i] - positions[j]))
            total += k_e * charges[i] * charges[j] / r
    return total


def gradient(positions: NDArray[np.float64], charges: NDArray[np.float64], *, k_e: float = COULOMB_CONSTANT) -> NDArray[np.float64]:
    """Gradient ∂E/∂r, shape ``(N, 3)``.

    ∂E/∂r_i = −Σ_{j≠i} k_e q_i q_j (r_i − r_j) / r_ij³ (the negative of the
    repulsive force on ion i).
    """
    positions = np.asarray(positions, dtype=float)
    charges = np.asarray(charges, dtype=float)
    n = len(positions)
    grad = np.zeros((n, 3))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = positions[i] - positions[j]
            r = float(np.linalg.norm(d))
            grad[i] += -k_e * charges[i] * charges[j] * d / r**3
    return grad


def hessian(positions: NDArray[np.float64], charges: NDArray[np.float64], *, k_e: float = COULOMB_CONSTANT) -> NDArray[np.float64]:
    """Hessian ∂²E/∂r∂r, shape ``(3N, 3N)``, per-ion-contiguous layout.

    Per 1/r pair (with d = r_i − r_j, r = |d|):
      ∂²E/∂r_i∂r_i block = +k_e q_i q_j (3 d⊗d / r⁵ − I / r³)
      ∂²E/∂r_i∂r_j block = −k_e q_i q_j (3 d⊗d / r⁵ − I / r³)
    """
    positions = np.asarray(positions, dtype=float)
    charges = np.asarray(charges, dtype=float)
    n = len(positions)
    H = np.zeros((3 * n, 3 * n))
    eye = np.eye(3)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = positions[i] - positions[j]
            r = float(np.linalg.norm(d))
            block = k_e * charges[i] * charges[j] * (3.0 * np.outer(d, d) / r**5 - eye / r**3)
            H[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] += block
            H[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] += -block
    return H
