# SPDX-License-Identifier: MIT
"""Structural diagnostics / order parameters (decoupled analysis layer).

This layer is engine-agnostic: it acts on any ``(N, 3)`` configuration
(origin task card §9.2). The plasma coupling parameter Γ is defined inline;
richer order parameters (Steinhardt/hexatic, Voronoi, RDF) are intended to be
delegated to the optional ``freud`` dependency rather than reimplemented.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ._constants import COULOMB_CONSTANT, boltzmann
from .results import StructuralDiagnostics


def mean_nearest_neighbour_distance(positions: NDArray[np.float64]) -> float:
    """Mean over ions of the nearest-neighbour distance, metres."""
    positions = np.asarray(positions, dtype=float)
    n = len(positions)
    if n < 2:
        raise ValueError("Need at least two ions to define a neighbour distance.")
    nn = []
    for i in range(n):
        d = np.linalg.norm(positions - positions[i], axis=1)
        d[i] = np.inf
        nn.append(float(np.min(d)))
    return float(np.mean(nn))


def coupling_parameter(
    positions: NDArray[np.float64],
    charges: NDArray[np.float64],
    temperature_kelvin: float,
) -> float:
    """Plasma coupling parameter Γ = k_e q² / (a k_B T).

    A finite-system estimator: the characteristic spacing ``a`` is taken as the
    mean nearest-neighbour distance, and ``q`` as the mean ion charge magnitude.
    Γ ≫ 1 is the crystalline regime; Γ ≲ 1 the gaseous/plasma regime. This is a
    deliberately simple estimator — the Wigner–Seitz definition for a bulk OCP
    differs, and bulk comparisons should use a density-based ``a``.
    """
    if temperature_kelvin <= 0:
        raise ValueError("temperature_kelvin must be positive.")
    a = mean_nearest_neighbour_distance(positions)
    q = float(np.mean(np.abs(np.asarray(charges, dtype=float))))
    return COULOMB_CONSTANT * q**2 / (a * boltzmann * temperature_kelvin)


def diagnostics(
    positions: NDArray[np.float64],
    charges: NDArray[np.float64],
    temperature_kelvin: float,
) -> StructuralDiagnostics:
    """Bundle the available structural diagnostics into a frozen record."""
    a = mean_nearest_neighbour_distance(positions)
    gamma = coupling_parameter(positions, charges, temperature_kelvin)
    return StructuralDiagnostics(
        coupling_parameter=gamma,
        mean_nn_distance=a,
        temperature_kelvin=float(temperature_kelvin),
    )
