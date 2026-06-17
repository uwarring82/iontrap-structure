# SPDX-License-Identifier: MIT
"""Physical constants (SI), sourced from :mod:`scipy.constants`.

The package works in SI throughout the public API (CONVENTIONS §1 of the
parent ``iontrap-dynamics``): positions in metres, masses in kilograms,
charges in coulombs, angular frequencies in rad·s⁻¹.
"""

from __future__ import annotations

import numpy as np
from scipy.constants import atomic_mass, elementary_charge, epsilon_0
from scipy.constants import k as boltzmann

#: Coulomb constant k_e = 1 / (4 π ε₀), in N·m²·C⁻².
COULOMB_CONSTANT: float = 1.0 / (4.0 * np.pi * epsilon_0)

__all__ = [
    "COULOMB_CONSTANT",
    "atomic_mass",
    "boltzmann",
    "elementary_charge",
    "epsilon_0",
]
