# SPDX-License-Identifier: MIT
"""iontrap-structure — classical structural dynamics of trapped-ion crystals.

First slice: equilibrium configurations and normal modes for the
linear/harmonic regime, exporting modes to the ``iontrap-dynamics``
``ModeConfig`` contract (CONVENTIONS §10/§11). Positions, the plasma coupling
parameter Γ, and other structural diagnostics live in this package's own
schema, not in ``ModeConfig``.

See the origin/provenance record forwarded from ``iontrap-dynamics``:
``task cards/TC-structural-dynamics-foundation-survey.md`` (frozen v1.0).
"""

from __future__ import annotations

from .diagnostics import coupling_parameter, diagnostics, mean_nearest_neighbour_distance
from .equilibrium import equilibrium, length_scale, linear_chain_guess
from .modes import normal_modes
from .results import EquilibriumResult, ModeConfigLike, ModeResult, StructuralDiagnostics
from .trap import HarmonicTrap

__version__ = "0.1.0"

__all__ = [
    "EquilibriumResult",
    "HarmonicTrap",
    "ModeConfigLike",
    "ModeResult",
    "StructuralDiagnostics",
    "__version__",
    "coupling_parameter",
    "diagnostics",
    "equilibrium",
    "length_scale",
    "linear_chain_guess",
    "mean_nearest_neighbour_distance",
    "normal_modes",
]
