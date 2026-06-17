# SPDX-License-Identifier: MIT
"""Immutable result records.

The package exposes **pure functions over frozen records** (origin task
card §7.1), matching the parent ``iontrap-dynamics`` house style. There is
no fluent/stateful builder and no public DSL.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .trap import HarmonicTrap


@dataclass(frozen=True, slots=True, kw_only=True)
class EquilibriumResult:
    """Equilibrium configuration of a trapped ion crystal.

    Attributes
    ----------
    positions
        Equilibrium positions, shape ``(N, 3)``, metres. NOT part of the
        ``ModeConfig`` contract — carried here in the sibling's own schema.
    masses, charges
        Per-ion mass (kg) and charge (C), shape ``(N,)``.
    trap
        The :class:`~iontrap_structure.trap.HarmonicTrap` used.
    converged
        Whether the force-balance solve reported success.
    residual_force
        Max |∂E/∂r| at the solution, in SI (N). A diagnostic of convergence.
    """

    positions: NDArray[np.float64]
    masses: NDArray[np.float64]
    charges: NDArray[np.float64]
    trap: HarmonicTrap
    converged: bool
    residual_force: float

    @property
    def n_ions(self) -> int:
        return len(self.positions)


@dataclass(frozen=True, slots=True, kw_only=True)
class ModeResult:
    """Normal modes about an equilibrium.

    Attributes
    ----------
    frequencies_rad_s
        Mode angular frequencies ω_m, shape ``(3N,)``, ascending, rad·s⁻¹.
    eigenvectors
        Mode eigenvectors, shape ``(3N, N, 3)``. ``eigenvectors[m]`` is the
        ``(N, 3)`` eigenvector of mode ``m``, normalised per the parent's
        CONVENTIONS §11: Σ_i ‖b_{i,m}‖² = 1, with distinct modes orthonormal.
        These are the eigenvectors of the *mass-symmetrised* dynamical matrix
        M^{-1/2} H M^{-1/2} (Euclidean-orthonormal), which is the §11 set —
        not the raw physical displacements (M-orthonormal).
    positions
        The equilibrium positions the modes were computed about, ``(N, 3)`` m.
    masses
        Per-ion masses, ``(N,)`` kg.
    """

    frequencies_rad_s: NDArray[np.float64]
    eigenvectors: NDArray[np.float64]
    positions: NDArray[np.float64]
    masses: NDArray[np.float64]

    @property
    def n_ions(self) -> int:
        return len(self.positions)

    @property
    def n_modes(self) -> int:
        return len(self.frequencies_rad_s)

    def to_mode_configs(self, labels: list[str] | None = None) -> list[Any]:
        """Export to ``iontrap_dynamics.ModeConfig`` records (CONVENTIONS §10/§11).

        If ``iontrap-dynamics`` is installed (the optional ``[interop]`` extra),
        returns a list of its ``ModeConfig`` objects. Otherwise returns a list
        of :class:`ModeConfigLike` records carrying the same fields, so the
        export is usable without the parent package present.
        """
        try:
            from iontrap_dynamics.modes import ModeConfig as _ModeConfig
        except ImportError:
            _ModeConfig = None

        out: list[Any] = []
        for m in range(self.n_modes):
            label = labels[m] if labels is not None else f"mode_{m}"
            freq = float(self.frequencies_rad_s[m])
            vec = np.ascontiguousarray(self.eigenvectors[m], dtype=float)
            if _ModeConfig is not None:
                out.append(_ModeConfig(label=label, frequency_rad_s=freq, eigenvector_per_ion=vec))
            else:
                out.append(ModeConfigLike(label=label, frequency_rad_s=freq, eigenvector_per_ion=vec))
        return out


@dataclass(frozen=True, slots=True, kw_only=True)
class ModeConfigLike:
    """Fallback mode record mirroring ``iontrap_dynamics.modes.ModeConfig``.

    Used by :meth:`ModeResult.to_mode_configs` when the parent package is not
    installed. Field names and semantics match the parent's §10/§11 contract.
    """

    label: str
    frequency_rad_s: float
    eigenvector_per_ion: NDArray[np.float64]


@dataclass(frozen=True, slots=True, kw_only=True)
class StructuralDiagnostics:
    """Order parameters / phase diagnostics for a configuration.

    Decoupled from the equilibrium/modes engine (origin task card §9.2); acts
    on any ``(N, 3)`` configuration.
    """

    coupling_parameter: float
    mean_nn_distance: float
    temperature_kelvin: float
