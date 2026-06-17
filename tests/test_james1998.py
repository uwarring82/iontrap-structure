# SPDX-License-Identifier: MIT
"""James-1998 linear-chain oracle (Appl. Phys. B 66, 181).

Validates equilibrium positions and the two universal low axial mode
frequencies against closed-form results, per the §9.1 acceptance criteria.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.constants import atomic_mass, elementary_charge

from iontrap_structure import HarmonicTrap, equilibrium, length_scale, normal_modes

TWO_PI = 2.0 * np.pi
M_MG = 25.0 * atomic_mass          # ²⁵Mg⁺
Q = elementary_charge


def _linear_trap() -> HarmonicTrap:
    # Strong radial confinement (α = 10) → ions form a chain on the z-axis.
    return HarmonicTrap(wx=TWO_PI * 10e6, wy=TWO_PI * 10e6, wz=TWO_PI * 1e6)


def _single_species(n: int) -> tuple[np.ndarray, np.ndarray]:
    return np.full(n, M_MG), np.full(n, Q)


def test_two_ion_equilibrium_positions() -> None:
    trap = _linear_trap()
    masses, charges = _single_species(2)
    eq = equilibrium(trap=trap, masses=masses, charges=charges)
    ell = length_scale(trap, M_MG, Q)

    assert eq.converged
    u_z = np.sort(eq.positions[:, 2] / ell)
    expected = (0.25) ** (1.0 / 3.0)  # u³ = 1/4
    assert u_z == pytest.approx([-expected, expected], rel=1e-6)
    assert np.allclose(eq.positions[:, :2], 0.0, atol=1e-9 * ell)


def test_three_ion_equilibrium_positions() -> None:
    trap = _linear_trap()
    masses, charges = _single_species(3)
    eq = equilibrium(trap=trap, masses=masses, charges=charges)
    ell = length_scale(trap, M_MG, Q)

    assert eq.converged
    u_z = np.sort(eq.positions[:, 2] / ell)
    outer = (1.25) ** (1.0 / 3.0)  # u³ = 5/4
    assert u_z == pytest.approx([-outer, 0.0, outer], rel=1e-6, abs=1e-9)


@pytest.mark.parametrize("n", [2, 3, 4])
def test_universal_axial_modes(n: int) -> None:
    """COM mode = ω_z and the stretch mode = √3 ω_z for any N (James 1998)."""
    trap = _linear_trap()
    masses, charges = _single_species(n)
    eq = equilibrium(trap=trap, masses=masses, charges=charges)
    modes = normal_modes(eq)

    freqs = modes.frequencies_rad_s  # ascending
    assert freqs[0] == pytest.approx(trap.wz, rel=1e-6)
    assert freqs[1] == pytest.approx(np.sqrt(3.0) * trap.wz, rel=1e-6)

    # The two lowest modes are axial (z-polarised).
    for m in (0, 1):
        b = modes.eigenvectors[m]
        z_fraction = float(np.sum(b[:, 2] ** 2) / np.sum(b**2))
        assert z_fraction > 0.999
