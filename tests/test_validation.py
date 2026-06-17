# SPDX-License-Identifier: MIT
"""Input-guard / contract tests for constructors and the equilibrium solver."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.constants import atomic_mass, elementary_charge

from iontrap_structure import HarmonicTrap, equilibrium, normal_modes

TWO_PI = 2.0 * np.pi


@pytest.mark.parametrize(
    "kwargs",
    [
        {"wx": 0.0, "wy": 1.0, "wz": 1.0},
        {"wx": 1.0, "wy": -1.0, "wz": 1.0},
        {"wx": 1.0, "wy": 1.0, "wz": 0.0},
    ],
)
def test_trap_rejects_nonpositive_frequencies(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        HarmonicTrap(**kwargs)


def test_equilibrium_rejects_mismatched_masses_charges() -> None:
    trap = HarmonicTrap(wx=TWO_PI * 8e6, wy=TWO_PI * 9e6, wz=TWO_PI * 1e6)
    masses = np.full(3, 25.0 * atomic_mass)
    charges = np.full(2, elementary_charge)  # deliberately wrong length
    with pytest.raises(ValueError):
        equilibrium(trap=trap, masses=masses, charges=charges)


def test_trap_energy_matches_definition() -> None:
    """E = Σ_i ½ m_i (ω_x² x² + ω_y² y² + ω_z² z²)."""
    trap = HarmonicTrap(wx=2.0, wy=3.0, wz=5.0)
    positions = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, -1.0]])
    masses = np.array([1.0, 4.0])
    expected = 0.5 * (1.0 * (2.0**2 * 1.0)) + 0.5 * (4.0 * (3.0**2 * 4.0 + 5.0**2 * 1.0))
    assert trap.energy(positions, masses) == pytest.approx(expected)


def test_equilibrium_accepts_custom_initial_positions() -> None:
    """A perturbed SI starting guess converges to the same equilibrium."""
    trap = HarmonicTrap(wx=TWO_PI * 8e6, wy=TWO_PI * 9e6, wz=TWO_PI * 1e6)
    masses = np.full(3, 25.0 * atomic_mass)
    charges = np.full(3, elementary_charge)

    default = equilibrium(trap=trap, masses=masses, charges=charges)
    guess = default.positions + np.array([[0.0, 0.0, 1e-6], [1e-7, 0.0, 0.0], [0.0, -1e-7, -1e-6]])
    custom = equilibrium(trap=trap, masses=masses, charges=charges, initial_positions=guess)

    assert custom.converged
    z_default = np.sort(default.positions[:, 2])
    z_custom = np.sort(custom.positions[:, 2])
    assert np.allclose(z_custom, z_default, atol=1e-12)


def test_normal_modes_rejects_unstable_equilibrium() -> None:
    """``normal_modes`` flags a saddle instead of clipping a negative ω² to a
    spurious real frequency.

    With radial confinement far weaker than axial, the on-axis 3-ion chain is
    past the linear→zigzag instability: ``equilibrium`` still converges to the
    (symmetric) on-axis *stationary* point, but it is a saddle, so the
    dynamical matrix is not positive semi-definite.
    """
    weak_radial = HarmonicTrap(wx=TWO_PI * 0.3e6, wy=TWO_PI * 0.3e6, wz=TWO_PI * 1e6)
    masses = np.full(3, 25.0 * atomic_mass)
    charges = np.full(3, elementary_charge)
    eq = equilibrium(trap=weak_radial, masses=masses, charges=charges)
    assert eq.converged  # the on-axis stationary point is found...
    with pytest.raises(ValueError, match="not positive semi-definite"):
        normal_modes(eq)  # ...but it is unstable, so this must not return real freqs
