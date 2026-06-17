# SPDX-License-Identifier: MIT
"""ModeConfig contract checks (parent CONVENTIONS §10/§11).

Independently asserts the normalisation, orthonormality, and positivity the
mode output must satisfy to be a valid ``ModeConfig`` payload (§9.1).
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest
from scipy.constants import atomic_mass, elementary_charge

from iontrap_structure import HarmonicTrap, equilibrium, normal_modes

TWO_PI = 2.0 * np.pi
M_MG = 25.0 * atomic_mass
Q = elementary_charge


def _modes(n: int):
    trap = HarmonicTrap(wx=TWO_PI * 8e6, wy=TWO_PI * 9e6, wz=TWO_PI * 1e6)
    masses, charges = np.full(n, M_MG), np.full(n, Q)
    return normal_modes(equilibrium(trap=trap, masses=masses, charges=charges))


@pytest.mark.parametrize("n", [2, 3, 5])
def test_per_mode_normalisation(n: int) -> None:
    modes = _modes(n)
    for m in range(modes.n_modes):
        norm = float(np.sum(modes.eigenvectors[m] ** 2))  # Σ_i ‖b_{i,m}‖²
        assert norm == pytest.approx(1.0, abs=1e-10)


@pytest.mark.parametrize("n", [2, 3, 5])
def test_cross_mode_orthonormality(n: int) -> None:
    modes = _modes(n)
    flat = modes.eigenvectors.reshape(modes.n_modes, -1)
    gram = flat @ flat.T
    assert np.allclose(gram, np.eye(modes.n_modes), atol=1e-8)


@pytest.mark.parametrize("n", [2, 3, 5])
def test_all_frequencies_positive(n: int) -> None:
    # A 3D trap confines every direction → no zero/soft modes.
    modes = _modes(n)
    assert np.all(modes.frequencies_rad_s > 0.0)
    assert modes.n_modes == 3 * n


def test_to_mode_configs_roundtrip() -> None:
    modes = _modes(3)
    cfgs = modes.to_mode_configs()
    assert len(cfgs) == modes.n_modes
    for m, cfg in enumerate(cfgs):
        assert cfg.frequency_rad_s == pytest.approx(modes.frequencies_rad_s[m])
        assert np.sum(cfg.eigenvector_per_ion**2) == pytest.approx(1.0, abs=1e-10)
        assert cfg.eigenvector_per_ion.shape == (3, 3)


def test_to_mode_configs_accepts_custom_labels() -> None:
    modes = _modes(2)
    labels = [f"axial_{m}" for m in range(modes.n_modes)]
    cfgs = modes.to_mode_configs(labels=labels)
    assert [c.label for c in cfgs] == labels


def test_result_n_ions_properties() -> None:
    trap = HarmonicTrap(wx=TWO_PI * 8e6, wy=TWO_PI * 9e6, wz=TWO_PI * 1e6)
    masses, charges = np.full(4, M_MG), np.full(4, Q)
    eq = equilibrium(trap=trap, masses=masses, charges=charges)
    modes = normal_modes(eq)
    assert eq.n_ions == 4
    assert modes.n_ions == 4
    assert modes.n_modes == 12


def test_to_mode_configs_uses_parent_class_when_importable(monkeypatch: pytest.MonkeyPatch) -> None:
    """The export prefers the real ``iontrap_dynamics`` ModeConfig when importable.

    Exercised deterministically (without the optional dependency) by injecting a
    stand-in ``iontrap_dynamics.modes.ModeConfig`` into ``sys.modules`` — this is
    the branch the real ``[interop]`` install would take.
    """

    class FakeModeConfig:
        def __init__(self, *, label: str, frequency_rad_s: float, eigenvector_per_ion):
            self.label = label
            self.frequency_rad_s = frequency_rad_s
            self.eigenvector_per_ion = eigenvector_per_ion

    pkg = types.ModuleType("iontrap_dynamics")
    mod = types.ModuleType("iontrap_dynamics.modes")
    mod.ModeConfig = FakeModeConfig  # type: ignore[attr-defined]
    pkg.modes = mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "iontrap_dynamics", pkg)
    monkeypatch.setitem(sys.modules, "iontrap_dynamics.modes", mod)

    modes = _modes(2)
    cfgs = modes.to_mode_configs()
    assert len(cfgs) == modes.n_modes
    assert all(isinstance(c, FakeModeConfig) for c in cfgs)


def test_to_mode_configs_uses_parent_if_available() -> None:
    """If iontrap-dynamics is genuinely installed, the export yields real objects."""
    ModeConfig = pytest.importorskip("iontrap_dynamics.modes").ModeConfig
    modes = _modes(2)
    cfgs = modes.to_mode_configs()
    assert all(isinstance(c, ModeConfig) for c in cfgs)
