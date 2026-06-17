# SPDX-License-Identifier: MIT
"""Diagnostics layer: nearest-neighbour spacing and the Γ coupling parameter.

The analysis layer is engine-agnostic — it acts on any ``(N, 3)`` configuration
(origin task card §9.2). These tests pin the definitions against hand
computations and check the input guards.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.constants import elementary_charge, epsilon_0
from scipy.constants import k as boltzmann

from iontrap_structure import (
    coupling_parameter,
    diagnostics,
    mean_nearest_neighbour_distance,
)

K_E = 1.0 / (4.0 * np.pi * epsilon_0)


def test_mean_nn_distance_simple_geometry() -> None:
    # Points on a line at 0, 2, 5 µm → nearest-neighbour distances 2, 2, 3 µm.
    positions = np.array([[0, 0, 0], [0, 0, 2e-6], [0, 0, 5e-6]], dtype=float)
    expected = (2e-6 + 2e-6 + 3e-6) / 3.0
    assert mean_nearest_neighbour_distance(positions) == pytest.approx(expected)


def test_coupling_parameter_matches_definition() -> None:
    a = 5e-6
    positions = np.array([[0, 0, 0], [0, 0, a]], dtype=float)  # two ions, spacing a
    charges = np.full(2, elementary_charge)
    temperature = 1e-3
    expected = K_E * elementary_charge**2 / (a * boltzmann * temperature)
    assert coupling_parameter(positions, charges, temperature) == pytest.approx(expected, rel=1e-12)


def test_coupling_parameter_scales_inversely_with_temperature() -> None:
    positions = np.array([[0, 0, 0], [0, 0, 5e-6]], dtype=float)
    charges = np.full(2, elementary_charge)
    g1 = coupling_parameter(positions, charges, 1e-3)
    g2 = coupling_parameter(positions, charges, 2e-3)
    assert g1 == pytest.approx(2.0 * g2, rel=1e-12)


def test_diagnostics_bundle_is_consistent() -> None:
    a = 5e-6
    positions = np.array([[0, 0, 0], [0, 0, a]], dtype=float)
    charges = np.full(2, elementary_charge)
    d = diagnostics(positions, charges, temperature_kelvin=1e-3)
    assert d.mean_nn_distance == pytest.approx(a)
    assert d.temperature_kelvin == pytest.approx(1e-3)
    assert d.coupling_parameter == pytest.approx(coupling_parameter(positions, charges, 1e-3))


def test_nn_distance_requires_two_ions() -> None:
    with pytest.raises(ValueError):
        mean_nearest_neighbour_distance(np.zeros((1, 3)))


def test_coupling_parameter_rejects_nonpositive_temperature() -> None:
    positions = np.array([[0, 0, 0], [0, 0, 1e-6]], dtype=float)
    charges = np.full(2, elementary_charge)
    for bad_t in (0.0, -1.0):
        with pytest.raises(ValueError):
            coupling_parameter(positions, charges, bad_t)
