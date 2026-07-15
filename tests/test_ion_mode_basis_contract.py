# SPDX-License-Identifier: MIT
"""IonModeBasis payload contract (GT3b producer half).

Locks the wire invariants of the ``iontrap-structure`` → ``iontrap-dynamics``
handshake (``TC-gt3b-ion-symplectic-adapter.md`` v0.3, producer obligations):
shapes, ``ω > 0``, the flattened ``(3N, 3N)`` Gram ``= I``, the exact
row/column ordering round-trip, the canonical ``coordinate_frame`` identifier,
the tagged local-reference gauge, and reproducibility for a given ``ModeResult``.
Mirrors ``test_modeconfig_contract.py``.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from scipy.constants import atomic_mass, elementary_charge

from iontrap_structure import (
    ION_MODE_BASIS_SCHEMA_VERSION,
    HarmonicTrap,
    IonModeBasis,
    equilibrium,
    ion_mode_basis,
    normal_modes,
)
from iontrap_structure import coulomb as _coulomb

TWO_PI = 2.0 * np.pi
M_MG = 25.0 * atomic_mass
Q = elementary_charge


def _trap() -> HarmonicTrap:
    return HarmonicTrap(wx=TWO_PI * 8e6, wy=TWO_PI * 9e6, wz=TWO_PI * 1e6)


def _eq_modes(n: int, *, mass_ratio: float = 1.0):
    trap = _trap()
    masses = np.full(n, M_MG)
    if n >= 2 and mass_ratio != 1.0:
        masses[-1] = M_MG * mass_ratio
    charges = np.full(n, Q)
    eq = equilibrium(trap=trap, masses=masses, charges=charges)
    return trap, charges, normal_modes(eq)


@pytest.mark.parametrize("n", [2, 3, 5])
def test_shapes(n: int) -> None:
    trap, _, modes = _eq_modes(n)
    payload = ion_mode_basis(modes, trap)
    tn = 3 * n
    assert isinstance(payload, IonModeBasis)
    assert payload.n_ions == n
    assert payload.n_modes == tn
    assert payload.frequencies_rad_s.shape == (tn,)
    assert payload.mass_weighted_eigenvectors.shape == (tn, tn)
    assert payload.masses_kg.shape == (n,)
    assert payload.local_reference_frequencies_rad_s.shape == (tn,)


@pytest.mark.parametrize("n", [2, 3, 5])
def test_frequencies_positive_and_aligned(n: int) -> None:
    trap, _, modes = _eq_modes(n)
    payload = ion_mode_basis(modes, trap)
    assert np.all(payload.frequencies_rad_s > 0.0)
    # Frequency↔column alignment: column m ↔ frequencies_rad_s[m], unchanged
    # from the ModeResult ordering.
    assert np.array_equal(payload.frequencies_rad_s, modes.frequencies_rad_s)


@pytest.mark.parametrize("n", [2, 3, 5])
def test_flattened_gram_is_identity(n: int) -> None:
    # Columns of B (the 3N modes) orthonormal → Gram = I. This is the producer
    # guarantee the consumer's S = diag(X, P) symplecticity (X Pᵀ = I) rests on.
    trap, _, modes = _eq_modes(n)
    b = ion_mode_basis(modes, trap).mass_weighted_eigenvectors
    gram = b.T @ b
    assert np.allclose(gram, np.eye(3 * n), atol=1e-8)
    assert np.allclose(np.linalg.norm(b, axis=0), 1.0, atol=1e-10)  # per-mode unit norm


@pytest.mark.parametrize("n", [2, 3])
def test_row_column_ordering_roundtrip(n: int) -> None:
    # B[3i+c, m] = ±eigenvectors[m, i, c] (sign canonicalised per column).
    trap, _, modes = _eq_modes(n)
    b = ion_mode_basis(modes, trap).mass_weighted_eigenvectors
    for m in range(3 * n):
        col = b[:, m]
        for i in range(n):
            for c in range(3):
                assert abs(col[3 * i + c]) == pytest.approx(abs(modes.eigenvectors[m, i, c]))
        # Reshaping a column back to (N, 3) recovers |eigenvector[m]| exactly.
        assert np.allclose(np.abs(col.reshape(n, 3)), np.abs(modes.eigenvectors[m]))


@pytest.mark.parametrize("n", [2, 3, 5])
def test_sign_canonicalization_convention(n: int) -> None:
    # Each column's largest-|component| entry is non-negative, and the payload is
    # ± the raw eigensolver column (a pure per-column sign gauge).
    trap, _, modes = _eq_modes(n)
    b = ion_mode_basis(modes, trap).mass_weighted_eigenvectors
    raw = np.asarray(modes.eigenvectors, dtype=float).reshape(3 * n, 3 * n).T
    for m in range(3 * n):
        col = b[:, m]
        assert col[int(np.argmax(np.abs(col)))] >= 0.0
        nz = np.abs(raw[:, m]) > 1e-12
        ratio = col[nz] / raw[nz, m]
        assert np.allclose(np.abs(ratio), 1.0)  # only signs differ
        assert np.allclose(ratio, ratio[0])  # a single global sign per column


@pytest.mark.parametrize("n", [2, 3, 5])
def test_local_reference_trap_curvature(n: int) -> None:
    # Default gauge (i): ω_local,3i+c = trap.omega[c] for every ion.
    trap, _, modes = _eq_modes(n)
    payload = ion_mode_basis(modes, trap)
    expected = np.tile(trap.omega, n)
    assert np.array_equal(payload.local_reference_frequencies_rad_s, expected)
    assert payload.normalization_weighting_tags["local_reference"] == "trap_curvature"


@pytest.mark.parametrize("n", [2, 3])
def test_local_reference_diagonal_hessian(n: int) -> None:
    # Gauge (ii): √(H_jj / m_j) with Coulomb curvature included.
    trap, charges, modes = _eq_modes(n)
    payload = ion_mode_basis(modes, trap, local_reference="diagonal_hessian", charges=charges)
    hess = trap.hessian(modes.masses) + _coulomb.hessian(modes.positions, charges)
    expected = np.sqrt(np.diag(hess) / np.repeat(modes.masses, 3))
    assert np.allclose(payload.local_reference_frequencies_rad_s, expected)
    assert payload.normalization_weighting_tags["local_reference"] == "diagonal_hessian"


def test_diagonal_hessian_requires_charges() -> None:
    trap, _, modes = _eq_modes(3)
    with pytest.raises(ValueError, match="charges"):
        ion_mode_basis(modes, trap, local_reference="diagonal_hessian")


def test_diagonal_hessian_rejects_bad_charges_shape() -> None:
    trap, _, modes = _eq_modes(3)
    bad_charges = np.full(modes.n_ions + 1, elementary_charge)
    with pytest.raises(ValueError, match="charges must have shape"):
        ion_mode_basis(modes, trap, local_reference="diagonal_hessian", charges=bad_charges)


def test_unknown_local_reference_raises() -> None:
    trap, _, modes = _eq_modes(2)
    with pytest.raises(ValueError, match="unknown local_reference"):
        ion_mode_basis(modes, trap, local_reference="not_a_gauge")


@pytest.mark.parametrize("mass_ratio", [1.0, 1.3, 26.0 / 25.0])
def test_mixed_species_gram_and_masses(mass_ratio: float) -> None:
    # The mass-symmetrised basis stays orthonormal for unequal masses, and the
    # payload carries the per-ion masses verbatim.
    trap, _, modes = _eq_modes(3, mass_ratio=mass_ratio)
    payload = ion_mode_basis(modes, trap)
    b = payload.mass_weighted_eigenvectors
    assert np.allclose(b.T @ b, np.eye(9), atol=1e-8)
    assert np.array_equal(payload.masses_kg, modes.masses)


# The consumer (iontrap-dynamics ion_modes._CANONICAL_COORDINATE_FRAME) enforces
# this exact string by byte-for-byte match; the producer must emit it verbatim.
_CONSUMER_CANONICAL_COORDINATE_FRAME = "ion-major-axis-minor;row=axes_per_ion*i+c;col=mode"


def test_schema_version_and_tags() -> None:
    trap, _, modes = _eq_modes(2)
    payload = ion_mode_basis(modes, trap)
    assert payload.schema_version == ION_MODE_BASIS_SCHEMA_VERSION
    tags = payload.normalization_weighting_tags
    assert tags["mass_symmetrised"] is True
    assert tags["per_mode_unit_norm"] is True
    assert tags["eigenvector_sign_convention"] == "largest_abs_component_positive"
    # Handedness/axis labels moved into the tags (the frame is now a fixed identifier).
    assert tags["trap_frame_handedness"] == "right_handed"
    assert tags["coordinate_axes"] == "x,y,z"


@pytest.mark.parametrize("n", [2, 3, 5])
def test_coordinate_frame_matches_consumer_canonical_id(n: int) -> None:
    # The coordinate_frame tag is a pinned wire invariant compared by exact match
    # on the consumer side — a prose divergence would be a loud rejection there.
    trap, _, modes = _eq_modes(n)
    payload = ion_mode_basis(modes, trap)
    assert payload.coordinate_frame == _CONSUMER_CANONICAL_COORDINATE_FRAME


def test_rejects_nonpositive_mode_frequency() -> None:
    # normal_modes clips soft modes to 0 (raising only on materially negative ω²),
    # so a critical-point/hand-built ModeResult can carry ω = 0. The consumer's
    # S-map divides by ω, so the producer must reject it at the source.
    trap, _, modes = _eq_modes(2)
    freqs = modes.frequencies_rad_s.copy()
    freqs[0] = 0.0
    bad = dataclasses.replace(modes, frequencies_rad_s=freqs)
    with pytest.raises(ValueError, match="finite and strictly"):
        ion_mode_basis(bad, trap)


def test_asdict_is_serialization_neutral() -> None:
    trap, _, modes = _eq_modes(3)
    payload = ion_mode_basis(modes, trap)
    wire = payload.asdict()
    assert set(wire) == {
        "schema_version",
        "frequencies_rad_s",
        "mass_weighted_eigenvectors",
        "masses_kg",
        "local_reference_frequencies_rad_s",
        "coordinate_frame",
        "normalization_weighting_tags",
    }
    # Plain types only — no IonModeBasis reference leaks into the wire form.
    assert isinstance(wire["schema_version"], int)
    assert isinstance(wire["mass_weighted_eigenvectors"], np.ndarray)
    assert type(wire["mass_weighted_eigenvectors"]) is np.ndarray
    assert isinstance(wire["normalization_weighting_tags"], dict)
    assert isinstance(wire["coordinate_frame"], str)
    # A copy, not an alias into the record's mapping.
    wire["normalization_weighting_tags"]["mass_symmetrised"] = "mutated"
    assert payload.normalization_weighting_tags["mass_symmetrised"] is True
    # Array buffers are also detached from the frozen record.
    for key in (
        "frequencies_rad_s",
        "mass_weighted_eigenvectors",
        "masses_kg",
        "local_reference_frequencies_rad_s",
    ):
        assert wire[key] is not getattr(payload, key)
    # as_arrays is the documented alias.
    assert set(payload.as_arrays()) == set(wire)


def test_materializes_in_iontrap_dynamics_consumer() -> None:
    """The emitted payload round-trips through the real ``iontrap-dynamics`` consumer.

    Skipped unless the optional GT3b consumer is importable. Unlike the consumer's
    own unit test (which hand-builds the kwargs), this exercises the *actual*
    handshake — ``materialize_ion_mode_basis(**payload.asdict())`` — so a wire drift
    (``coordinate_frame`` id, the ``normalization_weighting_tags`` kwarg, the
    ``schema_version`` type) surfaces here rather than in downstream physics.
    """
    im = pytest.importorskip("iontrap_dynamics.ion_modes")
    trap, _, modes = _eq_modes(3)
    payload = ion_mode_basis(modes, trap)
    rec = im.materialize_ion_mode_basis(**payload.asdict())
    assert rec.coordinate_frame == payload.coordinate_frame
    # The consumer builds S from our arrays; it must be symplectic (S Ω Sᵀ = Ω).
    s = im.normal_to_local_symplectic(rec)
    n = rec.n_coords
    omega = np.zeros((2 * n, 2 * n))
    for k in range(n):
        omega[2 * k, 2 * k + 1] = 1.0
        omega[2 * k + 1, 2 * k] = -1.0
    assert np.allclose(s @ omega @ s.T, omega, atol=1e-10)


def test_reproducible_bitforbit() -> None:
    # Determinism the consumer relies on: same inputs → identical payload.
    trap, _charges, modes = _eq_modes(4)
    a = ion_mode_basis(modes, trap)
    b = ion_mode_basis(modes, trap)
    assert np.array_equal(a.mass_weighted_eigenvectors, b.mass_weighted_eigenvectors)
    assert np.array_equal(a.frequencies_rad_s, b.frequencies_rad_s)
    assert np.array_equal(a.local_reference_frequencies_rad_s, b.local_reference_frequencies_rad_s)
