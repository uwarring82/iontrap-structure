# SPDX-License-Identifier: MIT
"""Mixed-species validation of the mass-symmetrised normal-mode path.

``modes.py`` diagonalises the *symmetric* dynamical matrix
``D = M^{-1/2} H M^{-1/2}`` rather than the non-symmetric ``M^{-1} H`` — which is
what makes the eigenvectors §11-orthonormal for *unequal* masses (the two routes
coincide only for equal masses). The first slice's other tests all use a single
species, so they never exercise that claim. These tests pin it down:

* the §11 contract under mixed species,
* the centre-of-mass frequency oracle (ω_c is a mode frequency for any masses),
* a direct, oracle-free generalized-eigenproblem residual ``H x = ω² M x``,
* a negative control: the physical displacements are M-orthonormal but *not*
  Euclidean-orthonormal for mixed species — the distinction the symmetric route
  is built to respect.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.constants import atomic_mass, elementary_charge

from iontrap_structure import HarmonicTrap, coulomb, equilibrium, normal_modes

TWO_PI = 2.0 * np.pi
AMU = atomic_mass
Q = elementary_charge

# Non-degenerate axes so the COM modes are uniquely identifiable.
TRAP = HarmonicTrap(wx=TWO_PI * 8e6, wy=TWO_PI * 9e6, wz=TWO_PI * 1e6)

# Mixed-species chains (unequal masses, singly charged). Amu values are
# illustrative real ions: ²⁵Mg⁺, ⁴⁴Ca⁺, ⁴⁰Ca⁺, ⁹Be⁺.
MIXED_AMU = [
    [25.0, 44.0],            # two species
    [25.0, 44.0, 25.0],      # sympathetic-cooling "sandwich"
    [9.0, 40.0, 40.0, 9.0],  # logic/coolant chain
]


def _modes_for(masses_amu: list[float]):
    masses = np.array(masses_amu) * AMU
    charges = np.full(len(masses), Q)
    eq = equilibrium(trap=TRAP, masses=masses, charges=charges)
    return eq, normal_modes(eq)


@pytest.mark.parametrize("masses_amu", MIXED_AMU)
def test_mixed_species_section11_contract(masses_amu: list[float]) -> None:
    """§11 still holds for unequal masses: per-mode norm 1, modes orthonormal."""
    _, modes = _modes_for(masses_amu)
    flat = modes.eigenvectors.reshape(modes.n_modes, -1)
    gram = flat @ flat.T
    assert np.allclose(gram, np.eye(modes.n_modes), atol=1e-8)
    assert np.all(modes.frequencies_rad_s > 0.0)


@pytest.mark.parametrize("masses_amu", MIXED_AMU)
def test_mixed_species_com_frequency_oracle(masses_amu: list[float]) -> None:
    """A mode sits at exactly ω_x, ω_y, ω_z for any masses (the COM modes).

    For mixed species the §11 eigenvector of the COM mode is the *physical*
    uniform translation re-expressed in the mass-weighted basis, i.e.
    ``b_i ∝ √m_i`` along the axis — but the frequency is still exactly ω_c.
    """
    masses = np.array(masses_amu)
    _, modes = _modes_for(masses_amu)
    n = len(masses)
    flat = modes.eigenvectors.reshape(modes.n_modes, -1)
    for axis, omega in enumerate(TRAP.omega):
        b = np.zeros((n, 3))
        # §11 COM eigenvector ∝ √m_i along the axis. Using amu here (the code
        # works in kg internally) is fine: the eigenvector direction is invariant
        # under a common positive rescaling of the masses, and it is normalised.
        b[:, axis] = np.sqrt(masses)
        b /= np.linalg.norm(b)
        overlaps = np.abs(flat @ b.ravel())
        m = int(np.argmax(overlaps))
        assert overlaps[m] == pytest.approx(1.0, abs=1e-8)
        assert modes.frequencies_rad_s[m] == pytest.approx(omega, rel=1e-8)


@pytest.mark.parametrize("masses_amu", MIXED_AMU)
def test_mixed_species_generalized_eigenproblem(masses_amu: list[float]) -> None:
    """Reported (ω_m, b_m) satisfy H x = ω² M x with x = M^{-1/2} b (oracle-free).

    This is the physical content of "normal mode": that the reported pairs
    diagonalise the equations of motion M ẍ = −H x. M is diagonal (masses
    repeated per axis), so M x is an elementwise scaling — no dense M needed.
    """
    eq, modes = _modes_for(masses_amu)
    H = TRAP.hessian(eq.masses) + coulomb.hessian(eq.positions, eq.charges)
    m3 = np.repeat(eq.masses, 3)
    for m in range(modes.n_modes):
        b = modes.eigenvectors[m].ravel()
        x = b / np.sqrt(m3)
        lhs = H @ x
        rhs = modes.frequencies_rad_s[m] ** 2 * (m3 * x)  # ω² M x
        rel = np.linalg.norm(lhs - rhs) / (np.linalg.norm(lhs) + np.linalg.norm(rhs))
        # Observed ≤ 2.1e-14 across these chains; 1e-11 keeps ~10³× margin for
        # cross-platform LAPACK variation while still asserting near-machine
        # precision (not the loose 1e-9 the first draft enforced).
        assert rel < 1e-11


@pytest.mark.parametrize("masses_amu", [[25.0, 44.0], [9.0, 40.0], [40.0, 9.0], [23.0, 87.0], [6.0, 138.0]])
def test_two_ion_axial_modes_are_mass_independent(masses_amu: list[float]) -> None:
    """The two axial modes of a 2-ion chain are ω_z and √3·ω_z for ANY mass ratio.

    Counter-intuitive but correct in this model: the trap stiffness is m·ω² (a
    shared secular *frequency*, not a shared curvature), so the relative-mode
    stiffness 3ω_z²μ_r over reduced mass μ_r gives √3·ω_z independent of mass.
    Locks the finding recorded in docs/LOG.md so it can't silently regress.
    (Mass dependence in the axial spectrum first appears at N ≥ 3.)
    """
    _, modes = _modes_for(masses_amu)
    # Strong radial confinement (8/9 MHz) ≫ axial (1 MHz) → the two lowest modes
    # are the axial COM and stretch.
    axial = np.sort(modes.frequencies_rad_s)[:2]
    assert axial[0] == pytest.approx(TRAP.wz, rel=1e-8)
    assert axial[1] == pytest.approx(np.sqrt(3.0) * TRAP.wz, rel=1e-8)


def test_physical_displacements_M_orthonormal_not_euclidean() -> None:
    """Negative control for the symmetric-D design choice.

    The §11 eigenvectors b are Euclidean-orthonormal (asserted elsewhere). The
    *physical* displacements x = M^{-1/2} b are M-orthonormal exactly, but for
    mixed species they are clearly **not** Euclidean-orthonormal — which is why
    ``modes.py`` diagonalises the symmetric D and exports b, not x.
    """
    eq, modes = _modes_for([25.0, 44.0, 25.0])
    nm = modes.n_modes
    m3 = np.repeat(eq.masses, 3)
    b = modes.eigenvectors.reshape(nm, -1)
    x = b / np.sqrt(m3)[None, :]

    # x is M-orthonormal: xᵀ M x = I exactly (M diagonal → elementwise scaling).
    assert np.allclose((x * m3[None, :]) @ x.T, np.eye(nm), atol=1e-8)

    # ...but the unit-normalised x are not mutually orthogonal for mixed species.
    xn = x / np.linalg.norm(x, axis=1, keepdims=True)
    off_diag = xn @ xn.T - np.eye(nm)
    assert np.max(np.abs(off_diag)) > 1e-3
