# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
semantic versioning once it leaves pre-alpha.

## [Unreleased]

Validation/hardening pass on the v0.1 slice — no public-API or physics changes
(one defensive guard in `normal_modes`, below). See [`docs/LOG.md`](docs/LOG.md)
for the decision/findings trail.

### Added

- Mixed-species normal-mode validation (`tests/test_mixed_species.py`): the §11
  contract for unequal masses, the centre-of-mass frequency oracle (ω_c is a
  mode for any masses), an oracle-free generalized-eigenproblem residual
  `H x = ω² M x` (enforced relative residual < 1e-11), a negative control showing
  the physical displacements are M-orthonormal but not Euclidean-orthonormal (the
  reason the symmetric `D` is diagonalised), and a regression test pinning the
  mass-independence of the 2-ion axial modes (ω_z, √3·ω_z for any mass ratio).
- Centre-of-mass mode oracle for the three trap axes (single species).
- Finite-difference checks closing the derivative chain: Coulomb energy→gradient
  and the full trap+Coulomb Hessian (the matrix `normal_modes` diagonalises).
- Diagnostics tests (Γ definition, NN spacing, input guards) and constructor/
  solver input-guard tests; public-API line coverage now 100%.
- A `normal_modes` stability test exercising the new guard (an unstable on-axis
  chain under weak radial confinement), and a deterministic test of the
  parent-`ModeConfig` export branch via an injected stand-in module.

### Changed

- CI now runs `mypy` (type-check), an import smoke step against the *installed*
  package (so a broken editable install can't hide behind src-injected pytest),
  and a separate `package` job that builds the wheel and smoke-tests it in a
  clean environment.
- `pytest` resolves the package from `src/` (`tool.pytest.ini_options.pythonpath`)
  and `mypy` from `src/` (`mypy_path`), so both run reproducibly regardless of
  how the editable install places the package on `sys.path`.
- `normal_modes` now raises `ValueError` when the dynamical matrix has a
  materially negative eigenvalue (an unstable/saddle configuration) instead of
  silently clipping ω² < 0 to a real frequency. Round-off-level negatives for
  genuine zero/soft modes are still clipped to zero.

### Fixed

- `mypy` is green: removed two redundant `# type: ignore` comments on the
  conditional `iontrap_dynamics` import in `results.py`.

## [0.1.0] — 2026-06-17

First slice, bootstrapped from the frozen origin task card in `iontrap-dynamics`
(`TC-structural-dynamics-foundation-survey.md`, v1.0).

### Added

- `HarmonicTrap` — anisotropic harmonic (pseudopotential) trap model.
- `coulomb` — pairwise Coulomb energy, gradient, and analytic Hessian
  (per-ion-contiguous layout; `k_e` overridable for dimensionless use).
- `equilibrium` — force-balance solver in dimensionless units with analytic
  Jacobian; returns a frozen `EquilibriumResult`.
- `normal_modes` — eigenproblem on the mass-symmetrised dynamical matrix
  `M⁻¹ᐟ² H M⁻¹ᐟ²`; returns a frozen `ModeResult` with `to_mode_configs()`
  exporting `ModeConfig`-compatible records (CONVENTIONS §10/§11).
- `diagnostics` — plasma coupling parameter Γ and nearest-neighbour spacing
  (decoupled analysis layer; `freud` optional).
- Tests: James-1998 oracle (positions; axial ω_z and √3 ω_z modes), the
  `ModeConfig` contract, and a finite-difference Hessian check.
