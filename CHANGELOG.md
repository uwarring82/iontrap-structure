# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
semantic versioning once it leaves pre-alpha.

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
