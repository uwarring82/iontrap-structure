# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
semantic versioning once it leaves pre-alpha.

## [Unreleased]

Validation/hardening of the v0.1 slice, a full documentation site, and the
**GT3b producer payload** (the first cross-repo *dynamics* handshake). No physics
changes to the existing engine (one defensive guard in `normal_modes`, below).
See [`docs/LOG.md`](docs/LOG.md) for the decision/findings trail.

### Documentation

- **MkDocs + Material site** mirroring the sibling `iontrap-dynamics`: welcome,
  overview (the producer/consumer boundary), getting-started, conventions
  (§10/§11), validation (oracle catalogue), provenance, and an `mkdocstrings`
  API reference. Branding (`tokens.css`/`extra.css`) is shared with the sibling.
- **Five runnable tutorials**, each generated into a Colab notebook and executed
  end-to-end in CI (asserts are the oracle): (1) first crystal, (2) the
  `ModeConfig` handoff to `iontrap-dynamics`, (3) mixed-species crystals,
  (4) the plasma coupling parameter Γ, (5) the linear→zigzag transition.
- `tools/build_tutorial_notebooks.py` (Markdown → Colab notebooks, `--check`
  freshness guard) and `tests/docs/test_tutorials_execute.py` (execute every
  tutorial). New CI jobs `docs` (strict build), `tutorials`, `notebooks`, and a
  `docs-deploy.yml` GitHub Pages workflow.
- New `[docs]` / `[plot]` extras, a `tutorial` pytest marker, and a
  `Documentation` project URL. Docs are dual-licensed (`docs/LICENCE`): Coastline
  CC BY-SA 4.0, Sail tutorials CC BY-NC-SA 4.0; code stays MIT.

### Added

- **`IonModeBasis` payload + `ion_mode_basis(...)` emitter** — the producer half
  of the ratified cross-repo GT3b handshake with `iontrap-dynamics`
  (`taskcards/TC-ion-mode-basis-payload.md`; consumer contract `iontrap-dynamics`
  `TC-gt3b-ion-symplectic-adapter.md` v0.3). A versioned, **serialization-neutral**
  payload (plain arrays + tags via `asdict()`/`as_arrays()`, no shared runtime
  class) carrying the flattened mass-symmetrised basis `B` `(3N, 3N)`, the mode
  frequencies `ω_m`, the masses, and a **tagged local-reference gauge**
  `local_reference_frequencies_rad_s` (D3: default `"trap_curvature"`, optional
  `"diagonal_hessian"`). Wire invariants are fixed explicitly: `coordinate_frame`
  is the consumer's canonical identifier (`ion-major-axis-minor;row=axes_per_ion*i+c;col=mode`,
  compared by exact match on the consumer side), with handedness / axis labels /
  frequency↔column alignment / mode ordering carried in the tags, and the
  eigenvector-sign gauge canonicalised (largest-|component| positive) so re-runs on
  a given `ModeResult` reproduce the payload exactly. A positivity/finiteness guard
  rejects zero/soft or non-finite frequencies at the source. This repo stays
  classical structural dynamics — the `S = diag(X, P)` symplectic-map math stays
  in `iontrap-dynamics`. Contract-tested in `tests/test_ion_mode_basis_contract.py`
  (shapes, `ω > 0`, flattened Gram `= I`, ordering round-trip, both local-reference
  gauges, reproducibility). The legacy `to_mode_configs()` per-mode seam is
  unchanged and is **not** the GT3b handshake.
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
