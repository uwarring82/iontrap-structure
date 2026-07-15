# TC — `IonModeBasis` payload emitter (GT3b producer side)

Version 0.1 · Drafted 2026-07-15 · **Status: Implemented** · Producer half of the ratified cross-repo handshake

> **Authoritative contract:** `iontrap-dynamics` `task cards/TC-gt3b-ion-symplectic-adapter.md` (ratified v0.3, commit `1a88145`). This card is the **`iontrap-structure` (producer) obligations only** — self-contained so implementation here needs no cross-repo reading. `iontrap-structure` owns the **physical correctness** of the exported fields; `iontrap-dynamics` owns the **schema version + consumer contract**.

## Goal

Emit a versioned, **serialization-neutral** `IonModeBasis` payload (plain arrays + metadata — **no shared runtime class**, so neither repo type-couples to the other) that the `iontrap-dynamics` GT3b adapter validates and materializes into its own record to build the normal→local symplectic map. Two deliverables:

1. **`IonModeBasis` payload** built from an existing `ModeResult` (+ its `HarmonicTrap`).
2. A new **`local_reference_frequencies_rad_s`** field (see D3 below).

The map math (`S = diag(X,P)`, §27 quadratures, `X Pᵀ = I`) lives entirely in `iontrap-dynamics` — **not here**. This repo stays classical structural dynamics.

## Payload schema

| Field | Shape / units | Source in this repo |
|---|---|---|
| `schema_version` | int/str | fixed by the consumer contract; emit the agreed value, mismatch = hard error |
| `frequencies_rad_s` | `(3N,)` rad·s⁻¹ | `ModeResult.frequencies_rad_s` (the `ω_m`), all `> 0` |
| `mass_weighted_eigenvectors` | `(3N, 3N)` | `B[:, m] = ModeResult.eigenvectors[m].reshape(3N)` (see ordering below); orthogonal `B` (columns orthonormal, Gram `= I`) |
| `masses_kg` | `(N,)` | `ModeResult.masses` |
| `local_reference_frequencies_rad_s` | `(3N,)` | new — see **D3** |
| `coordinate_frame` | tag | the ordering/handedness statement below |
| `normalization_weighting_tags` | tags | e.g. `mass_symmetrised=True`, `per_mode_unit_norm=True` |

`N = ModeResult.n_ions`, `3N = ModeResult.n_modes`.

## Wire invariants (fix these explicitly — producer and consumer must agree bit-for-bit)

- **Coordinate ordering (row index of `B`):** ion-major, axis-minor — row `r = 3·i + c` for ion `i ∈ [0,N)`, axis `c ∈ {0:x, 1:y, 2:z}`. This is exactly `ModeResult.eigenvectors[m].reshape(3N)` (row-major), so `B[3i+c, m] = eigenvectors[m, i, c]`.
- **Trap frame + handedness:** the right-handed `(x, y, z)` of `HarmonicTrap` (`wx, wy, wz`); state it in `coordinate_frame`.
- **Frequency↔column alignment:** column `m` of `B` corresponds to `frequencies_rad_s[m]` — same index `m`.
- **Mode ordering:** as returned by the eigensolver (ascending `ω²`); state it.
- **Residual freedoms:** the **eigenvector sign** (per column) and any **degenerate-subspace rotation** are gauge freedoms of the diagonalization. Either canonicalize them (e.g. fix the sign by the largest-|component| convention; document the degenerate-subspace choice) or tag that they are unpinned — so a re-run reproduces the same payload. The consumer needs this to be deterministic.

## D3 — `local_reference_frequencies_rad_s` (a tagged local-coordinate gauge)

This is the per-**coordinate** reference frequency `ω_local,j` (one per `3N` local coordinate, same row ordering as `B`). On the consumer side it only sets a **local single-mode squeezing gauge**: it changes the covariance representation, occupation, and `T_eff`, but **not** ion-cut entanglement (`E_N` is invariant). So the choice is not physically forced — just record which convention you emit, tagged. Candidates:

- **(i) bare single-ion trap curvature (recommended default):** `local_reference_frequencies_rad_s[3i+c] = HarmonicTrap.omega[c]` — i.e. the secular frequency of axis `c` for every ion. Directly available from the trap; no Coulomb.
- **(ii) diagonal local Hessian** `√(H_{jj}/m_j)` at equilibrium (includes Coulomb curvature at the ion positions) — `H = HarmonicTrap.hessian(masses) + coulomb.hessian(positions, charges)`, take `√(diag(H)[j] / m_{ion(j)})`.
- **(iii) another canonical reference.**

Emit **(i)** as the default and put the choice in a tag; the canonical default can be revisited without affecting `E_N`.

## Producer obligations (what this repo guarantees)

- Stable-equilibrium modes only (already: `normal_modes` raises on non-positive `ω²`).
- `B` orthogonal (columns unit-norm, cross-mode Gram `= I`) — already contract-tested in `tests/test_modeconfig_contract.py`; extend the contract test to the flattened `(3N,3N)` `B` and to the new field.
- Field **correctness** and the documented conventions/tags. (The consumer owns `schema_version` + shape/ordering validation.)

## Non-goals (kept in `iontrap-dynamics`)

- The §27 phase-space convention, the `√(2ω/ℏ)` quadrature scaling, the `S = diag(X,P)` builder, and any covariance/log-negativity math. This repo emits **only** the classical mode basis + trap frequencies as arrays.
- The legacy `ModeResult.to_mode_configs()` / `ModeConfig` per-mode seam stays as-is for existing per-mode consumers; it is **not** the GT3b handshake.

## Suggested shape (implementer's discretion)

A `frozen`/`slots` record + a builder, matching the repo's "pure functions over frozen records" style — e.g. `IonModeBasis` (the payload) and `ion_mode_basis(modes: ModeResult, trap: HarmonicTrap, *, local_reference="trap_curvature") -> IonModeBasis`, plus an `as_arrays()`/`asdict()` for the serialization-neutral wire form. A contract test mirroring `test_modeconfig_contract.py` (shapes, `ω>0`, Gram `= I`, ordering round-trip, `local_reference` values) locks the producer half.
