# Validation

Every physics claim in `iontrap-structure` is pinned by a test with an
independent oracle — an analytic result, a contract the output must satisfy, or
a finite-difference cross-check. This page catalogues those oracles; the tests
themselves live under [`tests/`](https://github.com/uwarring82/iontrap-structure/tree/main/tests)
and run on every push (`pytest`, plus a separate job that executes the
tutorials end-to-end).

## James-1998 linear chain

D. F. V. James, *Appl. Phys. B* **66**, 181 (1998) gives closed-form results
for the single-species linear chain that the solver must reproduce.

| Oracle | Value | Test |
|---|---|---|
| 2-ion equilibrium half-spacing | `u = (1/4)^{1/3} ℓ` | `test_james1998.py` |
| 3-ion outer-ion position | `u = (5/4)^{1/3} ℓ` | `test_james1998.py` |
| Axial centre-of-mass mode | `ω = ω_z` (any N) | `test_james1998.py` |
| Axial stretch mode | `ω = √3 ω_z` (any N) | `test_james1998.py` |
| Radial / 3-axis COM modes | `ω = ω_x, ω_y, ω_z` exactly | `test_james1998.py` |

The centre-of-mass modes sit at exactly the bare trap frequencies because a
uniform translation leaves every pair separation unchanged, so the Coulomb
interaction contributes no restoring force — true for any `N` and any masses
(see the mixed-species oracle below).

## The §11 mode contract

Independently of the physics, the exported modes must be a valid `ModeConfig`
payload (`tests/test_modeconfig_contract.py`):

- per-mode normalisation `Σᵢ ‖b_{i,m}‖² = 1`,
- cross-mode orthonormality (the Gram matrix is the identity),
- all frequencies real and positive for a confining trap,
- a clean round-trip through `to_mode_configs()`.

## Mixed-species correctness

For unequal masses there is no simple closed form, so the modes are validated
three ways (`tests/test_mixed_species.py`):

- **COM frequency oracle** — a mode sits at exactly `ω_c` for any masses.
- **Generalized eigenproblem residual** — the reported `(ω_m, b_m)` satisfy
  `H x = ω² M x` with `x = M⁻¹ᐟ² b`; the relative residual is enforced below
  `1e-11` (observed ≤ 2.1e-14).
- **Negative control** — the physical displacements are M-orthonormal but *not*
  Euclidean-orthonormal, which is precisely why the symmetric `D` is
  diagonalised rather than `M⁻¹H`.

A regression test also pins the (counter-intuitive) result that the 2-ion axial
modes are `ω_z` and `√3 ω_z` independent of the mass ratio, in this model where
the trap stiffness is `m·ω²`.

## Analytic derivatives

The analytic gradient and Hessian are finite-difference checked
(`tests/test_hessian.py`): the Coulomb energy→gradient relation, the Coulomb
Hessian, and the full trap+Coulomb Hessian — the exact matrix `normal_modes`
diagonalises — all agree with central differences to ~1e-10.

## Stability guard

`normal_modes` raises `ValueError` when the dynamical matrix has a materially
negative eigenvalue (an unstable / saddle configuration), rather than silently
clipping `ω² < 0` to a real frequency. Round-off-level negatives for genuine
zero/soft modes are still clipped. The guard is exercised on a real saddle — an
on-axis chain past the linear→zigzag instability (`tests/test_validation.py`,
and [Tutorial 5](tutorials/05_zigzag_stability.md)).

## Reproducibility

The tutorials are part of the validation surface: every `python` block in every
tutorial is executed end-to-end in CI, with its embedded `assert` statements as
the oracle (`tests/docs/test_tutorials_execute.py`). A tutorial whose physics
regressed fails the build.

## References

- D. F. V. James, *Appl. Phys. B* **66**, 181 (1998) — linear-chain equilibrium and modes.
- D. H. E. Dubin, T. M. O'Neil, *Rev. Mod. Phys.* **71**, 87 (1999) — non-neutral plasma, Γ.
- S. Fishman, G. De Chiara, T. Calarco, G. Morigi, *PRB* **77**, 064111 (2008) — linear→zigzag.

## Endorsement Marker

Local candidate framework under active stewardship. No external endorsement is
implied.
