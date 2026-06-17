# Decision & findings log

A dated, chronological lab-notebook for `iontrap-structure`: what was
investigated, what was found (including numbers and negative results), and which
decisions were taken and why. This complements — and does not replace — the
user-facing [`CHANGELOG.md`](../CHANGELOG.md) (what changed for users) and the
frozen [`docs/PROVENANCE.md`](PROVENANCE.md) (origin/clean-room provenance).

Kept in the spirit of FAIR principles and reproducible scientific practice:
decisions should be auditable from this trail, not reverse-engineered from diffs.

Newest entries first.

---

## 2026-06-17 — Documentation site + reproducible tutorials (mirroring the sibling)

**Context.** With the v0.1 foundation validated, stood up a proper documentation
site. Decision: mirror the sibling `iontrap-dynamics` docs stack 1:1 rather than
invent a new one — same tooling, structure, branding, and reproducibility
guarantees — so the sibling pair is consistent. (Recon was straightforward: a
local clone of the sibling exists alongside this repo.)

**What was built.**

- **MkDocs + Material + mkdocstrings.** `mkdocs.yml` mirrors the sibling (custom
  palette, `tokens.css`/`extra.css` copied verbatim for visual consistency — see
  `docs/LICENCE` for provenance). Added an `mkdocstrings` API-reference page
  (`api.md`), a small enhancement over the sibling (which ships the dep but no
  API page yet). Narrative pages: index (hero), overview (the producer/consumer
  boundary), getting-started, conventions (§10/§11), validation (the oracle
  catalogue). `PROVENANCE.md` is on the site; the internal `LOG.md` is
  `exclude_docs`-ed (a dev artifact, like the sibling's off-site WP/LOGBOOK).
- **Reproducible tutorials — the FAIR core.** Markdown is the single source of
  truth; `tools/build_tutorial_notebooks.py` (adapted from the sibling) generates
  Colab notebooks, with a `--check` freshness guard.
  `tests/docs/test_tutorials_execute.py` executes every tutorial end-to-end with
  its embedded `assert`s as the oracle (marked `tutorial`, needs `[plot]`).
- **Five tutorials**, drafted-and-self-verified in parallel by a Workflow (each
  agent ran the actual pytest harness on its page until green), then manually
  reviewed: (1) first crystal — James-1998 positions, ω_z COM, √3 ω_z stretch to
  machine precision; (2) the ModeConfig handoff (runs with or without
  iontrap-dynamics via the fallback); (3) mixed-species — the symmetric-D vs
  M⁻¹H contrast, generalized-eigenproblem residual ~8e-15; (4) Γ — exact 1/T
  scaling, Γ(1 mK)≈3728 ≫ 170, crossover ~22 mK; (5) linear→zigzag — soft mode,
  critical aspect ratio ω_r/ω_z ≈ 2.5 for N=5, and the stability guard firing.
- **CI/CD.** Added `docs` (strict build), `tutorials` (execute), `notebooks`
  (freshness) jobs to `ci.yml` (matrix deselects `tutorial`), and a
  `docs-deploy.yml` Pages workflow. New `[docs]`/`[plot]` extras, a `tutorial`
  marker, and `--strict-markers`.

**Findings / decisions worth recording.**

- The Workflow's first launch failed entirely on transient API 529s (0 tokens);
  a clean retry produced all five. Recorded as a reminder that workflow agents
  inherit upstream availability — retry is the right response, not a redesign.
- Ruff lints `.ipynb`, so the *generated* notebooks tripped style rules
  (physics-notation locals, etc.). Excluded `docs/tutorials/notebooks` from ruff
  (mirroring the sibling): the notebooks are a derived artifact validated by
  *execution*, not lint. Separately fixed one genuine wart — an unused `k_B`
  import in tutorial 4's source.
- `mkdocs build --strict` swept the pre-existing `docs/LOG.md` and
  `docs/PROVENANCE.md` into the build; resolved by putting Provenance in the nav
  and `exclude_docs`-ing the lab-notebook (whose `../CHANGELOG.md` link does not
  resolve inside the site tree).
- Normalised two cosmetic inconsistencies the (529-failed) review agent would
  have caught: tutorial 5's Next-steps link format and tutorial 3's numbered
  link text now match the dominant `**[Title](file.md)** — …` style.

**Outcome.** `ruff` clean, `mypy` clean, `mkdocs build --strict` clean, notebook
freshness clean, `pytest` 58 passed / 1 skipped (53 unit + 5 tutorials; the skip
is the optional iontrap-dynamics interop export). No package source or public-API
change — docs/tooling/tests only.

---

## 2026-06-17 — Review follow-ups (packaging gate, stability guard, audit-trail fix)

**Context.** Acting on external review of the hardening pass below. Six points,
all addressed; still validation/tooling + one defensive code change, no public-API
or physics change.

1. **CI could hide editable-install breakage.** Because `pytest` now injects
   `src/` (`tool.pytest.ini_options.pythonpath`), a broken editable install would
   still give green tests. Added (a) an **import smoke step** that imports the
   *installed* package from `$RUNNER_TEMP` (no `src/` on the path) in the test
   job, and (b) a separate **`package` job** that builds the wheel, installs it
   into a clean venv, and runs an import + minimal-computation smoke. Validated
   the wheel path locally: it ships `iontrap_structure/py.typed` and imports +
   computes correctly from a clean venv.

2. **Audit-trail inconsistency (residual numbers).** The entry below first cited
   a raw absolute residual ~3.4e-13 (an early exploratory `max|Hx−ω²Mx|`, before
   normalisation) and later a *relative* residual ≤2.1e-14; the test enforced
   only `< 1e-9`. These measure different things. Reconciled: the enforced
   property is now the **relative** residual `< 1e-11` in
   `test_mixed_species.py` (observed ≤2.1e-14 across all chains → ~10³× margin
   for cross-platform LAPACK variation); the raw-vs-relative distinction is
   called out so the numbers below are no longer presented as one quantity.

3. **`modes.py` silently clipped negative eigenvalues.** `np.sqrt(np.clip(ω², 0,
   None))` would turn a genuinely negative eigenvalue into a spurious real
   frequency, masking an unstable equilibrium or a Hessian bug. Replaced with a
   spectrum-scaled guard: tiny round-off negatives (≈ε·‖D‖) are still clipped,
   but `min(ω²) < −1e-9·max|ω²|` now raises `ValueError`. Confirmed it fires on a
   real saddle (weak-radial trap, on-axis 3-ion chain past the zigzag
   instability: min eigenvalue ≈ −9.1e13 rad²·s⁻²). For that config the two
   relevant scales are the round-off floor ε·‖D‖ ≈ 0.05 rad²·s⁻² and the guard
   trigger 1e-9·max|ω²| ≈ 2.3e5 rad²·s⁻²; the saddle eigenvalue is ~18 and ~8
   orders of magnitude beyond them, so it fires decisively — and never on a valid
   confining trap (where the smallest eigenvalue is ω_z² ≫ the trigger).

4. **2-ion mass-independence finding promoted to a regression test.** The
   counter-intuitive result (axial modes = ω_z, √3·ω_z for any mass ratio) was
   only in this log; now parametrised over ratios up to 6/138 in
   `test_mixed_species.py`.

5. **Minor (clarity):** documented the amu-vs-kg scale-invariance in the
   COM-frequency oracle test (the eigenvector direction is invariant under a
   common positive mass rescaling).

6. **Parent-`ModeConfig` branch coverage.** Added a deterministic test that
   exercises the `iontrap_dynamics` export branch via an injected stand-in
   module (`monkeypatch.setitem` on `sys.modules`), so the previously-uncovered
   line is hit without the optional dependency — replacing the need for an
   `[interop]` CI lane just for coverage.

**Outcome.** `ruff`/`mypy` clean, `pytest` 53 passed / 1 skipped, **coverage
100%** (the previously-missed parent-`ModeConfig` line is now covered; the lone
skip is still the *real*-dependency interop test). The local wheel build +
clean-venv smoke passes.

---

## 2026-06-17 — Validation/hardening pass on the v0.1 slice

**Context.** First working session after the v0.1.0 bootstrap. Goal: deepen and
validate the existing slice before adding new physics (chosen over the large-N
eigensolve spike and a second abstraction use case, which remain open in
`PROVENANCE.md`).

**Baseline verified.** Default `python` here is 3.9, but the project needs ≥3.11;
set up a gitignored `.venv` on Python 3.13.7 (numpy 2.4.4, scipy 1.17.1).
`ruff check .` clean; `pytest -q` → 17 passed, 1 skipped (the skip is the optional
`iontrap_dynamics` interop export test — expected, parent not installed). The
README quick-start runs verbatim and gives physically sensible output
(axial modes 1.0, √3, 2.41 MHz; radial COM at exactly 10 MHz; Γ ≈ 2982 at 1 mK).

**Findings.**

1. **mypy is configured but not green, and not in CI.** `mypy` reports two unused
   `type: ignore` comments in `results.py` (the conditional `iontrap_dynamics`
   import): with `ignore_missing_imports = true` the `[import-not-found]` and
   `[assignment]` ignores are redundant and `warn_unused_ignores = true` flags
   them. CI only ran ruff + pytest. → Decision: fix the typing of the conditional
   import and add a mypy step to CI, so the type gate is actually enforced.

2. **Radial COM modes are an untested universal oracle.** With ω_x ≠ ω_y ≠ ω_z,
   single-species chains show modes at *exactly* ω_x, ω_y, ω_z (verified N=3,5).
   Physics: uniform translation along an axis is Coulomb-invariant, so the trap
   alone sets its frequency — true for any N. Only the axial ω_z (and √3 ω_z)
   were previously asserted. → Decision: add a radial/3-axis COM-mode oracle.

3. **Mixed-species path was completely untested**, despite `modes.py` making a
   pointed claim that diagonalising the *symmetric* D (not M⁻¹H) is what yields
   §11-orthonormal eigenvectors "for mixed species." Investigated what oracles
   exist:
   - A candidate closed form for 2-ion mixed-species axial frequencies was
     **falsified by the numerics**: in *this* model the trap stiffness is m·ω²
     (a shared secular *frequency*, mass-independent), so the 2-ion axial modes
     are ω_z and √3 ω_z **regardless of mass ratio** (checked 25/25, 25/44,
     40/9). Derivation confirms it (relative-coordinate stiffness 3ω²μ_r over
     reduced mass μ_r → √3 ω). So 2-ion axial freqs do **not** discriminate
     species here; mass-dependence first appears at N ≥ 3.
   - Better mixed-species oracle: the translational/COM modes sit at exactly ω_c
     for *any* masses (same argument as #2; the §11 eigenvector is ∝ √m_i, but
     the frequency is exactly ω_c). This is a real analytic oracle for mixed
     species.
   - Strongest check is oracle-free: the reported (ω_m, b_m) must satisfy the
     generalized eigenproblem H x = ω² M x with x = M⁻¹ᐟ² b. This first
     exploratory probe used the *raw absolute* residual max‖Hx − ω²Mx‖ ≈ 3.4e-13
     for a `[25,44,25]` chain; the test instead enforces the dimensionless
     *relative* residual (see the 2026-06-17 review-follow-ups entry above —
     observed ≤2.1e-14, enforced < 1e-11). → Decision: validate the
     mixed-species path via (a) the ω_c COM-frequency oracle, (b) the generalized
     eigenproblem residual, (c) §11 contract preservation, and (d) a negative
     control showing the naive M⁻¹H eigenvectors are *not* Euclidean-orthonormal.

4. **Analytic derivatives only partially FD-checked.** `test_hessian.py`
   finite-differences the *Coulomb* Hessian only. The trap Hessian, the total
   Hessian that `normal_modes` actually consumes, and the energy→gradient
   relationship are unverified. → Decision: add energy→gradient and total-Hessian
   finite-difference checks.

**Scope decision.** This pass is validation/test-hardening + CI/typing only — no
changes to the physics or public API. New physics (zigzag transition, Penning
shells) stays out until the foundation is locked.

**Surprise (worth recording): the editable install was not honoured across
processes.** Hatchling's editable install writes a path-config
`_editable_impl_iontrap_structure.pth` pointing at `src/`, but in this local
sandbox the `site` module did not append it to `sys.path` for fresh interpreters
(`pip install -e` then `import` only worked within the *same* shell command);
`pytest` collection then failed too. Rather than depend on that, made resolution
explicit and install-independent: `mypy_path = "src"` for mypy and
`tool.pytest.ini_options.pythonpath = ["src"]` for pytest. This also makes CI more
robust. (Likely a local quirk; CI on clean Ubuntu would probably honour the
`.pth` — but not relying on it is strictly better.)

**Outcome.**

- `ruff` clean, `mypy` green (now also enforced in CI), `pytest` 46 passed,
  1 skipped (the skip is the optional `iontrap_dynamics` interop export).
- Test count 17 → 46. Public-API line coverage 88% → 99% (the single remaining
  uncovered line is the parent-`ModeConfig` branch, exercised only when the
  optional `[interop]` package is installed — i.e. by the skipped test).
- New/extended tests: mixed-species mode validation (`test_mixed_species.py`),
  3-axis COM oracle (`test_james1998.py`), energy→gradient and total-Hessian FD
  (`test_hessian.py`), Γ/NN diagnostics + guards (`test_diagnostics.py`),
  constructor/solver guards + `trap.energy` + custom initial guess
  (`test_validation.py`), and `to_mode_configs(labels=…)` / `n_ions`
  (`test_modeconfig_contract.py`).

**Numbers captured (for reproducibility).** Generalized-eigenproblem relative
residual ≤ 2.1e-14 across all mixed chains (the *enforced* assertion is < 1e-11,
see the review-follow-ups entry); COM-mode overlap = 1.0 and
ω_mode/ω_c = 1.0 to ≤ 1e-9; physical-displacement M-orthonormality |xᵀMx − I|
≈ 7e-16; Euclidean off-diagonal for the mixed `[25,44,25]` chain ≈ 0.27 (the
negative control); FD derivative agreement ~1e-10.

**Still open / candidate next steps** (not done here): a doctest/smoke test of the
README quick-start to stop doc-rot; and the larger backlog items in
`PROVENANCE.md` (large-N eigensolve spike; second abstraction use case;
neutral-background goal). *(The parent-`ModeConfig` branch — previously flagged
as needing an `[interop]` CI lane — is now covered deterministically via an
injected stand-in module; a real `[interop]` lane is no longer required just for
coverage, though it would still be a useful integration check.)*
