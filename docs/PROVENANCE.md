# Provenance & carried-forward backlog

This repository was bootstrapped on **2026-06-17** from a frozen deliberation
and survey record in `iontrap-dynamics`:

> `task cards/TC-structural-dynamics-foundation-survey.md` (v1.0, frozen)
> — https://github.com/uwarring82/iontrap-dynamics

That card surveyed 32 existing GitHub/web packages (29 ranked) for classical
large-N ion structural-dynamics code and concluded the niche is unfilled:
**build a fresh MIT sibling that produces `ModeConfig` output, clean-room by
default**, using surveyed solvers (oqd-trical, trimos, levitated-cooling) only as
validation references — never as code seeds.

## Resolved at freeze

- Classical / quantum split: this package produces mode structure;
  `iontrap-dynamics` consumes it (its Design Principle 2 / `modes.py`).
- Clean-room / self-owned by default (no copyleft or unlicensed source).
- Abstraction kept as an internal *seam* behind concrete ion-trap presets
  (pure functions over frozen records — no public DSL, no fluent framework).
- Neutral-atom background deferred to a force/energy seam (no collision API yet).
- Host/name: `uwarring82/iontrap-structure`.

## Opening backlog (carried forward)

**Done in the first slice (v0.1.0):** linear/harmonic equilibrium + modes →
`ModeConfig` export; James-1998 validation; the decoupled Γ analysis layer.

**Still open — to resolve here:**

1. **Large-N eigensolve spike (first hard research risk).** FMM accelerates
   equilibrium-finding but not diagonalising a dense 3N×3N Hessian; at
   N ≳ 10³ that is the wall. Spike a sparse/iterative or structure-exploiting
   eigensolver *before* committing scope.
2. **Second abstraction use case** — Penning shell modes, dusty plasmas, or
   another confined-particle system — to pressure-test the kernel seam before
   freezing it. The transition/shell physics is clean-room from
   Wang–Keith–Freericks (2013) and Dubin–O'Neil (1999).
3. **Neutral-background goal** — sympathetic cooling/thermalisation (a bath
   tier) vs reactive collisions/chemistry — before designing any collision API.
   Note micromotion-interruption heating needs full time-dependent RF dynamics,
   not the static pseudopotential used here.

**Optional, non-blocking:** relicensing outreach to the authors of
`WesLeeJohnson/mode_analysis` and `jzaris/coldatoms_051623` (the only surveyed
code that actually models the crystal→plasma / shell regime, but currently
unlicensed / GPL) for validation-reference use.

## References (clean-room sources)

- D. F. V. James, Appl. Phys. B **66**, 181 (1998) — linear-chain equilibrium + modes.
- C.-C. J. Wang, A. C. Keith, J. K. Freericks, PRA **87**, 013422 (2013) — Penning normal modes.
- D. H. E. Dubin, T. M. O'Neil, Rev. Mod. Phys. **71**, 87 (1999) — non-neutral plasma / shells / Γ.
- S. Fishman, G. De Chiara, T. Calarco, G. Morigi, PRB **77**, 064111 (2008) — linear→zigzag transition.
