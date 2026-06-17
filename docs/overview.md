# Overview

`iontrap-structure` answers one question: **given a trap and a set of ions,
where do they sit, and how do they vibrate?** It then hands the answer to
[`iontrap-dynamics`](https://uwarring82.github.io/iontrap-dynamics/) in the
shape that package expects.

## The producer / consumer boundary

```
        iontrap-structure                         iontrap-dynamics
   ┌──────────────────────────┐             ┌──────────────────────────┐
   │  HarmonicTrap, masses,    │             │  consumes externally-     │
   │  charges                  │             │  supplied normal modes    │
   │        │                  │             │  (Design Principle 2:     │
   │        ▼                  │  ModeConfig │  modes.py does NOT derive │
   │  equilibrium()  ──► positions  ───────► │  mode structure itself)   │
   │        │                  │   records   │        │                  │
   │        ▼                  │             │        ▼                  │
   │  normal_modes() ──► (ω, b) │             │  Hamiltonians, dynamics, │
   │        │                  │             │  measurement, ...         │
   │        ▼                  │             │                           │
   │  to_mode_configs()        │             │                           │
   └──────────────────────────┘             └──────────────────────────┘
```

The two packages are deliberately separate. `iontrap-dynamics` is about the
*quantum* evolution of spin and motion; deriving the classical mode structure
is a different problem with different numerics (root-finding, dense
eigensolves, eventually fast multipole methods). Keeping them apart means
neither grows a dependency it does not need, and the `ModeConfig` contract is
the only thing that crosses the boundary.

## The physics, briefly

For `N` ions of mass `mᵢ` and charge `qᵢ` in an anisotropic harmonic trap, the
potential energy is

```
E(r) = Σᵢ ½ mᵢ (ω_x² xᵢ² + ω_y² yᵢ² + ω_z² zᵢ²)  +  Σ_{i<j} k_e qᵢ qⱼ / |rᵢ − rⱼ|
```

- **Equilibrium** is the configuration where `∂E/∂r = 0`. `equilibrium()`
  solves this force balance. To keep the problem well-conditioned it works in
  dimensionless units (length scale `ℓ = (k_e q_ref² / (m_ref ω_z²))^{1/3}`)
  and supplies the analytic Jacobian (the scaled Hessian), so the root find
  reaches near machine precision.

- **Normal modes** are the eigenvectors of the Hessian about that equilibrium.
  Rather than the non-symmetric `M⁻¹H`, `normal_modes()` diagonalises the
  **mass-symmetrised dynamical matrix** `D = M⁻¹ᐟ² H M⁻¹ᐟ²`. `D` is symmetric,
  so its eigenvectors are Euclidean-orthonormal — which is exactly the
  CONVENTIONS §11 normalisation `Σᵢ ‖bᵢ‖² = 1`, and the reason mixed-species
  crystals come out right (the two routes coincide only for equal masses).

## Design stance

- **Pure functions over frozen records.** `EquilibriumResult`, `ModeResult`,
  and `StructuralDiagnostics` are immutable dataclasses. There is no stateful
  engine and no public DSL — abstraction is kept as an internal seam behind
  concrete entry points.
- **SI in, SI out.** Positions in metres, masses in kilograms, charges in
  coulombs, angular frequencies in rad·s⁻¹ (CONVENTIONS §1 of the parent). The
  dimensionless rescaling is an internal implementation detail of the solver.
- **Decoupled analysis.** The diagnostics layer (Γ, nearest-neighbour spacing)
  acts on any `(N, 3)` configuration and does not depend on the
  equilibrium/modes engine.

## Where to go next

- [Getting Started](getting-started.md) — install and a runnable quick start.
- [Tutorials](tutorials/index.md) — feature-by-feature, runnable on Colab.
- [Conventions](conventions.md) — units and the §10/§11 mode contract.
- [Validation](validation.md) — the oracles that pin the physics down.

## Endorsement Marker

Local candidate framework under active stewardship. No external endorsement is
implied.
