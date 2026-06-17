# Welcome

<section class="hero-panel">
  <p class="hero-kicker">iontrap-structure · v0.1 — equilibrium configurations + normal modes, first slice</p>
  <h1>Classical structural dynamics of trapped-ion crystals</h1>
  <p class="hero-lede">
    A small, typed, clean-room Python library that computes the equilibrium
    configuration and normal modes of a trapped-ion Coulomb crystal — and hands
    the modes back to <a href="https://uwarring82.github.io/iontrap-dynamics/">iontrap-dynamics</a>
    as <code>ModeConfig</code> records.
  </p>
  <div class="hero-actions">
    <a class="hero-button hero-button-primary" href="getting-started/">Read the quick start</a>
    <a class="hero-button" href="tutorials/">Browse the tutorials</a>
  </div>
  <p class="hero-meta">
    Anisotropic harmonic trap · analytic Coulomb Hessian · mass-symmetrised normal modes · Γ diagnostics.
  </p>
</section>

<div class="grid cards landing-cards" markdown>

-   :material-atom-variant:{ .lg .middle } __What this is__

    The *producer* side of a boundary that `iontrap-dynamics` deliberately
    delegates: it computes equilibrium positions and normal-mode structure and
    exports them as `ModeConfig`-compatible records (CONVENTIONS §10/§11).

-   :material-vector-triangle:{ .lg .middle } __Pure functions over frozen records__

    No fluent builder, no stateful engine, no public DSL. `equilibrium(...)`
    and `normal_modes(...)` return immutable result dataclasses you can pickle,
    hash, and pass around.

-   :material-check-decagram-outline:{ .lg .middle } __Validated, not just written__

    The James-1998 linear chain (positions, ω_z COM, √3 ω_z stretch), the §11
    mode contract, a mixed-species generalized-eigenproblem residual, and
    finite-difference checks of every analytic derivative.

-   :material-license:{ .lg .middle } __Clean-room and MIT__

    The physics is implemented from published papers, never ported from
    copyleft or unlicensed code. No copyleft runtime dependency.

</div>

## Why this project exists

`iontrap-dynamics` models the open-system *quantum* dynamics of trapped-ion
spin-motion systems. Its Design Principle 2 bars its `modes.py` from deriving
mode structure itself — it **consumes** externally supplied normal modes.
`iontrap-structure` is the sibling that **produces** that structure: given a
trap, masses, and charges, it solves the classical force balance for the
equilibrium crystal and diagonalises the mass-symmetrised dynamical matrix to
get the modes, then exports them across the boundary.

That split keeps each package honest about its scope. Changing the crystal —
ion number, species mix, trap anisotropy — is a configuration change here, and
the quantum-dynamics layer downstream never has to know how the modes were
computed.

## What is in the first slice

<div class="status-strip" markdown>

Anisotropic harmonic (pseudopotential) trap model  
Pairwise Coulomb energy / gradient / analytic Hessian  
Dimensionless equilibrium solver with analytic Jacobian  
Normal modes via the mass-symmetrised dynamical matrix `M⁻¹ᐟ² H M⁻¹ᐟ²`  
`ModeConfig` export (CONVENTIONS §10/§11) · plasma coupling parameter Γ

</div>

- **`HarmonicTrap`** — anisotropic harmonic confinement `(ω_x, ω_y, ω_z)`.
- **`equilibrium`** — force-balance solver in dimensionless units with an
  analytic Jacobian; returns a frozen `EquilibriumResult`.
- **`normal_modes`** — eigenproblem on `D = M⁻¹ᐟ² H M⁻¹ᐟ²`; returns a frozen
  `ModeResult` whose `to_mode_configs()` emits `ModeConfig`-compatible records.
- **`coupling_parameter` / `diagnostics`** — a decoupled, engine-agnostic
  analysis layer (plasma coupling parameter Γ, nearest-neighbour spacing).

## Boundaries

<div class="grid cards landing-cards" markdown>

-   __In scope__

    Linear / harmonic equilibrium and normal modes, mixed-species crystals,
    the `ModeConfig` export, and a decoupled structural-diagnostics layer.

-   __Out of scope (v0.1)__

    Large-N molecular dynamics and the full eigensolve at N ≳ 10³, neutral-atom
    backgrounds / hybrid collisions, reaction chemistry, electrode / BEM field
    solvers, and any public DSL.

</div>

## Endorsement Marker

Local candidate framework under active stewardship. No external endorsement is
implied.
