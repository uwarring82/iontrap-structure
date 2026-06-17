# Conventions

`iontrap-structure` adopts the conventions of the parent project so its output
drops straight into `iontrap-dynamics`. The binding document is the parent's
[`CONVENTIONS.md`](https://github.com/uwarring82/iontrap-dynamics/blob/main/CONVENTIONS.md);
this page summarises the parts this package touches.

## Units (§1)

Everything in the public API is SI:

| Quantity | Unit |
|---|---|
| position | metre (m) |
| mass | kilogram (kg) |
| charge | coulomb (C) |
| angular frequency | rad·s⁻¹ |
| temperature | kelvin (K) |

The equilibrium solver rescales to dimensionless units internally (length scale
`ℓ = (k_e q_ref² / (m_ref ω_z²))^{1/3}`) for conditioning, but inputs and
outputs are always SI.

## Coordinate layout

Gradients and Hessians use a **per-ion-contiguous** 3N layout:

```
[x₀, y₀, z₀, x₁, y₁, z₁, …, x_{N-1}, y_{N-1}, z_{N-1}]
```

so a `(3N,)` mode eigenvector reshapes directly to `(N, 3)` — one row per ion —
with no axis-blocked transpose.

## The mode contract (§10 / §11)

A normal mode is exported as a `ModeConfig`-compatible record with:

- **`frequency_rad_s`** — the mode angular frequency `ω_m > 0`, in rad·s⁻¹.
- **`eigenvector_per_ion`** — shape `(N, 3)`, the per-ion displacement pattern,
  normalised so that

    ```
    Σᵢ ‖b_{i,m}‖² = 1
    ```

  and distinct modes are mutually orthonormal.

!!! note "Why the *mass-symmetrised* eigenvectors"

    `normal_modes` diagonalises the symmetric dynamical matrix
    `D = M⁻¹ᐟ² H M⁻¹ᐟ²`, not the non-symmetric `M⁻¹H`. The eigenvectors of `D`
    are Euclidean-orthonormal, which is exactly the §11 set. They are the
    *mass-weighted* displacements, related to the raw physical displacements
    `x` by `b = M¹ᐟ² x`. The two coincide only for equal masses — for a
    mixed-species crystal you must use the symmetric route to satisfy §11
    (see [Tutorial 3](tutorials/03_mixed_species.md)).

## Sign and ordering

- Mode frequencies are returned **ascending**.
- A fully confining 3D trap has no zero or soft modes at a stable equilibrium;
  `normal_modes` raises `ValueError` if the dynamical matrix is not positive
  semi-definite (an unstable / saddle configuration — see
  [Tutorial 5](tutorials/05_zigzag_stability.md)).

## Endorsement Marker

Local candidate framework under active stewardship. No external endorsement is
implied.
