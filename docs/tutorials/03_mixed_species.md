# Tutorial 3 — Mixed-species crystals

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uwarring82/iontrap-structure/blob/main/docs/tutorials/notebooks/03_mixed_species.ipynb) — run every step live in your browser, no install needed. The notebook is generated from this page by [`tools/build_tutorial_notebooks.py`](https://github.com/uwarring82/iontrap-structure/blob/main/tools/build_tutorial_notebooks.py).

**Goal.** By the end of this tutorial you will have computed the normal
modes of a sympathetic-cooling "sandwich" — a ²⁵Mg⁺ / ⁴⁴Ca⁺ / ²⁵Mg⁺
chain with unequal masses — and seen *exactly why* `normal_modes`
diagonalises the symmetric, mass-symmetrised dynamical matrix
`D = M⁻¹ᐟ² H M⁻¹ᐟ²` rather than the non-symmetric `M⁻¹H`. For equal
masses the two routes coincide; for mixed species they emphatically do
not, and only the symmetric route gives the orthonormal eigenvectors the
[CONVENTIONS §11](https://github.com/uwarring82/iontrap-dynamics) contract demands.

**Expected time.** ~12 min reading; ~2 s runtime.

**Prerequisites.** A working install (`pip install -e ".[dev,plot]"` in
the repo root) and Tutorials [1](01_first_crystal.md) and
[2](02_modeconfig_handoff.md) for the four-step pattern
(`HarmonicTrap` → `equilibrium` → `normal_modes` → analyse / export).

---

## The scenario

Three ions on the trap axis: a heavy ⁴⁴Ca⁺ "coolant" in the centre,
flanked by two light ²⁵Mg⁺ "logic" ions. This is the canonical
sympathetic-cooling geometry — laser-cool the Ca⁺, and the shared
motional modes carry the cooling to the Mg⁺ ions that you actually do
quantum logic on. The trap is a typical linear Paul well,
`(ω_x, ω_y, ω_z) / 2π = (8, 9, 1) MHz`, with `ω_x, ω_y ≫ ω_z` so the
crystal sits as a linear chain along `z`.

The mass asymmetry is the whole point. When all ions are identical, the
ordinary eigenvalue problem `M⁻¹H b = ω² b` already produces orthonormal
`b`, because `M ∝ I` factors out. With unequal masses `M⁻¹H` is no longer
symmetric, its eigenvectors are *not* Euclidean-orthonormal, and a
downstream `ModeConfig` built from them would violate the §11
normalisation. The fix is textbook: symmetrise to
`D = M⁻¹ᐟ² H M⁻¹ᐟ²` and diagonalise that.

## Step 1 — Build the sandwich and solve its modes

We follow the same four-step pattern as Tutorial 1, only with a
per-ion mass array `[25, 44, 25] amu` instead of a uniform one. The
imports for the *whole page* live in this first block.

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import atomic_mass, elementary_charge

from iontrap_structure import HarmonicTrap, coulomb, equilibrium, normal_modes

# House colours.
BLUE, RED, GREEN, GREY = "#1f77b4", "#d62728", "#2ca02c", "#444444"

trap = HarmonicTrap(wx=2 * np.pi * 8e6, wy=2 * np.pi * 9e6, wz=2 * np.pi * 1e6)

# ²⁵Mg⁺ / ⁴⁴Ca⁺ / ²⁵Mg⁺ — heavy coolant in the middle, all singly charged.
masses = np.array([25.0, 44.0, 25.0]) * atomic_mass
charges = np.full(3, elementary_charge)

eq = equilibrium(trap=trap, masses=masses, charges=charges)
modes = normal_modes(eq)

freqs_mhz = modes.frequencies_rad_s / (2 * np.pi) / 1e6
print(f"converged: {eq.converged};  residual force = {eq.residual_force:.2e} N")
print(f"equilibrium z (µm): {np.round(eq.positions[:, 2] * 1e6, 3)}")
print(f"mode frequencies (MHz): {np.round(freqs_mhz, 4)}")

assert eq.converged
assert modes.n_modes == 9  # 3N modes for N = 3
```

The chain is symmetric about the centre — the heavy Ca⁺ sits at the
origin, the two Mg⁺ ions at `±5.6 µm` — but the *mode structure* is not
the equal-mass one: the nine frequencies no longer come in the tidy
ratios of a monospecies chain.

## Step 2 — The §11 contract still holds for unequal masses

`ModeResult.eigenvectors` has shape `(3N, N, 3)`; flatten each mode to a
`3N`-vector, stack them, and the result must be an orthogonal matrix —
the §11 normalisation `Σ_i ‖b_{i,m}‖² = 1` plus mutual orthogonality.
This is the guarantee a `ModeConfig` consumer relies on, and it holds
*regardless of the mass spread* precisely because `normal_modes`
diagonalises the symmetric `D`.

```python
# Stack the 3N eigenvectors as rows of a (3N, 3N) matrix.
B = modes.eigenvectors.reshape(modes.n_modes, -1)
gram = B @ B.T

assert np.allclose(gram, np.eye(modes.n_modes), atol=1e-8)  # Euclidean-orthonormal
assert np.all(modes.frequencies_rad_s > 0.0)  # a stable minimum: every ω² > 0

print(f"max |Gram − I| off the diagonal = {np.max(np.abs(gram - np.eye(modes.n_modes))):.2e}")
print(f"lowest mode = {modes.frequencies_rad_s.min() / (2 * np.pi) / 1e6:.4f} MHz (> 0)")
```

!!! note "Why orthonormality is non-trivial here"

    For equal masses, `M = m·I`, so `M⁻¹H = (1/m)·H` is symmetric and its
    eigenvectors are automatically orthonormal — you could skip the
    symmetrisation entirely. With mixed masses `M⁻¹H` is *not* symmetric,
    its eigenvectors are not orthogonal, and feeding them to a
    `ModeConfig` would silently break the §11 contract. The orthonormal
    `B` above is the eigenbasis of `D = M⁻¹ᐟ² H M⁻¹ᐟ²`, which `eigh`
    returns orthonormal by construction.

## Step 3 — The centre-of-mass frequency oracle

A uniform rigid translation of the whole crystal leaves every
inter-ion distance unchanged, so the Coulomb energy is invariant and
the restoring force comes purely from the trap. That makes the three
centre-of-mass (COM) modes sit at *exactly* `ω_x`, `ω_y`, `ω_z` — for
**any** masses, any charges, any number of ions. It is the cleanest
sanity check in the whole package.

```python
wx, wy, wz = trap.wx, trap.wy, trap.wz
f = modes.frequencies_rad_s

# Each trap frequency must appear (essentially) exactly in the spectrum.
for label, w in [("ω_z", wz), ("ω_x", wx), ("ω_y", wy)]:
    closest = float(np.min(np.abs(f - w)))
    print(f"min |ω − {label}| = {closest / (2 * np.pi):.3e} Hz")
    assert np.isclose(closest, 0.0, atol=1e-3 * w)  # COM mode sits exactly on the trap line

# The axial COM mode (ω_z) is the lowest of all nine — it is the
# softest direction, and the one sympathetic cooling drives hardest.
assert np.isclose(f.min(), wz, rtol=1e-6)
print(f"axial COM mode = {f.min() / (2 * np.pi) / 1e6:.4f} MHz  (= ω_z / 2π)")
```

## Step 4 — The heart of it: exported `b` vs. physical displacement `x`

Here is the subtlety the symmetrisation exists for. The exported
eigenvectors `b` are the orthonormal eigenbasis of `D`. The *physical*
displacement pattern of a mode is `x = M⁻¹ᐟ² b` — the actual metres each
ion moves. These two are **different vectors** for mixed species, and
they obey **different** orthonormality relations:

- `b` is Euclidean-orthonormal: `Bᵀ B = I` (Step 2).
- `x` is **mass**-orthonormal: `xᵀ M x = I`, *not* `xᵀ x = I`.

The §11 contract exports `b` because Euclidean orthonormality is the
basis-independent statement a consumer can rely on. But you must never
forget that `b` is not the displacement — to draw the motion, transform
to `x` first.

```python
# Per-axis mass vector for the [x0,y0,z0, x1,y1,z1, …] layout.
m3 = np.repeat(masses, 3)
inv_sqrt_m = 1.0 / np.sqrt(m3)

# Physical displacements: x_m = M^{-1/2} b_m, one row per mode.
X = B * inv_sqrt_m[None, :]
M = np.diag(m3)

# (a) x is M-orthonormal: x^T M x = I.
xMx = X @ M @ X.T
assert np.allclose(xMx, np.eye(modes.n_modes), atol=1e-8)

# (b) x is NOT Euclidean-orthonormal. Normalise each x to unit length,
#     then the off-diagonal of x x^T is plainly non-zero.
Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
xxT = Xn @ Xn.T
off_diag = xxT - np.diag(np.diagonal(xxT))
max_off = float(np.max(np.abs(off_diag)))

assert max_off > 1e-3  # physical displacements are visibly non-orthogonal
print(f"max |xᵀMx − I| off-diagonal = {np.max(np.abs(xMx - np.eye(modes.n_modes))):.2e}  (≈ 0)")
print(f"max off-diagonal of normalised x xᵀ = {max_off:.4f}  (≫ 1e-3)")
```

!!! note "This is why `modes.py` diagonalises `D`, not `M⁻¹H`"

    If we diagonalised the non-symmetric `M⁻¹H` we would get the physical
    displacements `x` directly — but they are only `M`-orthonormal, so
    exporting them would break §11. By diagonalising the symmetric `D`
    we get the Euclidean-orthonormal `b = M¹ᐟ² x` instead, which *is* the
    §11 set. The mass information is not lost; it is folded into the
    `M⁻¹ᐟ²` that relates the two. The contrast above — `xᵀMx ≈ I` but
    `xxᵀ` visibly off-diagonal — is the entire reason the symmetrisation
    is not optional for mixed species.

## Step 5 — Cross-check via the generalised eigenproblem

The definitive test that `b` (or equivalently `x`) really are modes:
plug them back into the generalised eigenvalue equation
`H x = ω² M x`. We rebuild `H` from scratch — trap Hessian plus Coulomb
Hessian at the equilibrium — and check the *relative* residual of every
mode, so the test is scale-free.

```python
H = trap.hessian(masses) + coulomb.hessian(eq.positions, eq.charges)

max_residual = 0.0
for m in range(modes.n_modes):
    x = X[m]                       # physical displacement of mode m
    omega2 = modes.frequencies_rad_s[m] ** 2
    lhs = H @ x
    rhs = omega2 * (M @ x)
    rel = np.linalg.norm(lhs - rhs) / (np.linalg.norm(lhs) + np.linalg.norm(rhs))
    max_residual = max(max_residual, rel)

assert max_residual < 1e-9  # every mode solves H x = ω² M x to machine precision
print(f"max relative residual ‖Hx − ω²Mx‖ / (‖Hx‖ + ‖ω²Mx‖) = {max_residual:.2e}")
```

A residual at the `1e-15` level confirms the symmetric route reproduces
the true generalised eigenproblem exactly — it is not an approximation,
just a better-conditioned, symmetry-preserving way to solve the same
equation.

## Step 6 — Picture a mode: the heavy ion barely moves

The axial **stretch** mode (at `√3 ω_z ≈ 1.73 MHz`) is the prettiest
illustration of mass asymmetry. By the chain's mirror symmetry the
central ion sits at a node — and here that node is the *heavy* ⁴⁴Ca⁺,
which stays put while the two light ²⁵Mg⁺ ions swing in opposition. We
plot the axial `z`-component of each ion's displacement.

```python
# Identify the axial stretch mode: closest frequency to √3 ω_z.
stretch_idx = int(np.argmin(np.abs(modes.frequencies_rad_s - np.sqrt(3.0) * wz)))
b_stretch = modes.eigenvectors[stretch_idx]      # (N, 3) exported eigenvector
z_disp = b_stretch[:, 2]
amps = np.linalg.norm(b_stretch, axis=1)

# The heavy central ion is the node: its amplitude is ~0, far below the Mg ions'.
assert amps[1] < 1e-6
assert amps[0] > 0.5 and amps[2] > 0.5
print(f"stretch mode at {modes.frequencies_rad_s[stretch_idx] / (2 * np.pi) / 1e6:.4f} MHz")
print(f"per-ion amplitudes (Mg, Ca, Mg): {np.round(amps, 4)}")

z0 = eq.positions[:, 2] * 1e6  # equilibrium z in µm
species = ["²⁵Mg⁺", "⁴⁴Ca⁺", "²⁵Mg⁺"]
colours = [BLUE, RED, BLUE]

fig, ax = plt.subplots(figsize=(5.4, 3.0))
ax.axhline(0.0, color=GREY, linewidth=0.8, zorder=1)
ax.scatter(z0, np.zeros_like(z0), s=90, c=colours, zorder=3, edgecolor="k", linewidth=0.5)
# Arrows = the axial displacement pattern (scaled for visibility).
scale = 3.0
ax.quiver(z0, np.zeros_like(z0), z_disp * scale, np.zeros_like(z0),
          angles="xy", scale_units="xy", scale=1.0, color=GREEN,
          width=0.012, zorder=2)
for zi, s in zip(z0, species):
    ax.annotate(s, (zi, 0.0), textcoords="offset points", xytext=(0, 14), ha="center")
ax.set_xlabel(r"equilibrium position $z$ (µm)")
ax.set_yticks([])
ax.set_ylim(-1.0, 1.0)
ax.set_title("Step 6 · axial stretch mode — the heavy ⁴⁴Ca⁺ sits at the node")
plt.show()
```

The green arrows show the two ²⁵Mg⁺ ions moving outward/inward in
opposition while the ⁴⁴Ca⁺ stays fixed. In a sympathetic-cooling run
this mode would couple poorly to a cooling laser aimed at the central
Ca⁺ — a structural fact that falls straight out of the eigenvector,
before any dynamics simulation.

!!! note "Two ions are deceptively mass-blind"

    For a *two-ion* chain the axial modes are `ω_z` (COM) and `√3 ω_z`
    (stretch) for **any** mass ratio. The reason is that in this model
    the trap stiffness scales as `m ω²`, so the mass cancels in the
    axial 2×2 problem. Genuine mass-dependence in the spectrum first
    appears at `N ≥ 3` — which is exactly why this tutorial uses the
    three-ion sandwich rather than a pair.

## Next steps

- **[Your first ion crystal](01_first_crystal.md)** — the equal-mass
  baseline whose `ω_z`, `√3 ω_z` ladder this tutorial generalises.
- **[The ModeConfig handoff](02_modeconfig_handoff.md)** — export
  these mixed-species modes across the boundary into `iontrap-dynamics`
  with `to_mode_configs()`; the orthonormal `b` is what crosses.
- **[The plasma coupling parameter Γ](04_coupling_parameter.md)** —
  quantify how crystalline this configuration is.
- **[Stability and the linear→zigzag transition](05_zigzag_stability.md)** —
  soften the radial confinement until a transverse mode goes soft and
  `normal_modes` refuses to return real frequencies.

## Licence

This tutorial is Sail material — adaptive guidance with specific worked
parameter choices, not a coastline constraint. Licensed under
**CC BY-NC-SA 4.0** per [`docs/LICENCE`](https://github.com/uwarring82/iontrap-structure/blob/main/docs/LICENCE).
