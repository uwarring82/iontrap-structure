# Tutorial 1 — Your first ion crystal

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uwarring82/iontrap-structure/blob/main/docs/tutorials/notebooks/01_first_crystal.ipynb) — run every step live in your browser, no install needed. The notebook is generated from this page by [`tools/build_tutorial_notebooks.py`](https://github.com/uwarring82/iontrap-structure/blob/main/tools/build_tutorial_notebooks.py).

**Goal.** Build a three-ion ²⁵Mg⁺ crystal from scratch: configure an
anisotropic harmonic trap, solve for the equilibrium positions, compute
the normal modes, and read off the textbook axial centre-of-mass and
stretch frequencies. This is the canonical "Hello world" for
`iontrap-structure` and establishes the **configure → solve → modes →
analyse** pattern every other tutorial reuses.

**Expected time.** ~10 min reading; ~1 s runtime.

**Prerequisites.** A working install (`pip install -e ".[dev,plot]"` in
the repo root) and ion-trap terminology at the level of an introductory
course. No derivations required — the formulas appear only for orientation.

---

## The scenario

Three ²⁵Mg⁺ ions in a linear Paul trap, modelled here as its
time-averaged secular **pseudopotential**: an anisotropic 3D harmonic
well. We make the radial confinement stiff and the axial confinement
soft,

```
ω_x / 2π = ω_y / 2π = 10 MHz   (radial, strong)
ω_z / 2π =  1 MHz              (axial, weak)
```

so the Coulomb repulsion pushes the ions apart along the soft `z` axis
and they settle into a **linear chain**. For `N = 3` the equilibrium is
known in closed form (James, *Appl. Phys. B* **66**, 181, 1998): the
outer ions sit at `±(5/4)^{1/3} ℓ` where `ℓ` is the trap's natural
length scale, and the two lowest axial modes are the centre-of-mass mode
at `ω_z` and the stretch mode at `√3 ω_z`. We'll reproduce both exactly.

## Step 1 — Configure the physical system

Three objects describe the lab: the trap (a `HarmonicTrap` of secular
frequencies in rad·s⁻¹) and two per-ion arrays giving each ion's mass
and charge. ²⁵Mg⁺ is one isotope, so all three ions share `m = 25 u`
and `q = +e`.

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import atomic_mass, elementary_charge

from iontrap_structure import HarmonicTrap, equilibrium, length_scale, normal_modes

# House colours — keep figures consistent across the tutorial series.
BLUE, RED, GREEN, GREY = "#1f77b4", "#d62728", "#2ca02c", "#444444"

# Anisotropic well: stiff radial (x, y), soft axial (z) → ions chain on z.
trap = HarmonicTrap(
    wx=2 * np.pi * 10e6,
    wy=2 * np.pi * 10e6,
    wz=2 * np.pi * 1e6,
)

# Three identical ²⁵Mg⁺ ions: mass 25 u, charge +e.
n_ions = 3
mass = 25 * atomic_mass
masses = np.full(n_ions, mass)
charges = np.full(n_ions, elementary_charge)

print(f"ω_x/2π = ω_y/2π = {trap.wx / (2 * np.pi) * 1e-6:.1f} MHz  "
      f"(radial);  ω_z/2π = {trap.wz / (2 * np.pi) * 1e-6:.1f} MHz  (axial)")
```

!!! note "Why the frequency anisotropy decides the geometry"

    Whether a crystal is a line, a zigzag, or a 3D ball is set by the
    ratio of radial to axial confinement. With `ω_radial / ω_z = 10`
    the radial well is far too stiff for the ions to spread sideways,
    so they line up on the `z` axis. Tutorial 5 traces what happens as
    that ratio is squeezed toward the linear→zigzag instability.

## Step 2 — Solve for the equilibrium

`equilibrium()` finds the force balance `∂E/∂r = 0` for the combined
trap + Coulomb potential. It returns a frozen `EquilibriumResult`; the
default initial guess is an on-axis chain, which is exactly right when
the radial confinement dominates. Always check `.converged` before
trusting the positions.

```python
eq = equilibrium(trap=trap, masses=masses, charges=charges)

assert eq.converged                       # force-balance solve succeeded
assert eq.n_ions == 3
print(f"converged = {eq.converged};  residual force = {eq.residual_force:.2e} N")

# positions is (N, 3) in metres. With z soft, the ions chain on the z-axis,
# so the x and y coordinates should be essentially zero.
xy = eq.positions[:, :2]
assert np.allclose(xy, 0.0, atol=1e-12)   # ions lie on the z-axis
print("positions (µm):\n", np.round(eq.positions * 1e6, 4))
```

The chain is centred on the origin: ion 0 and ion 2 sit symmetrically
about the centre ion. The closed-form prediction uses the trap's
**length scale** `ℓ = (k_e q² / (m ω_z²))^{1/3}`, the distance at which
Coulomb repulsion and axial confinement balance. James (1998) gives the
outer-ion position for `N = 3` as `z = ±(5/4)^{1/3} ℓ`.

```python
ell = length_scale(trap, mass, elementary_charge)
print(f"length scale ℓ = {ell * 1e6:.4f} µm")

z = eq.positions[:, 2]
z_outer = np.max(np.abs(z))               # the symmetric outer-ion distance
z_outer_expected = (5.0 / 4.0) ** (1.0 / 3.0) * ell

assert np.isclose(z_outer, z_outer_expected, rtol=1e-6)  # James 1998 closed form
print(f"outer ion at z = ±{z_outer * 1e6:.4f} µm  "
      f"= ±{z_outer / ell:.6f} ℓ  (expected (5/4)^(1/3) = "
      f"{(5 / 4) ** (1 / 3):.6f})")
```

## Step 3 — Compute the normal modes

`normal_modes()` builds the total Hessian at equilibrium, forms the
mass-symmetrised dynamical matrix `D = M^{-1/2} H M^{-1/2}`, and
diagonalises it. The result is a `ModeResult` with `3N = 9` frequencies
sorted **ascending** in rad·s⁻¹, and matching `(3N, N, 3)` eigenvectors
(one `(N, 3)` displacement pattern per mode).

```python
modes = normal_modes(eq)

assert modes.n_modes == 3 * n_ions        # 9 modes for 3 ions in 3D
freqs = modes.frequencies_rad_s
assert np.all(np.diff(freqs) >= 0)        # frequencies are sorted ascending

freqs_mhz = freqs / (2 * np.pi) * 1e-6
print("mode frequencies / 2π (MHz):")
print(np.round(freqs_mhz, 4))
```

The nine frequencies fall into two families: three **axial** modes
(polarised along `z`, near and above `ω_z = 1 MHz`) and six **radial**
modes (polarised in `x`/`y`, bunched just below the bare radial
frequency of 10 MHz). Because the axial well is soft, the axial modes
are the lowest-lying — and the most useful for quantum-logic schemes.

## Step 4 — Identify the axial COM and stretch modes

The two lowest modes have textbook closed forms. The **centre-of-mass
(COM)** mode is a rigid axial translation of the whole chain: every ion
moves the same amount along `z`. A uniform shift leaves all interparticle
distances — and hence the Coulomb energy — unchanged, so the only
restoring force is the trap itself, giving `ω_COM = ω_z` exactly.

```python
com_freq = freqs[0]
assert np.isclose(com_freq, trap.wz, rtol=1e-6)   # COM mode = bare axial freq
print(f"COM mode:    ω/2π = {com_freq / (2 * np.pi) * 1e-6:.6f} MHz  "
      f"(= ω_z = {trap.wz / (2 * np.pi) * 1e-6:.4f} MHz)")

# COM eigenvector: all three ions displace equally along +z (same sign).
com_vec = modes.eigenvectors[0]
com_z = com_vec[:, 2]
assert np.allclose(np.abs(com_z), np.abs(com_z[0]), rtol=1e-6)   # equal amplitude
assert np.all(np.sign(com_z) == np.sign(com_z[0]))              # same direction
```

The **stretch** (breathing) mode is the next one up: the centre ion
stays put while the two outer ions move outward and inward in
antiphase, modulating the spacing. Linearising the Coulomb restoring
force about the `N = 3` equilibrium adds exactly `2 ω_z²` of stiffness
on top of the trap, so `ω_stretch² = 3 ω_z²`, i.e. `ω_stretch = √3 ω_z`.

```python
stretch_freq = freqs[1]
assert np.isclose(stretch_freq, np.sqrt(3.0) * trap.wz, rtol=1e-6)  # √3 ω_z
print(f"stretch mode: ω/2π = {stretch_freq / (2 * np.pi) * 1e-6:.6f} MHz  "
      f"= √3 ω_z  (ratio {stretch_freq / com_freq:.6f}, √3 = {np.sqrt(3):.6f})")

# Both modes are purely axial: all of the mode "energy" is in the z-component.
for label, m_idx in (("COM", 0), ("stretch", 1)):
    vec = modes.eigenvectors[m_idx]
    z_fraction = float(np.sum(vec[:, 2] ** 2))   # Σ|b_z|² ; ‖b‖²=1 per §11
    assert np.isclose(z_fraction, 1.0, atol=1e-9)   # z-polarised
    print(f"{label:>7} mode: z-polarisation fraction = {z_fraction:.6f}")

# Stretch eigenvector: centre ion barely moves; outer ions are antiphase.
stretch_z = modes.eigenvectors[1][:, 2]
assert abs(stretch_z[1]) < 1e-6                    # centre ion stationary
assert np.sign(stretch_z[0]) == -np.sign(stretch_z[2])   # outer ions antiphase
```

!!! note "Eigenvector normalisation (CONVENTIONS §11)"

    The eigenvectors come from the *symmetric* dynamical matrix, so each
    mode satisfies `Σ_i ‖b_{i,m}‖² = 1` and distinct modes are
    orthonormal. That is precisely the normalisation the sibling
    `iontrap-dynamics` package expects in its `ModeConfig` Lamb–Dicke
    factors — which is why `ModeResult.to_mode_configs()` (Tutorial 2)
    can hand modes across with no rescaling.

## Step 5 — Visualise positions and the mode spectrum

Two panels tell the whole story: the chain geometry on the left, the
mode spectrum on the right. The axial modes (COM, stretch, and the third
axial mode) sit at low frequency; the six radial modes pile up near
10 MHz.

```python
# Sort ions by z for a tidy left-to-right scatter.
order = np.argsort(eq.positions[:, 2])
z_um = eq.positions[order, 2] * 1e6

# Tag each mode axial (mostly z) vs radial (mostly x/y) for colouring.
z_fracs = np.sum(modes.eigenvectors[:, :, 2] ** 2, axis=1)
is_axial = z_fracs > 0.5
colours = np.where(is_axial, BLUE, GREY)

fig, (ax_pos, ax_spec) = plt.subplots(1, 2, figsize=(9.0, 3.4))

# Left: ion positions on the z-axis.
ax_pos.scatter(z_um, np.zeros_like(z_um), s=180, color=RED, edgecolor=GREY, zorder=3)
for zi in z_um:
    ax_pos.annotate(f"{zi:.2f}", (zi, 0), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=8, color=GREY)
ax_pos.axhline(0, color=GREY, linewidth=0.8, zorder=1)
ax_pos.set_yticks([])
ax_pos.set_xlabel("axial position z (µm)")
ax_pos.set_title("Step 5 · linear ²⁵Mg⁺ chain (N = 3)")

# Right: the nine mode frequencies as a stem plot, axial vs radial.
idx = np.arange(modes.n_modes)
ax_spec.vlines(idx, 0, freqs_mhz, color=colours, linewidth=1.5)
ax_spec.scatter(idx, freqs_mhz, color=colours, s=30, zorder=3)
ax_spec.axhline(trap.wz / (2 * np.pi) * 1e-6, color=GREEN, linestyle="--",
                linewidth=1.0, label=r"$\omega_z/2\pi$")
ax_spec.axhline(np.sqrt(3) * trap.wz / (2 * np.pi) * 1e-6, color=BLUE,
                linestyle=":", linewidth=1.0, label=r"$\sqrt{3}\,\omega_z/2\pi$")
ax_spec.set_xlabel("mode index (ascending)")
ax_spec.set_ylabel(r"$\omega_m/2\pi$ (MHz)")
ax_spec.set_title("Step 5 · normal-mode spectrum")
ax_spec.legend(frameon=False, fontsize=8, loc="center right")

fig.tight_layout()
plt.show()

# Three axial modes (z-polarised), six radial — a quick spectral sanity check.
assert int(np.sum(is_axial)) == 3
print(f"axial modes: {int(np.sum(is_axial))};  radial modes: "
      f"{int(np.sum(~is_axial))}")
```

The left panel shows the three ions evenly straddling the origin, the
outer pair at `±(5/4)^{1/3} ℓ ≈ ±5.60 µm`. The right panel makes the
two-family structure obvious: the green dashed line lands on mode 0
(COM), the blue dotted line on mode 1 (stretch), and the six radial
modes cluster near the bare 10 MHz radial frequency.

## Next steps

You now have the core workflow — **configure** a `HarmonicTrap`,
**solve** with `equilibrium()`, compute **modes** with `normal_modes()`,
and **analyse** the result. Each remaining tutorial swaps one piece into
this same skeleton:

- **[The ModeConfig handoff](02_modeconfig_handoff.md)** — export these
  modes to `iontrap-dynamics` via `ModeResult.to_mode_configs()` and run
  spin–motion dynamics on them.
- **[Mixed-species crystals](03_mixed_species.md)** — give the ions
  different masses and watch the modes localise onto the lighter species.
- **[The plasma coupling parameter Γ](04_coupling_parameter.md)** — add a
  temperature and quantify how deep in the crystalline regime you are.
- **[Stability and the linear→zigzag transition](05_zigzag_stability.md)**
  — squeeze the radial confinement until the linear chain buckles.

## Licence

This tutorial is Sail material — adaptive guidance with specific
parameter choices, not a coastline constraint. Licensed under
**CC BY-NC-SA 4.0** per [`docs/LICENCE`](https://github.com/uwarring82/iontrap-structure/blob/main/docs/LICENCE).
