# Tutorial 2 — The ModeConfig handoff to iontrap-dynamics

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uwarring82/iontrap-structure/blob/main/docs/tutorials/notebooks/02_modeconfig_handoff.ipynb) — run every step live in your browser, no install needed. The notebook is generated from this page by [`tools/build_tutorial_notebooks.py`](https://github.com/uwarring82/iontrap-structure/blob/main/tools/build_tutorial_notebooks.py).

**Goal.** `iontrap-structure` exists to compute crystal *structure* —
equilibria and normal modes — and hand the modes across the package
boundary to `iontrap-dynamics`, which simulates the quantum dynamics on
top of them. This tutorial walks that handoff end to end: you compute the
normal modes of a three-ion ²⁵Mg⁺ chain, export them as `ModeConfig`
records, verify the §10/§11 normalisation conventions hold exactly, and
see how a downstream consumer receives them **whether or not** the parent
package is installed.

**Expected time.** ~10 min reading; ~1 s runtime.

**Prerequisites.** A working install (`pip install -e ".[dev,plot]"` in the
repo root). Tutorial 1, *Your first ion crystal*, motivates the
`equilibrium` → `normal_modes` pipeline reused here. `iontrap-dynamics`
itself is **optional** — the export degrades gracefully without it, which
is the whole point of Step 5.

---

## The scenario

Three ²⁵Mg⁺ ions in a linear Paul trap modelled as a harmonic well with
secular frequencies `(ω_x, ω_y) / 2π = 4 MHz` (radial) and
`ω_z / 2π = 1 MHz` (axial). Because the radial confinement is four times
stiffer than the axial, the ions settle into a string along `z`. A
three-ion chain has `3N = 9` normal modes; the two lowest are the axial
**centre-of-mass (COM)** mode at `ω_z` and the axial **stretch** mode at
`√3 ω_z`, the textbook ratio we will assert against.

The contract we are honouring lives in the parent package's
`CONVENTIONS.md`: §10 fixes the `ModeConfig` field layout (`label`,
`frequency_rad_s`, `eigenvector_per_ion` of shape `(N, 3)`), and §11 fixes
the normalisation — the eigenvectors are those of the **mass-symmetrised**
dynamical matrix `D = M^{-1/2} H M^{-1/2}`, hence **Euclidean-orthonormal**:
`Σ_i ‖b_{i,m}‖² = 1` for each mode `m`, with distinct modes orthogonal.

## Step 1 — Compute the equilibrium and its normal modes

We reuse the Tutorial 1 pipeline verbatim: build a `HarmonicTrap`, solve the
force balance with `equilibrium`, then diagonalise the trap+Coulomb Hessian
with `normal_modes`. The chain is stable, so all `3N` squared frequencies are
non-negative and `normal_modes` returns real `ω_m`.

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import atomic_mass, elementary_charge

from iontrap_structure import HarmonicTrap, equilibrium, normal_modes
from iontrap_structure.results import ModeConfigLike

# House colours.
BLUE, RED, GREEN, GREY = "#1f77b4", "#d62728", "#2ca02c", "#444444"

TWO_PI = 2 * np.pi
N_IONS = 3
mass = 25 * atomic_mass         # ²⁵Mg⁺
charge = elementary_charge

trap = HarmonicTrap(wx=TWO_PI * 4.0e6, wy=TWO_PI * 4.0e6, wz=TWO_PI * 1.0e6)
masses = np.full(N_IONS, mass)
charges = np.full(N_IONS, charge)

eq = equilibrium(trap=trap, masses=masses, charges=charges)
modes = normal_modes(eq)

freqs_mhz = modes.frequencies_rad_s / TWO_PI * 1e-6
print(f"Step 1 — converged: {eq.converged};  residual force = {eq.residual_force:.2e} N")
print(f"         {modes.n_modes} modes (3N = {3 * modes.n_ions});  "
      f"lowest two: {freqs_mhz[0]:.4f} MHz (COM), {freqs_mhz[1]:.4f} MHz (stretch)")

assert modes.n_modes == 3 * modes.n_ions == 9
assert np.all(modes.frequencies_rad_s > 0.0)                       # stable: real, positive ω
assert np.isclose(modes.frequencies_rad_s[0], trap.wz, rtol=1e-9)  # COM rides the bare ω_z
assert np.isclose(freqs_mhz[1] / freqs_mhz[0], np.sqrt(3.0), rtol=1e-9)  # stretch/COM = √3
```

!!! note "Why the COM mode sits exactly at ω_z"

    The Coulomb force is internal, so it cancels for any rigid
    translation of the whole crystal. The COM mode therefore feels only
    the trap, oscillating at the bare secular frequency `ω_z` regardless
    of ion number or spacing. The √3 stretch ratio is likewise
    ion-number-specific (it is 1, √3, √(29/5)… for the axial breathing
    family) and a sharp check that the Hessian assembly is correct.

## Step 2 — Export to ModeConfig records

`ModeResult.to_mode_configs()` is the boundary-crossing call. It returns one
record per mode carrying exactly the §10 fields. **If `iontrap-dynamics` is
installed** it returns its native `iontrap_dynamics.modes.ModeConfig`; **if
not**, it returns `ModeConfigLike` fallbacks with identical field names and
semantics, so the export is usable standalone. Either way the consuming code
is the same — duck typing on `.label / .frequency_rad_s / .eigenvector_per_ion`.

```python
configs = modes.to_mode_configs()

assert len(configs) == modes.n_modes
for cfg in configs:
    assert cfg.frequency_rad_s > 0.0
    assert cfg.eigenvector_per_ion.shape == (modes.n_ions, 3)
    # §11 normalisation: Σ_i ‖b_{i,m}‖² = 1 for every mode.
    norm_sq = float(np.sum(cfg.eigenvector_per_ion ** 2))
    assert np.isclose(norm_sq, 1.0, atol=1e-10)

max_dev = max(abs(float(np.sum(c.eigenvector_per_ion ** 2)) - 1.0) for c in configs)
print(f"Step 2 — exported {len(configs)} configs;  record type = {type(configs[0]).__name__}")
print(f"         every Σ_i ‖b_i‖² = 1 (max deviation {max_dev:.1e})")
```

The record type printed above will be `ModeConfigLike` in this docs build
(the parent package is an optional `[interop]` extra and is not pulled in by
`[dev,plot]`), or `ModeConfig` if you have installed `iontrap-dynamics`
alongside. The fields and values are identical either way.

## Step 3 — Cross-mode orthonormality

§11 promises more than per-mode unit norm: the full set of `3N` eigenvectors
is **Euclidean-orthonormal**. Stacking each flattened `(N, 3)` eigenvector
into the rows of a `3N × 3N` matrix `B`, the Gram matrix `B Bᵀ` must equal the
identity. This is exactly what makes the modes a clean basis for the
downstream Lamb–Dicke coupling sums.

```python
B = np.stack([cfg.eigenvector_per_ion.reshape(-1) for cfg in configs], axis=0)
gram = B @ B.T
identity = np.eye(modes.n_modes)
max_off = float(np.max(np.abs(gram - identity)))

print(f"Step 3 — Gram matrix B Bᵀ is the {gram.shape[0]}×{gram.shape[1]} identity "
      f"to {max_off:.1e}")
assert np.allclose(gram, identity, atol=1e-9)
```

!!! note "Euclidean- vs mass-orthonormal"

    These eigenvectors diagonalise the *symmetric* `D = M^{-1/2} H M^{-1/2}`,
    so they are orthonormal in the plain Euclidean sense — that is the §11
    set. The raw physical displacements (eigenvectors of `M^{-1} H`) are
    instead `M`-orthonormal and coincide with these only for equal masses.
    Tutorial 3, *Mixed-species crystals*, is where that distinction starts
    to bite.

## Step 4 — Carry physically meaningful labels across the boundary

The default labels are `mode_0 … mode_{3N-1}` in ascending-frequency order.
For a chain you usually want names a human (and a downstream gate compiler)
can read. Pass `labels=` and they propagate straight into the records — the
label is the only free-form field in the §10 contract, so this is where
domain knowledge attaches.

```python
labels = [f"mode_{m}" for m in range(modes.n_modes)]
labels[0] = "axial_com"
labels[1] = "axial_stretch"

named = modes.to_mode_configs(labels=labels)
by_label = {cfg.label: cfg for cfg in named}

assert named[0].label == "axial_com"
assert named[1].label == "axial_stretch"
assert set(by_label) == set(labels)            # all labels present, one per mode

com = by_label["axial_com"]
stretch = by_label["axial_stretch"]
print(f"Step 4 — {com.label}: {com.frequency_rad_s / TWO_PI * 1e-6:.4f} MHz")
print(f"         {stretch.label}: {stretch.frequency_rad_s / TWO_PI * 1e-6:.4f} MHz")
```

## Step 5 — How a downstream consumer receives the modes

Here is the boundary made explicit. A consumer in `iontrap-dynamics` would
*prefer* the native `ModeConfig`, but robust code never hard-imports an
optional dependency at module top level — it tries, and falls back. The
snippet below runs identically in both worlds because both record types
expose the same three attributes.

```python
try:
    from iontrap_dynamics.modes import ModeConfig
    target_type = ModeConfig
    flavour = "native iontrap_dynamics.modes.ModeConfig"
except ImportError:
    target_type = ModeConfigLike
    flavour = "iontrap-structure ModeConfigLike fallback"


# A downstream builder consumes the records by duck-typed attribute access —
# it neither knows nor cares which concrete class produced them.
def lamb_dicke_axial_weights(config):
    """Per-ion |z-projection| of a mode — the input a sideband builder needs."""
    return np.abs(config.eigenvector_per_ion[:, 2])


handoff = modes.to_mode_configs(labels=labels)
assert all(isinstance(cfg, target_type) for cfg in handoff)

com_weights = lamb_dicke_axial_weights(by_label["axial_com"])
print(f"Step 5 — consumer would receive: {flavour}")
print(f"         COM |z| weights per ion = {np.round(com_weights, 4)}")

# Physics checks on the two axial modes the consumer cares about:
#  COM  — all ions move together: |z| identical = 1/√3 on each of 3 ions.
assert np.allclose(com_weights, 1.0 / np.sqrt(3.0), atol=1e-9)
#  stretch — outer ions ±1/√2, centre ion stationary.
stretch_z = stretch.eigenvector_per_ion[:, 2]
assert np.isclose(abs(stretch_z[0]), 1.0 / np.sqrt(2.0), atol=1e-9)
assert np.isclose(abs(stretch_z[2]), 1.0 / np.sqrt(2.0), atol=1e-9)
assert abs(stretch_z[1]) < 1e-9
```

!!! note "The structure/dynamics contract"

    `iontrap-structure` owns positions, Γ, and other *structural*
    quantities in its own schema; it deliberately exports **only** the
    §10/§11 mode fields across the boundary — frequency and eigenvector,
    nothing else. That narrow waist is what lets the two packages version
    independently while staying interoperable.

## Step 6 — Visualise the COM vs stretch displacement patterns

The clearest way to see the two lowest axial modes is to plot the per-ion
`z`-displacement directly. The COM mode moves all three ions in lockstep; the
stretch mode moves the outer two against each other and leaves the centre ion
fixed — the visual signature of the √3 ratio.

```python
ion_index = np.arange(modes.n_ions)
com_z = com.eigenvector_per_ion[:, 2]
stretch_z_plot = stretch.eigenvector_per_ion[:, 2]

fig, (ax_com, ax_str) = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
ax_com.bar(ion_index, com_z, color=BLUE, width=0.6)
ax_com.set_title(f"COM · {com.frequency_rad_s / TWO_PI * 1e-6:.3f} MHz")
ax_com.set_xlabel("ion index")
ax_com.set_ylabel(r"$b_{i,z}$ (mass-symmetrised)")

ax_str.bar(ion_index, stretch_z_plot, color=RED, width=0.6)
ax_str.set_title(f"stretch · {stretch.frequency_rad_s / TWO_PI * 1e-6:.3f} MHz")
ax_str.set_xlabel("ion index")

for ax in (ax_com, ax_str):
    ax.axhline(0.0, color=GREY, linewidth=0.8)
    ax.set_xticks(ion_index)
fig.suptitle("Step 6 · axial mode eigenvectors of the 3-ion chain")
fig.tight_layout()
plt.show()
```

The blue bars are equal (rigid translation); the red bars are antisymmetric
about a stationary centre. Both bar sets satisfy `Σ_i b_{i,z}² = 1` — the same
§11 normalisation the consumer in Step 5 relies on for correctly scaled
sideband couplings.

## Next steps

- **[Your first ion crystal](01_first_crystal.md)** — the `equilibrium` →
  `normal_modes` pipeline this tutorial assumes, from scratch.
- **[Mixed-species crystals](03_mixed_species.md)** — where Euclidean- vs
  mass-orthonormality (Step 3's note) genuinely diverges.
- **[The plasma coupling parameter Γ](04_coupling_parameter.md)** — the
  structural diagnostics that stay *inside* this package's schema.
- **[Stability and the linear→zigzag transition](05_zigzag_stability.md)** —
  what happens to the modes when the chain stops being a stable minimum and
  `normal_modes` refuses to return real frequencies.

## Licence

This tutorial is Sail material — adaptive guidance with specific parameter
choices, not a coastline constraint. Licensed under **CC BY-NC-SA 4.0** per
[`docs/LICENCE`](https://github.com/uwarring82/iontrap-structure/blob/main/docs/LICENCE).
