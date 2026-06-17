# Tutorial 4 — The plasma coupling parameter Γ

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uwarring82/iontrap-structure/blob/main/docs/tutorials/notebooks/04_coupling_parameter.ipynb) — run every step live in your browser, no install needed. The notebook is generated from this page by [`tools/build_tutorial_notebooks.py`](https://github.com/uwarring82/iontrap-structure/blob/main/tools/build_tutorial_notebooks.py).

**Goal.** By the end of this tutorial you will be able to put a single
dimensionless number on the question *"how crystalline is this
configuration?"* — the plasma coupling parameter
`Γ = k_e q² / (a k_B T)`. We compute Γ for a 5-ion ²⁵Mg⁺ chain, sweep
it over four decades of temperature, verify its exact `1/T` scaling,
and read off the crossover from a Coulomb crystal to a gas.

**Expected time.** ~9 min reading; ~1 s runtime.

**Prerequisites.** A working install (`pip install -e ".[dev,plot]"`
in the repo root). Tutorial 1 ("Your first ion crystal") introduces
[`equilibrium`](01_first_crystal.md); here we reuse that crystal and
focus on the diagnostics layer.

---

## The scenario

Five ²⁵Mg⁺ ions in a linear Paul trap with radial confinement
`ω_x = ω_y = 2π · 10 MHz` and a soft axial well
`ω_z = 2π · 1 MHz`. The radial-to-axial anisotropy is `10:1`, far
above the linear→zigzag threshold (Tutorial 5), so the ions settle
onto the trap axis as a one-dimensional chain. The structure itself
is set purely by force balance — it does **not** know the temperature.
What temperature decides is whether that structure *survives* thermal
agitation, and Γ is exactly the ratio that quantifies it: the Coulomb
energy between nearest neighbours over the thermal energy `k_B T`.

## Step 1 — Build the chain and measure its spacing

The coupling parameter needs a characteristic length `a`. For a
finite crystal we use the **mean nearest-neighbour distance**:
[`mean_nearest_neighbour_distance`](https://github.com/uwarring82/iontrap-structure/blob/main/src/iontrap_structure/diagnostics.py)
averages, over all ions, the distance to each ion's closest partner.

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import atomic_mass, elementary_charge

from iontrap_structure import (
    HarmonicTrap,
    coupling_parameter,
    diagnostics,
    equilibrium,
    mean_nearest_neighbour_distance,
)

# House colours.
BLUE, RED, GREEN, GREY = "#1f77b4", "#d62728", "#2ca02c", "#444444"

n_ions = 5
masses = np.full(n_ions, 25 * atomic_mass)      # ²⁵Mg⁺ : 25 u
charges = np.full(n_ions, elementary_charge)    #          +1 e

trap = HarmonicTrap(wx=2 * np.pi * 10e6, wy=2 * np.pi * 10e6, wz=2 * np.pi * 1e6)
eq = equilibrium(trap=trap, masses=masses, charges=charges)
assert eq.converged  # force balance reached

a = mean_nearest_neighbour_distance(eq.positions)
print(f"Step 1 — converged={eq.converged};  mean NN spacing a = {a * 1e6:.3f} µm")

# Sanity: a 5-ion chain is slightly inhomogeneous — the inner gaps are
# tighter than the outer ones, and a sits between them.
z = np.sort(eq.positions[:, 2])
gaps = np.diff(z)
assert np.isclose(a, 4.48195456e-6, rtol=1e-6)        # ≈ 4.482 µm
assert gaps.min() < a < gaps.max()                    # a is a genuine mean
```

!!! note "Why nearest-neighbour, not Wigner–Seitz?"

    A bulk one-component plasma (OCP) defines `a` from the *density*
    via the Wigner–Seitz radius, `a_WS = (3 / 4πn)^{1/3}`. A 5-ion
    chain has no well-defined bulk density, so this estimator uses the
    geometric nearest-neighbour spacing instead. The two definitions
    agree up to an O(1) factor but are **not** identical — keep that in
    mind before comparing absolute melting points (see the closing
    admonition).

## Step 2 — Compute Γ and sweep it over temperature

[`coupling_parameter(positions, charges, T)`](https://github.com/uwarring82/iontrap-structure/blob/main/src/iontrap_structure/diagnostics.py)
returns `Γ = k_e q² / (a k_B T)`. Geometry (`a`) and charge (`q`) are
fixed by the crystal, so the entire temperature dependence is the
`1/T` out front. We evaluate Γ at 200 log-spaced temperatures from
0.1 mK to 1 K.

```python
T_grid = np.logspace(-4, 0, 200)   # 1e-4 K (0.1 mK) … 1e0 K (1 K)
gamma = np.array([coupling_parameter(eq.positions, charges, T) for T in T_grid])

# Γ is strictly decreasing in T (it is ∝ 1/T).
assert np.all(np.diff(gamma) < 0)

gamma_lo = coupling_parameter(eq.positions, charges, T_grid[0])    # coldest
gamma_hi = coupling_parameter(eq.positions, charges, T_grid[-1])   # hottest
print(f"Step 2 — Γ(0.1 mK) = {gamma_lo:.3e}   Γ(1 K) = {gamma_hi:.3f}")
print(f"         Γ spans {gamma_lo / gamma_hi:.0f}× across the 4-decade sweep")
```

The cold end sits in the many-thousands; the warm end drops below 4.
That spread is the whole story — but to trust it we should check the
scaling is exactly the `1/T` the formula promises.

## Step 3 — Verify the exact 1/T scaling

Because `a` and `q` are temperature-independent, Γ is a pure inverse
law: `Γ(T₁) / Γ(T₂) = T₂ / T₁`, with **no** approximation. This is a
strong, falsifiable prediction, so we assert it to near machine
precision rather than with a loose tolerance.

```python
T1, T2 = 1.0e-3, 1.0e-2          # 1 mK and 10 mK
g1 = coupling_parameter(eq.positions, charges, T1)
g2 = coupling_parameter(eq.positions, charges, T2)

ratio_gamma = g1 / g2
ratio_temp = T2 / T1
print(f"Step 3 — Γ(1 mK)/Γ(10 mK) = {ratio_gamma:.12f};  T₂/T₁ = {ratio_temp:.12f}")
assert np.isclose(ratio_gamma, ratio_temp, rtol=1e-12)   # exact 1/T, no fudge
```

!!! note "What is *not* in this estimator"

    The clean `1/T` law holds because the spacing `a` is taken from
    the (temperature-independent) zero-temperature equilibrium. A more
    complete treatment would let thermal motion inflate the effective
    spacing as the crystal softens near melting; that feedback is
    deliberately outside this first-slice diagnostic.

## Step 4 — Interpret the regimes and find the crossover

The physics lives in two reference values. `Γ ≫ 1` means Coulomb
energy dominates thermal energy: the ions are pinned, **crystalline**.
`Γ ≲ 1` means thermal energy wins and the structure melts into a
**gas/plasma**. The classic bulk-OCP simulations of Dubin & O'Neil
place the 3D fluid→crystal freezing transition near `Γ ≈ 170`.

```python
GAMMA_FREEZE = 170.0   # bulk 3D OCP fluid→crystal (Dubin & O'Neil 1999)

gamma_1mK = coupling_parameter(eq.positions, charges, 1.0e-3)
print(f"Step 4 — Γ(1 mK) = {gamma_1mK:.1f}  →  {gamma_1mK / GAMMA_FREEZE:.1f}× the OCP freezing value")
assert gamma_1mK > GAMMA_FREEZE          # deep in the crystalline regime
assert gamma_1mK > 3000.0                # in fact ≫ freezing: ~3.7e3

# Crossover temperature where Γ = 1. Since Γ = C / T with
# C ≡ Γ(T = 1 K) · (1 K), the solution is simply T_cross = C.
C = coupling_parameter(eq.positions, charges, 1.0)   # Γ at T = 1 K equals C numerically
T_cross = C * 1.0                                     # kelvin
gamma_at_cross = coupling_parameter(eq.positions, charges, T_cross)
print(f"         Γ = 1 at T_cross = {T_cross:.3f} K   (check: Γ(T_cross) = {gamma_at_cross:.6f})")
assert np.isclose(gamma_at_cross, 1.0, rtol=1e-12)

# Below ~22 mK the chain is already past Γ = 170; the Doppler-cooling
# floor for ²⁵Mg⁺ (~1 mK) sits far inside the crystalline region.
T_freeze = C / GAMMA_FREEZE
print(f"         Γ crosses 170 at T = {T_freeze * 1e3:.1f} mK")
assert 10e-3 < T_freeze < 30e-3
```

A real ²⁵Mg⁺ experiment laser-cools to roughly the Doppler limit
(~1 mK), where `Γ ≈ 3700`. The crystal is not marginally stable — it
is more than an order of magnitude colder, in Γ, than even the bulk
freezing threshold. You would have to heat it past ~22 mK to approach
that line, and to a few kelvin before thermal energy finally wins.

## Step 5 — The diagnostics bundle

[`diagnostics(positions, charges, T)`](https://github.com/uwarring82/iontrap-structure/blob/main/src/iontrap_structure/diagnostics.py)
packages the spacing, Γ, and the temperature into one frozen
`StructuralDiagnostics` record — the convenient form to log or pass
downstream. Its `coupling_parameter` field is computed by the very
same function, so the two must agree bit-for-bit.

```python
diag = diagnostics(eq.positions, charges, 1.0e-3)
print(f"Step 5 — diagnostics @ 1 mK:  Γ = {diag.coupling_parameter:.1f},  "
      f"a = {diag.mean_nn_distance * 1e6:.3f} µm,  T = {diag.temperature_kelvin * 1e3:.1f} mK")

assert diag.coupling_parameter == gamma_1mK              # same value, same function
assert np.isclose(diag.mean_nn_distance, a, rtol=0, atol=0)
assert diag.temperature_kelvin == 1.0e-3
```

## Step 6 — Visualise Γ(T)

On log-log axes the `1/T` law is a straight line of slope `−1`. We
overlay the two reference lines (`Γ = 1` and `Γ ≈ 170`) and shade the
crystalline region above freezing.

```python
fig, ax = plt.subplots(figsize=(5.6, 3.6))

# Shaded crystalline region: everything above the OCP freezing line.
ax.axhspan(GAMMA_FREEZE, gamma.max() * 1.5, color=GREEN, alpha=0.10)

ax.loglog(T_grid * 1e3, gamma, color=BLUE, linewidth=1.8, label=r"$\Gamma(T) \propto 1/T$")
ax.axhline(GAMMA_FREEZE, color=GREEN, linestyle="--", linewidth=1.0,
           label=r"$\Gamma \approx 170$ (3D OCP freezing)")
ax.axhline(1.0, color=RED, linestyle="--", linewidth=1.0,
           label=r"$\Gamma = 1$ (gas/liquid)")
ax.axvline(1.0, color=GREY, linestyle=":", linewidth=1.0)  # Doppler ~1 mK marker

ax.text(2e-3, gamma.max() * 0.25, "crystalline", color=GREEN, fontsize=9)
ax.text(2e2, 3.0, "gaseous", color=RED, fontsize=9)

ax.set_xlabel("temperature $T$ (mK)")
ax.set_ylabel(r"coupling parameter $\Gamma$")
ax.set_title("Step 6 · Γ vs T for a 5-ion ²⁵Mg⁺ chain")
ax.legend(frameon=False, fontsize=8, loc="upper right")
plt.show()
```

The blue line is the diagnostic; the dashed lines are the two physical
benchmarks; the grey dotted marker is the ~1 mK Doppler scale, sitting
comfortably inside the green crystalline band. Read top-to-bottom, the
crystal melts only after Γ has fallen through more than three decades.

!!! warning "A finite-system estimator — do not over-interpret"

    This Γ uses the **nearest-neighbour** spacing of a finite chain,
    not the density-based Wigner–Seitz radius of a bulk OCP. The
    `Γ ≈ 170` freezing value is itself a *3D bulk* result from
    [Dubin & O'Neil, *Rev. Mod. Phys.* **71**, 87 (1999)]; a finite,
    low-dimensional ion crystal has its own (size- and
    geometry-dependent) order–disorder behaviour. Treat the absolute
    crossover temperatures here as order-of-magnitude guidance, not a
    measured melting point.

## Next steps

- **[Your first ion crystal](01_first_crystal.md)** — where the
  `equilibrium` configuration reused above comes from.
- **[The ModeConfig handoff to iontrap-dynamics](02_modeconfig_handoff.md)** —
  export the chain's normal modes to the dynamics package.
- **[Mixed-species crystals](03_mixed_species.md)** — how Γ generalises
  when the ions have different masses (and the mean charge enters).
- **[Stability and the linear→zigzag transition](05_zigzag_stability.md)** —
  the *structural* counterpart to thermal melting: at fixed (cold) T,
  softening the radial confinement buckles the chain.

## Licence

This tutorial is Sail material — adaptive guidance with specific
parameter choices, not a coastline constraint. Licensed under
**CC BY-NC-SA 4.0** per [`docs/LICENCE`](https://github.com/uwarring82/iontrap-structure/blob/main/docs/LICENCE).
