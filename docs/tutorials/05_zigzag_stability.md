# Tutorial 5 — Stability and the linear→zigzag transition

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uwarring82/iontrap-structure/blob/main/docs/tutorials/notebooks/05_zigzag_stability.ipynb) — run every step live in your browser, no install needed. The notebook is generated from this page by [`tools/build_tutorial_notebooks.py`](https://github.com/uwarring82/iontrap-structure/blob/main/tools/build_tutorial_notebooks.py).

**Goal.** By the end of this tutorial you will have watched a normal
mode go *soft* as you relax the radial confinement of a five-ion
chain, located the linear→zigzag instability where that mode crosses
zero, and seen `normal_modes()` refuse to return imaginary
frequencies once the on-axis chain stops being a minimum. The whole
exercise is built from two functions you already met:
`equilibrium()` and `normal_modes()`.

**Expected time.** ~10 min reading; ~2 s runtime.

**Prerequisites.** A working install (`pip install -e ".[dev,plot]"`
in the repo root). Tutorials [01](01_first_crystal.md) and
[02](02_modeconfig_handoff.md) introduce `HarmonicTrap`,
`equilibrium()`, and `normal_modes()` — this one assumes you have
seen them.

---

## The scenario

Five ²⁵Mg⁺ ions in a linear Paul trap with a **fixed** axial secular
frequency `ω_z / 2π = 1 MHz`. We hold the two radial frequencies
equal, `ω_x = ω_y ≡ ω_r`, and sweep the **aspect ratio** `ω_r / ω_z`
downward from strong radial confinement (4) toward weak (2.5).

A linear chain on the trap axis is the ground state only while the
radial restoring force is stiff enough to win against the mutual
Coulomb repulsion that pushes the ions sideways. As `ω_r / ω_z`
falls, the lowest *transverse* (radial) mode — the zigzag mode, in
which neighbouring ions displace in opposite radial directions —
**softens** toward zero frequency. At a critical aspect ratio the
chain buckles into a planar zigzag; below it, the on-axis
configuration is a saddle, not a minimum. We will find that critical
ratio numerically and confirm the library guards against the
unphysical "imaginary frequency" you would otherwise compute there.

## Step 1 — A helper that returns the lowest mode (or `'unstable'`)

We wrap the full pipeline — build trap, solve for equilibrium from
an on-axis guess, diagonalise the Hessian — in one helper of the
aspect ratio `ω_r / ω_z`. The key move is the `try/except`: past the
instability `normal_modes()` *raises* `ValueError` rather than
return a fictitious real frequency, so we catch it and report the
string `'unstable'`.

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import atomic_mass, elementary_charge

from iontrap_structure import (
    HarmonicTrap,
    equilibrium,
    length_scale,
    linear_chain_guess,
    normal_modes,
)

# House colours.
BLUE, RED, GREEN, GREY = "#1f77b4", "#d62728", "#2ca02c", "#444444"

N = 5
M_MG = 25 * atomic_mass            # ²⁵Mg⁺ mass, kg
Q = elementary_charge              # singly ionised, C
masses = np.full(N, M_MG)
charges = np.full(N, Q)
WZ = 2 * np.pi * 1.0e6             # fixed axial secular frequency, rad/s


def lowest_mode(ratio):
    """Lowest normal-mode frequency (rad/s) for ω_r/ω_z = ``ratio``.

    Returns the string ``'unstable'`` if the on-axis chain is a saddle
    (``normal_modes`` raises because the dynamical matrix is not PSD).
    """
    wr = ratio * WZ
    trap = HarmonicTrap(wx=wr, wy=wr, wz=WZ)
    # Seed the root find from the on-axis linear chain (dimensionless guess
    # rescaled to metres via the natural length ℓ).
    ell = length_scale(trap, M_MG, Q)
    guess = linear_chain_guess(N) * ell
    eq = equilibrium(trap=trap, masses=masses, charges=charges, initial_positions=guess)
    try:
        modes = normal_modes(eq)
    except ValueError as exc:
        assert "not positive semi-definite" in str(exc)
        return "unstable"
    return float(modes.frequencies_rad_s[0])


# Sanity check the two extremes before scanning.
strong = lowest_mode(4.0)
weak = lowest_mode(2.0)
print(f"ω_r/ω_z = 4.0  →  lowest mode = {strong / (2 * np.pi) * 1e-6:.4f} MHz")
print(f"ω_r/ω_z = 2.0  →  {weak}")
assert isinstance(strong, float) and weak == "unstable"
```

!!! note "Why `equilibrium()` still converges past the instability"

    The on-axis chain is *always* a stationary point of the
    potential: every ion sits at `(0, 0, z_i)`, so the radial forces
    cancel by symmetry and `∂E/∂r = 0` holds regardless of `ω_r`.
    What changes is the **curvature**. Above the critical ratio the
    on-axis chain is a minimum; below it, it is a saddle — still a
    valid root of the force balance, so `equilibrium()` reports
    `converged=True`, but no longer a stable structure.

## Step 2 — Scan the aspect ratio downward

We sweep `ω_r / ω_z` from 4.0 down through the transition, with a
finer grid near the onset so the soft-mode drop is well resolved.
Each stable point contributes a lowest-mode frequency; the unstable
points contribute the sentinel.

```python
ratios = np.concatenate([
    np.round(np.arange(4.0, 2.69, -0.1), 2),   # coarse, deep in the linear regime
    np.array([2.65, 2.60, 2.55, 2.52, 2.50, 2.49, 2.45, 2.40]),  # fine, across onset
])

results = {r: lowest_mode(float(r)) for r in ratios}

stable_ratios = np.array([r for r in ratios if results[r] != "unstable"])
stable_freqs_mhz = np.array(
    [results[r] / (2 * np.pi) * 1e-6 for r in ratios if results[r] != "unstable"]
)
unstable_ratios = np.array([r for r in ratios if results[r] == "unstable"])

print(f"scanned {len(ratios)} aspect ratios: "
      f"{len(stable_ratios)} stable, {len(unstable_ratios)} unstable")
print(f"lowest stable mode at ω_r/ω_z = {stable_ratios.min():.2f}: "
      f"{stable_freqs_mhz[stable_ratios.argmin()]:.4f} MHz")
```

At large `ω_r / ω_z` the lowest mode sits flat at `1.0 MHz`: that is
the **axial** centre-of-mass mode (frequency `ω_z`, independent of
`ω_r`), which the stiff radial modes sit far above. As `ω_r`
decreases the radial zigzag mode dives down through the axial COM
mode and becomes the lowest mode of the spectrum — that is the
branch we are about to watch soften.

## Step 3 — The softening branch is monotonic

Restrict to the part of the scan where the radial zigzag mode is the
lowest mode (here `ω_r / ω_z ≤ 2.65`). Across that branch, lowering
the radial confinement can only reduce the radial restoring
curvature, so the soft-mode frequency must **decrease monotonically**
as `ω_r / ω_z` decreases — softening, never stiffening.

```python
soft_mask = stable_ratios <= 2.65
soft_ratios = stable_ratios[soft_mask]
soft_freqs = stable_freqs_mhz[soft_mask]

# stable_ratios is descending, so soft_freqs should be strictly decreasing.
order = np.argsort(soft_ratios)[::-1]           # descending ω_r/ω_z
soft_ratios_desc = soft_ratios[order]
soft_freqs_desc = soft_freqs[order]
diffs = np.diff(soft_freqs_desc)

print("ω_r/ω_z :", np.round(soft_ratios_desc, 3))
print("soft f/2π (MHz):", np.round(soft_freqs_desc, 4))
assert np.all(diffs < 0.0), "soft mode must drop as radial confinement weakens"
assert soft_freqs_desc[0] > 0.8 and soft_freqs_desc[-1] < 0.3
```

!!! note "The physics of a soft mode"

    The zigzag mode's `ω²` is the curvature of the potential along
    the symmetric transverse-buckling direction. It is the sum of a
    positive trap term `∝ ω_r²` and a *negative* Coulomb term: the
    repulsion between neighbours that already want to splay apart. As
    `ω_r` shrinks the trap term loses, `ω² → 0⁺`, and the mode goes
    soft. At the critical ratio `ω² = 0` and the chain is marginally
    stable; below it `ω² < 0` and the on-axis chain is a saddle along
    that direction. This is the continuous (second-order)
    linear→zigzag transition analysed by Fishman *et al.*, *Phys.
    Rev. B* **77**, 064111 (2008).

## Step 4 — Below critical, `normal_modes()` raises

We picked `ω_r / ω_z = 2.0` in Step 1 and saw `'unstable'`. Here we
let the exception surface so you can read it: the dynamical matrix
has a genuinely negative eigenvalue, `normal_modes()` refuses to
take its square root, and raises with a message naming the cause.

```python
wr_unstable = 2.0 * WZ
trap_unstable = HarmonicTrap(wx=wr_unstable, wy=wr_unstable, wz=WZ)
ell = length_scale(trap_unstable, M_MG, Q)
eq_unstable = equilibrium(
    trap=trap_unstable,
    masses=masses,
    charges=charges,
    initial_positions=linear_chain_guess(N) * ell,
)

# equilibrium() still converged — the on-axis chain is a (saddle) stationary point.
print(f"ω_r/ω_z = 2.0  →  equilibrium converged = {eq_unstable.converged}, "
      f"max |force| = {eq_unstable.residual_force:.2e} N")
assert eq_unstable.converged

raised = False
try:
    normal_modes(eq_unstable)
except ValueError as exc:
    raised = True
    message = str(exc)
    print(f"normal_modes() raised: {message.splitlines()[0]}")
assert raised, "an on-axis chain past the zigzag instability must raise"
assert "not positive semi-definite" in message
```

!!! note "Why guard instead of clip"

    `normal_modes()` *does* clip eigenvalues that are negative only
    by round-off (order machine-ε · ‖D‖) up to zero — those are the
    genuine translational/rotational zero modes. But it refuses to
    clip a *materially* negative eigenvalue: doing so would invent a
    real frequency for a direction along which the configuration is
    actually unstable, silently hiding the zigzag transition. A loud
    `ValueError` is the correct behaviour for a saddle.

## Step 5 — Locate the onset and plot the softening curve

The critical ratio lies between the lowest stable ratio and the
highest unstable one. We report that bracket, then plot the
lowest-mode frequency against `ω_r / ω_z`, shading the region below
onset where the on-axis chain is unstable.

```python
last_stable = stable_ratios.min()
first_unstable = unstable_ratios.max()
critical_ratio = 0.5 * (last_stable + first_unstable)
print(f"linear→zigzag onset: last stable ω_r/ω_z = {last_stable:.2f}, "
      f"first unstable = {first_unstable:.2f}")
print(f"approximate critical ratio ≈ {critical_ratio:.3f}")
assert first_unstable < last_stable                 # onset is bracketed
assert 2.4 < critical_ratio < 2.6

fig, ax = plt.subplots(figsize=(5.2, 3.4))
ax.axvspan(unstable_ratios.min() - 0.05, last_stable, color=RED, alpha=0.12,
           label="on-axis chain unstable")
ax.axvline(last_stable, color=GREY, linestyle=":", linewidth=1.0)
ax.plot(stable_ratios, stable_freqs_mhz, "o-", color=BLUE, markersize=4,
        label="lowest normal mode")
ax.scatter([last_stable], [stable_freqs_mhz[stable_ratios.argmin()]],
           color=RED, zorder=5, s=40, label="last stable point")
ax.axhline(1.0, color=GREEN, linestyle="--", linewidth=1.0,
           label=r"axial COM ($\omega_z$)")
ax.set_xlabel(r"aspect ratio $\omega_r / \omega_z$")
ax.set_ylabel(r"lowest mode $\,\omega/2\pi$ (MHz)")
ax.set_title("Step 5 · soft mode of the 5-ion chain vs radial confinement")
ax.set_ylim(0.0, 1.15)
ax.legend(frameon=False, fontsize=8, loc="lower right")
plt.show()
```

The blue curve traces the lowest mode: flat at the axial COM
frequency (green dashed, `ω_z`) while the radial modes are stiff,
then peeling off and plunging toward zero as the radial zigzag mode
softens. The red marker is the last aspect ratio at which the
on-axis chain is still a minimum; cross into the shaded region and
`normal_modes()` raises. For these parameters the onset sits near
`ω_r / ω_z ≈ 2.5` — close to the `O(1)` value set by the chain's
length and ion number, exactly as the Fishman *et al.* scaling
predicts.

## Next steps

- **[Your first ion crystal](01_first_crystal.md)** — build a single
  equilibrium configuration from scratch and read off its positions.
- **[The ModeConfig handoff](02_modeconfig_handoff.md)** — export the
  stable normal modes computed here to `iontrap-dynamics` for a
  quantum-dynamics simulation.
- **[Mixed-species crystals](03_mixed_species.md)** — repeat the
  stability story with two ion species and watch the mode structure
  rearrange.
- **[The plasma coupling parameter Γ](04_coupling_parameter.md)** —
  quantify *how* crystalline a configuration is with the structural
  diagnostics layer.

## Licence

This tutorial is Sail material — adaptive guidance with specific
parameter choices, not a coastline constraint. Licensed under
**CC BY-NC-SA 4.0** per [`docs/LICENCE`](https://github.com/uwarring82/iontrap-structure/blob/main/docs/LICENCE).
