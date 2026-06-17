# Getting Started

## Install

`iontrap-structure` targets Python 3.11+ and depends only on NumPy and SciPy.

```sh
python -m pip install -e ".[dev]"            # core + test toolchain
```

Optional extras:

```sh
python -m pip install -e ".[dev,plot]"       # + matplotlib (for the tutorials)
python -m pip install -e ".[dev,docs]"       # + mkdocs-material (to build this site)
python -m pip install -e ".[dev,interop]"    # + iontrap-dynamics (real ModeConfig export)
python -m pip install -e ".[dev,analysis]"   # + freud order parameters (Python ≥3.12)
```

## Quick start

A linear chain of three ²⁵Mg⁺ ions: solve the equilibrium, compute the normal
modes, export them, and read off the plasma coupling parameter.

```python
import numpy as np
from scipy.constants import atomic_mass, elementary_charge

from iontrap_structure import (
    HarmonicTrap,
    equilibrium,
    normal_modes,
    coupling_parameter,
)

# Strong radial confinement (ω_x, ω_y ≫ ω_z) → the ions form a chain on z.
trap = HarmonicTrap(wx=2 * np.pi * 10e6, wy=2 * np.pi * 10e6, wz=2 * np.pi * 1e6)
masses = np.full(3, 25 * atomic_mass)        # ²⁵Mg⁺
charges = np.full(3, elementary_charge)

eq = equilibrium(trap=trap, masses=masses, charges=charges)
modes = normal_modes(eq)

configs = modes.to_mode_configs()            # → iontrap_dynamics.ModeConfig (if installed)
gamma = coupling_parameter(eq.positions, charges, temperature_kelvin=1e-3)

print("converged:", eq.converged)
print("lowest two mode frequencies / 2π (MHz):",
      np.round(modes.frequencies_rad_s[:2] / (2 * np.pi) / 1e6, 4))
print("Γ at 1 mK:", round(gamma))
```

The two lowest modes come out at `1.0` and `√3 ≈ 1.732` MHz — the universal
axial centre-of-mass and stretch modes of a linear chain (James 1998).

## What you can import today

The full v0.1 public surface, all from the top-level package:

- **`HarmonicTrap`** — anisotropic harmonic trap.
- **`equilibrium`, `length_scale`, `linear_chain_guess`** — equilibrium solver
  and helpers; returns `EquilibriumResult`.
- **`normal_modes`** — normal-mode analysis; returns `ModeResult` with
  `to_mode_configs()`.
- **`coupling_parameter`, `mean_nearest_neighbour_distance`, `diagnostics`** —
  the structural-diagnostics layer; `diagnostics()` returns
  `StructuralDiagnostics`.
- **Result records** — `EquilibriumResult`, `ModeResult`, `ModeConfigLike`,
  `StructuralDiagnostics`.

See the [API reference](api.md) for full signatures, generated from the
docstrings.

## Read in this order

1. [Overview](overview.md) — the producer/consumer boundary and the physics.
2. [Conventions](conventions.md) — units and the §10/§11 mode contract.
3. The [Tutorials](tutorials/index.md) — each one runnable on Colab.

## Build the docs site locally

```sh
python -m pip install -e ".[dev,docs]"
mkdocs build --strict        # or: mkdocs serve
```

The site configuration lives in `mkdocs.yml`; custom presentation styles live
in `docs/stylesheets/extra.css`.

## Endorsement Marker

Local candidate framework under active stewardship. No external endorsement is
implied.
