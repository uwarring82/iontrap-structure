# iontrap-structure

**Classical structural dynamics of trapped-ion crystals** — equilibrium
configurations and normal modes, with a clean export to the
[`iontrap-dynamics`](https://github.com/uwarring82/iontrap-dynamics)
`ModeConfig` contract.

> **Status: pre-alpha, first slice.** This is the *producer* side of the
> boundary that `iontrap-dynamics` deliberately delegates (its
> `modes.py` consumes externally-supplied normal modes; Design Principle 2 bars
> it from deriving mode structure itself). `iontrap-structure` computes that
> structure and hands it back.

## Scope

The package is **MIT-licensed and clean-room**: the physics is implemented from
published papers, not ported from copyleft or unlicensed code. It interoperates
with existing tools (e.g. pylion/LAMMPS as an out-of-process MD cross-check) but
depends on nothing copyleft at runtime.

**First slice (this release):**

- Anisotropic harmonic (pseudopotential) trap model.
- Pairwise Coulomb energy / gradient / analytic Hessian.
- Equilibrium-configuration solver (dimensionless force balance, analytic Jacobian).
- Normal modes via the **mass-symmetrised** dynamical matrix `M⁻¹ᐟ² H M⁻¹ᐟ²`,
  exported as `ModeConfig`-compatible records (CONVENTIONS §10/§11: per-mode
  eigenvectors `(N, 3)` with `Σᵢ‖bᵢ‖² = 1`, frequencies in rad·s⁻¹).
- A decoupled, engine-agnostic analysis layer (plasma coupling parameter Γ;
  richer order parameters delegate to the optional [`freud`] dependency).

**Explicitly out of scope (v0.1):** large-N MD and the full eigensolve at
N ≳ 10³, neutral-atom backgrounds / hybrid atom-ion collisions, reaction
chemistry, electrode/BEM field solvers, and any public DSL. See the origin task
card for the rationale and the opening backlog.

## Quick start

```python
import numpy as np
from scipy.constants import atomic_mass, elementary_charge
from iontrap_structure import HarmonicTrap, equilibrium, normal_modes, coupling_parameter

trap = HarmonicTrap(wx=2*np.pi*10e6, wy=2*np.pi*10e6, wz=2*np.pi*1e6)  # linear chain
masses = np.full(3, 25 * atomic_mass)   # ²⁵Mg⁺
charges = np.full(3, elementary_charge)

eq = equilibrium(trap=trap, masses=masses, charges=charges)
modes = normal_modes(eq)

configs = modes.to_mode_configs()        # -> iontrap_dynamics.ModeConfig (if installed)
gamma = coupling_parameter(eq.positions, charges, temperature_kelvin=1e-3)
```

## Install

```bash
pip install -e ".[dev]"            # core + test toolchain
pip install -e ".[dev,analysis]"   # + freud order parameters (Python ≥3.12)
pip install -e ".[dev,interop]"    # + iontrap-dynamics ModeConfig export
```

## Validation

The first slice is validated against the **James-1998** analytic linear chain
(equilibrium positions; the universal axial COM = ω_z and stretch = √3 ω_z
modes) and asserts the `ModeConfig` contract (normalisation, orthonormality,
positive frequencies) plus a finite-difference check of the analytic Hessian.

## Provenance

Bootstrapped from a frozen deliberation/survey record in `iontrap-dynamics`:
[`task cards/TC-structural-dynamics-foundation-survey.md`](https://github.com/uwarring82/iontrap-dynamics/blob/main/task%20cards/TC-structural-dynamics-foundation-survey.md)
(v1.0). See [`docs/PROVENANCE.md`](docs/PROVENANCE.md) for the carried-forward backlog.

[`freud`]: https://github.com/glotzerlab/freud
