# Tutorials

Each tutorial is a short, runnable walkthrough of one feature. They are written
as linear scripts: the prose explains *why*, the code blocks do the work, and
embedded `assert` statements are the built-in self-test. Every tutorial can be
**run live on Google Colab** (a badge at the top of each page) — the notebooks
are generated from these Markdown pages by
[`tools/build_tutorial_notebooks.py`](https://github.com/uwarring82/iontrap-structure/blob/main/tools/build_tutorial_notebooks.py),
and CI executes every page end-to-end so the physics can't silently regress.

Read them in order the first time — later tutorials reuse the four-step pattern
(`HarmonicTrap` → `equilibrium` → `normal_modes` → analyse / export) the first
one establishes.

<div class="grid cards landing-cards" markdown>

-   __[1. Your first ion crystal](01_first_crystal.md)__

    Solve the equilibrium of a three-ion linear chain, compute its normal
    modes, and recover the textbook ω_z (centre-of-mass) and √3 ω_z (stretch)
    frequencies — the James-1998 oracle.

-   __[2. The ModeConfig handoff](02_modeconfig_handoff.md)__

    Export modes with `to_mode_configs()` and see the `ModeConfig` records that
    cross the boundary into `iontrap-dynamics` — the package's reason to exist.

-   __[3. Mixed-species crystals](03_mixed_species.md)__

    A sympathetic-cooling "sandwich" of unequal masses, and why the
    mass-symmetrised dynamical matrix is what makes the modes come out right.

-   __[4. The plasma coupling parameter Γ](04_coupling_parameter.md)__

    Quantify how crystalline a configuration is, and watch Γ cross from the
    crystalline regime into the gaseous one as temperature rises.

-   __[5. Stability and the linear→zigzag transition](05_zigzag_stability.md)__

    Soften the radial confinement, watch the lowest transverse mode go soft,
    and see `normal_modes` refuse to return real frequencies for an unstable
    chain.

</div>

## Licence

The tutorials are Sail material — adaptive guidance with specific worked
parameter choices. Licensed under **CC BY-NC-SA 4.0** per
[`docs/LICENCE`](https://github.com/uwarring82/iontrap-structure/blob/main/docs/LICENCE).
