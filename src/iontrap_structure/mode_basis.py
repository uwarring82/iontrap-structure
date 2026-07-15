# SPDX-License-Identifier: MIT
"""``IonModeBasis`` payload emitter — the GT3b producer half.

Builds the versioned, serialization-neutral :class:`~iontrap_structure.results.
IonModeBasis` payload from a :class:`~iontrap_structure.results.ModeResult` (+ its
:class:`~iontrap_structure.trap.HarmonicTrap`), per the cross-repo handshake
ratified in ``iontrap-dynamics`` ``TC-gt3b-ion-symplectic-adapter.md`` (v0.3).

This repo stays **classical structural dynamics**: it emits only the mode basis
``B``, the mode frequencies ``ω_m``, the masses, and a tagged local-coordinate
reference frequency. The normal→local symplectic map ``S = diag(X, P)`` and all
phase-space (§27) conventions live entirely in ``iontrap-dynamics`` — not here.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from . import coulomb
from .results import ION_MODE_BASIS_SCHEMA_VERSION, IonModeBasis, ModeResult
from .trap import HarmonicTrap

#: The ``coordinate_frame`` wire tag. It **must equal the consumer's canonical
#: identifier verbatim** — ``iontrap-dynamics`` ``ion_modes._CANONICAL_COORDINATE_FRAME``
#: — which ``materialize_ion_mode_basis`` enforces by an **exact string match** (a
#: mismatch is a loud rejection, TC-GT3b §4; the consumer owns the identifier, the
#: producer emits it). Semantics: coordinate rows are ion-major, axis-minor —
#: ``r = axes_per_ion*i + c`` for ion ``i``, axis ``c`` — and column ``m`` of ``B``
#: aligns with ``frequencies_rad_s[m]``. The trap-frame handedness/axis labels, mode
#: ordering, and frequency↔column alignment ride in ``normalization_weighting_tags``,
#: since this token has to stay a fixed, byte-comparable identifier.
_COORDINATE_FRAME = "ion-major-axis-minor;row=axes_per_ion*i+c;col=mode"

#: Accepted ``local_reference`` gauges (TC-GT3b D3). ``"trap_curvature"`` is the
#: recommended default; ``"diagonal_hessian"`` also needs ``charges``.
_LOCAL_REFERENCE_CHOICES = ("trap_curvature", "diagonal_hessian")


def _canonicalize_signs(basis: NDArray[np.float64]) -> NDArray[np.float64]:
    """Pin the per-column eigenvector-sign gauge for reproducibility.

    The eigenvector sign (per mode/column) is a gauge freedom of the
    diagonalisation; ``eigh`` does not guarantee it is stable across platforms.
    Fix each column so its **largest-magnitude component is positive** (ties
    broken by the first index). Sign flips preserve column orthonormality, so
    the Gram matrix stays ``I``. The residual **degenerate-subspace rotation**
    is left as the eigensolver returns it (tagged ``unpinned``).
    """
    out = np.array(basis, dtype=float, copy=True)
    for m in range(out.shape[1]):
        col = out[:, m]
        pivot = int(np.argmax(np.abs(col)))
        if col[pivot] < 0.0:
            out[:, m] = -col
    return out


def _local_reference_frequencies(
    local_reference: str,
    modes: ModeResult,
    trap: HarmonicTrap,
    charges: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    """Per-coordinate reference frequency ``ω_local,j`` for the chosen gauge."""
    masses = np.asarray(modes.masses, dtype=float)

    if local_reference == "trap_curvature":
        # (i) bare single-ion trap curvature: the secular frequency of axis c for
        # every ion. Row 3i+c holds axis c, so tiling omega=[wx,wy,wz] per ion
        # lands trap.omega[c] at each row 3i+c. No Coulomb.
        return np.ascontiguousarray(np.tile(trap.omega, modes.n_ions), dtype=float)

    if local_reference == "diagonal_hessian":
        # (ii) sqrt(H_jj / m_j) at equilibrium — includes Coulomb curvature.
        if charges is None:
            raise ValueError(
                "local_reference='diagonal_hessian' needs the ion charges "
                "(Coulomb curvature enters H); pass charges= (the same charges "
                "used for the equilibrium the modes were computed about)."
            )
        charges = np.asarray(charges, dtype=float)
        if charges.shape != (modes.n_ions,):
            raise ValueError(f"charges must have shape ({modes.n_ions},), got {charges.shape}.")
        hess = trap.hessian(masses) + coulomb.hessian(modes.positions, charges)
        m3 = np.repeat(masses, 3)
        diag = np.diag(hess) / m3
        # Tiny round-off negatives can appear even for a PSD Hessian. Use the
        # same spectrum-scaled tolerance as ``normal_modes`` so legitimate stable
        # configurations do not fail on numerical noise.
        scale = max(float(np.max(np.abs(diag))), 1.0)
        tol = 1e-9 * scale
        if float(np.min(diag)) < -tol:
            # A defocused diagonal coordinate (Coulomb defocusing beats the trap
            # on that axis) has no real diagonal reference frequency. The full
            # Hessian is materially non-PSD; the *diagonal* gauge isn't defined.
            raise ValueError(
                "local_reference='diagonal_hessian' has a negative diagonal "
                f"curvature (min H_jj/m_j = {float(np.min(diag)):.3e} rad^2/s^2): "
                "sqrt() would be imaginary. Use local_reference='trap_curvature' "
                "for this configuration, or supply a canonical reference."
            )
        return np.ascontiguousarray(np.sqrt(np.clip(diag, 0.0, None)), dtype=float)

    raise ValueError(
        f"unknown local_reference {local_reference!r}; expected one of {_LOCAL_REFERENCE_CHOICES}."
    )


def ion_mode_basis(
    modes: ModeResult,
    trap: HarmonicTrap,
    *,
    local_reference: str = "trap_curvature",
    charges: NDArray[np.float64] | None = None,
) -> IonModeBasis:
    """Emit the versioned :class:`IonModeBasis` payload for the GT3b handshake.

    Parameters
    ----------
    modes
        Normal modes about a **stable** equilibrium (``normal_modes`` already
        raises on non-positive ``ω²``, so all frequencies are real and ``> 0``).
    trap
        The :class:`HarmonicTrap` the modes were computed in — defines the
        right-handed ``(x, y, z)`` frame and the default local-reference gauge.
    local_reference
        The ``ω_local,j`` gauge (TC-GT3b D3), recorded in the tags:

        - ``"trap_curvature"`` (default): the bare single-ion secular frequency
          of each axis, ``ω_local,3i+c = trap.omega[c]``. No Coulomb.
        - ``"diagonal_hessian"``: ``√(H_jj / m_j)`` at equilibrium, including
          Coulomb curvature. **Requires** ``charges``.

        The choice sets only a **local single-mode squeezing gauge** on the
        consumer side (it changes the covariance representation / occupation /
        ``T_eff`` but **not** ion-cut ``E_N``), so it can be revisited without
        breaking entanglement results.
    charges
        Per-ion charge (C), shape ``(N,)`` — required only for
        ``local_reference="diagonal_hessian"``. Must match the equilibrium the
        modes were computed about.

    Returns
    -------
    IonModeBasis
        The payload; call :meth:`IonModeBasis.asdict` for the plain-array wire
        form. Re-running on the **same** ``modes``/``trap`` reproduces the payload
        exactly (the eigenvector sign is canonicalised). The residual
        degenerate-subspace rotation is left as the eigensolver returns it (tagged
        ``unpinned``), so payloads from two *independent* diagonalisations of a
        degenerate spectrum may still differ within that subspace.
    """
    three_n = modes.n_modes

    # B[3i+c, m] = eigenvectors[m, i, c]. eigenvectors is (3N, N, 3); reshaping to
    # (3N, 3N) row-major makes row m the flattened mode-m eigenvector in the
    # ion-major/axis-minor layout [x0,y0,z0,x1,...]; transpose puts modes on the
    # columns and coordinates on the rows, matching the wire invariant.
    basis = np.ascontiguousarray(np.asarray(modes.eigenvectors, dtype=float).reshape(three_n, three_n).T)
    basis = _canonicalize_signs(basis)

    frequencies = np.ascontiguousarray(np.asarray(modes.frequencies_rad_s, dtype=float))
    masses = np.ascontiguousarray(np.asarray(modes.masses, dtype=float))
    local = _local_reference_frequencies(local_reference, modes, trap, charges)

    # Producer guarantee, enforced at the source: the consumer's normal→local S-map
    # puts ω_m and ω_local,j in denominators (X ∝ √(ω_local/ω_m), P ∝ √(ω_m/ω_local)),
    # so every frequency must be finite and strictly positive. normal_modes clips
    # soft/zero modes to 0 and raises only on a *materially* negative eigenvalue — a
    # critical-point (e.g. exactly at the linear→zigzag transition) or hand-built
    # ModeResult can therefore still carry a zero frequency. Reject it here with a
    # clear message rather than emitting a payload the consumer must bounce.
    if not (np.all(np.isfinite(frequencies)) and np.all(frequencies > 0.0)):
        raise ValueError(
            "ion_mode_basis: all mode frequencies_rad_s must be finite and strictly "
            f"positive (got min {float(np.min(frequencies)):.3e} rad/s); a zero/soft or "
            "non-finite mode cannot enter the normal→local S-map."
        )
    if not np.all(np.isfinite(basis)):
        raise ValueError("ion_mode_basis: mass_weighted_eigenvectors has non-finite entries.")
    if not (np.all(np.isfinite(local)) and np.all(local > 0.0)):
        raise ValueError(
            "ion_mode_basis: local_reference_frequencies_rad_s must be finite and strictly "
            f"positive (got min {float(np.min(local)):.3e} rad/s)."
        )

    tags: dict[str, object] = {
        "mass_symmetrised": True,
        "per_mode_unit_norm": True,
        "orthonormal_columns_gram_identity": True,
        "coordinate_axes": "x,y,z",
        "trap_frame_handedness": "right_handed",
        "frequency_column_alignment": "column_m_matches_frequencies_rad_s_m",
        "mode_ordering": "ascending_omega_squared",
        "eigenvector_sign_convention": "largest_abs_component_positive",
        "degenerate_subspace_rotation": "unpinned_eigensolver_order",
        "local_reference": local_reference,
    }

    return IonModeBasis(
        schema_version=ION_MODE_BASIS_SCHEMA_VERSION,
        frequencies_rad_s=frequencies,
        mass_weighted_eigenvectors=basis,
        masses_kg=masses,
        local_reference_frequencies_rad_s=local,
        coordinate_frame=_COORDINATE_FRAME,
        normalization_weighting_tags=tags,
    )
