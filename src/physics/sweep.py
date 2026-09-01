"""
Directional sweep: reconstruct -> Riemann solve -> face fluxes.

One function serves both directions.  The Riemann solvers are batched over a
flat list of interfaces, so the 2D face arrays are reshaped to ``(N,)`` on the
way in and back to face shape on the way out; both reshapes are views, not
copies (see ``state.flat``).

Face arrays follow the convention of ``reconstruction`` and ``Grid2D``:
along the sweep axis, face ``i`` sits between cell ``i-1`` and cell ``i``, so
an ``(nxt, nyt)`` cell array produces ``(nxt+1, nyt)`` x-faces.

Fluxes are computed over the FULL ghost-inclusive domain.  At 512^2 with
ng=2 that is ~1.6% extra work, and it removes an entire class of off-by-one
errors — in particular the constrained-transport corner EMF needs face
values one row/column beyond the physical block.
"""

from __future__ import annotations

import torch

from .hlld import hlld_flux
from .reconstruction import reconstruct_prims
from .state import flat, unflat

_BKEY = ("Bx", "By", "Bz")


def sweep(
    prims: dict[str, torch.Tensor],
    grid,
    eos,
    axis: int,
    flux_fn=hlld_flux,
    limiter: str = "mc",
    vel_var: str = "Wv",
    Bnf: torch.Tensor | None = None,
    freeze_normal_B: bool = True,
    want_hll_aux: bool = False,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Compute face fluxes along ``axis``.

    Parameters
    ----------
    prims
        Cell-centred primitives, each ``(nxt, nyt)``.
    axis
        0 for the x sweep, 1 for the y sweep.  Passed to the solver as
        ``idir``; the solvers are verified direction-generic by
        ``tests/test_idir.py``.
    Bnf
        Face-centred NORMAL magnetic field, same shape as the output faces.
        When given it overrides the reconstructed normal component on both
        sides of every face.  This is the constrained-transport coupling:
        the normal field is single-valued at a face by construction, so it
        must not be reconstructed independently from the left and right.
    freeze_normal_B
        Zero the normal-component magnetic flux.  ``F^n(B^n)`` vanishes
        analytically, but the HLL dissipation term does not when the
        reconstructed normal field differs across the face.  Correct for
        problems with no variation along the transverse direction; replaced
        by the CT update once that is in place.

    Returns
    -------
    F
        Face fluxes, each with face shape along ``axis``.
    aux
        ``cmin``/``cmax``/``uL``/``uR``/``fL``/``fR`` plus the solver
        diagnostics, all in face shape.  The CT stage needs these to build
        the face EMFs, and the diagnostics are the run-time robustness
        report.
    """
    from . import hlld as _hlld

    L, R = reconstruct_prims(prims, axis=axis, eos=eos, limiter=limiter,
                             vel_var=vel_var)

    face_shape = L["rho"].shape
    if Bnf is not None:
        if Bnf.shape != face_shape:
            raise ValueError(
                f"Bnf shape {tuple(Bnf.shape)} != face shape {tuple(face_shape)}"
            )
        key = _BKEY[axis]
        L[key] = Bnf
        R[key] = Bnf

    fF, uF, p_star = flux_fn(flat(L), flat(R), eos, idir=axis)
    F = unflat(fF, face_shape)

    if freeze_normal_B:
        F[_BKEY[axis]] = torch.zeros_like(F[_BKEY[axis]])

    aux = {
        "p_star": p_star.reshape(face_shape),
        "diag": dict(_hlld.LAST_DIAG),
    }
    if want_hll_aux:
        # Only for emf_mode="hll", which needs the raw HLL ingredients to
        # form the transverse-field EMF independently of the chosen solver.
        # Paid only in that mode; the production "solver" mode reads the
        # transverse-B flux straight out of F.
        from .hlld import compute_srmhd_fluxes
        uL, uR, fL, fR, cmax, cmin = compute_srmhd_fluxes(
            flat(L), flat(R), eos, axis)
        aux["hll"] = {
            "uL": unflat(uL, face_shape), "uR": unflat(uR, face_shape),
            "fL": unflat(fL, face_shape), "fR": unflat(fR, face_shape),
            "cmax": cmax.reshape(face_shape), "cmin": cmin.reshape(face_shape),
        }
    return F, aux


def flux_divergence(F: dict[str, torch.Tensor], axis: int, delta: float,
                    keys) -> dict[str, torch.Tensor]:
    """``-(F[i+1] - F[i]) / delta`` on cells, from face fluxes.

    With ``nxt+1`` faces along ``axis`` this returns ``nxt`` cell values, so
    the result is defined on the full ghost-inclusive block.
    """
    nd = F[next(iter(keys))].ndim
    lo = [slice(None)] * nd
    hi = [slice(None)] * nd
    lo[axis] = slice(None, -1)
    hi[axis] = slice(1, None)
    lo, hi = tuple(lo), tuple(hi)
    return {k: -(F[k][hi] - F[k][lo]) / delta for k in keys}
