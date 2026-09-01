"""
Constrained transport (Gardiner & Stone 2005 upwinded corner EMF).

Layout
------
Cells ``(i, j)`` on ``(nxt, nyt)``.  Index ``i`` on a staggered array denotes
the LOWER face of cell ``i``:

    Bxf : (nxt+1, nyt)     lower-x faces
    Byf : (nxt, nyt+1)     lower-y faces
    Ez  : (nxt+1, nyt+1)   corners, Ez[i, j] at (x_{i-1/2}, y_{j-1/2})

Sign conventions
----------------
``E = -v x B``, so in 2D the only surviving component is

    E^z = v^y B^x - v^x B^y

and the induction equation ``dB/dt = -curl E`` becomes

    dBx/dt = -dEz/dy ,   dBy/dt = +dEz/dx

The transverse-field fluxes returned by the Riemann solver are exactly
plus/minus this EMF:

    F^x(By) = By vx - Bx vy = -E^z      ->  Efx = -F^x(By)
    F^y(Bx) = Bx vy - By vx = +E^z      ->  Efy = +F^y(Bx)

Why div(B) stays at machine precision
-------------------------------------
Each corner value is read by exactly the four faces surrounding it, with
opposite signs in the two cells sharing each face, so ``div(curl E)``
telescopes to bit-zero per cell — provided every cell that touches an edge
uses THE SAME floating-point EMF value there.  On a unigrid that is
automatic.  (It is what AMR refluxing exists to restore at coarse-fine
boundaries.)
"""

from __future__ import annotations

import torch


def cell_centred_B(Bxf: torch.Tensor, Byf: torch.Tensor
                   ) -> tuple[torch.Tensor, torch.Tensor]:
    """Arithmetic average of the two bounding faces; shape ``(nxt, nyt)``.

    Face-centred B is the primary evolved variable; the cell-centred value
    is derived from it and recomputed every stage.  It is what feeds
    conserved-to-primitive inversion, reconstruction and the wave speeds.
    """
    Bcx = 0.5 * (Bxf[:-1, :] + Bxf[1:, :])
    Bcy = 0.5 * (Byf[:, :-1] + Byf[:, 1:])
    return Bcx, Bcy


def emf_cell(Bcx: torch.Tensor, Bcy: torch.Tensor,
             vx: torch.Tensor, vy: torch.Tensor) -> torch.Tensor:
    """Cell-centred ``E^z = v^y B^x - v^x B^y``."""
    return Bcx * vy - Bcy * vx


def face_emf_x(Fx: dict, aux: dict | None = None, mode: str = "solver"
               ) -> torch.Tensor:
    """``E^z`` on x-faces, shape ``(nxt+1, nyt)``."""
    if mode == "solver":
        return -Fx["By"]
    if mode == "hll":
        if aux is None or "hll" not in aux:
            raise ValueError("emf_mode='hll' requires sweep(want_hll_aux=True)")
        h = aux["hll"]
        return -_hll(h["fL"]["By"], h["fR"]["By"],
                     h["uL"]["By"], h["uR"]["By"], h["cmin"], h["cmax"])
    raise ValueError(f"unknown emf mode {mode!r}")


def face_emf_y(Fy: dict, aux: dict | None = None, mode: str = "solver"
               ) -> torch.Tensor:
    """``E^z`` on y-faces, shape ``(nxt, nyt+1)``."""
    if mode == "solver":
        return Fy["Bx"]
    if mode == "hll":
        if aux is None or "hll" not in aux:
            raise ValueError("emf_mode='hll' requires sweep(want_hll_aux=True)")
        h = aux["hll"]
        return _hll(h["fL"]["Bx"], h["fR"]["Bx"],
                    h["uL"]["Bx"], h["uR"]["Bx"], h["cmin"], h["cmax"])
    raise ValueError(f"unknown emf mode {mode!r}")


def _hll(fL, fR, uL, uR, cmin, cmax):
    """HLL average with cmin, cmax >= 0 (so S_L = -cmin, S_R = +cmax)."""
    return (cmax * fL + cmin * fR - cmax * cmin * (uR - uL)) / (cmax + cmin)


def _upwind_sign(F: torch.Tensor, rel_tol: float) -> torch.Tensor:
    """``sign(F)`` with a deadband around zero.

    ``sign`` is discontinuous at 0, so it turns an arbitrarily small
    perturbation of the mass flux into an O(1) change in the upwind weights.
    That is not hypothetical here: in the rotor's static ambient medium the
    physical mass flux is exactly zero, but the computed flux is O(1e-12)
    round-off left over from the HLL dissipation term ``-cmax*cmin*(uR-uL)``,
    with no definite sign.  Taking ``sign`` of that noise selected a
    direction at ~97% of faces and destroyed the discrete pi-rotation
    symmetry of the rotor (measured: 1.3e-01, versus 3e-15 for the
    non-upwinded variant).

    Suppressing the selector where the flux is negligible restores the
    centred limit exactly where upwinding carries no information anyway.
    The scale is the largest mass flux in the sweep, so the test is
    dimensionless and independent of the problem's units.
    """
    scale = F.abs().max()
    return torch.where(F.abs() > rel_tol * scale, torch.sign(F),
                       torch.zeros_like(F))


def corner_emf(Ec: torch.Tensor, Efx: torch.Tensor, Efy: torch.Tensor,
               Fx_rho: torch.Tensor, Fy_rho: torch.Tensor,
               upwind: bool = True, sign_rel_tol: float = 1e-9
               ) -> torch.Tensor:
    """Gardiner-Stone corner EMF; shape ``(nxt+1, nyt+1)``.

    The first term is the plain four-face arithmetic average — i.e. the
    Balsara-Spicer / Toth flux-CT scheme, recovered exactly by
    ``upwind=False``.  The remaining terms are the Gardiner-Stone upwinded
    transverse-derivative corrections, selected by the sign of the mass flux.

    The upwind selector is ``_upwind_sign``: ``sign`` with a deadband (see
    its docstring for why the deadband is required, not merely defensive).
    Note ``torch.sign`` and not ``copysign``: copysign(1, 0) returns +1 and
    would fix a handedness at an exact zero crossing.

    Design property (asserted in tests): if ``E^z`` varies only with y, the
    result reduces EXACTLY to the y-face value, and symmetrically for x.
    """
    nxt, nyt = Ec.shape
    Ez = torch.zeros(nxt + 1, nyt + 1, dtype=Ec.dtype, device=Ec.device)

    # Corners i = 1..nxt-1, j = 1..nyt-1; all physical corners lie inside that
    # range for ng >= 1.  The bounds are written out numerically rather than
    # as reusable slice objects because the arrays have DIFFERENT shapes
    # ((nxt,nyt), (nxt+1,nyt), (nxt,nyt+1)) — a shared slice like [1:-1] means
    # a different index range on each of them.
    I, Im = slice(1, nxt), slice(0, nxt - 1)        # index i, i-1
    J, Jm = slice(1, nyt), slice(0, nyt - 1)        # index j, j-1

    Ezyf = Efy[I, J]        # y-face EMF at (x_i,     y_{j-1/2})
    Ezyfm = Efy[Im, J]      #                (x_{i-1}, y_{j-1/2})
    Ezxf = Efx[I, J]        # x-face EMF at (x_{i-1/2}, y_j)
    Ezxfm = Efx[I, Jm]      #                (x_{i-1/2}, y_{j-1})

    EzNE = Ec[I, J]
    EzNW = Ec[Im, J]
    EzSE = Ec[I, Jm]
    EzSW = Ec[Im, Jm]

    Eavg = 0.25 * ((Ezyf + Ezyfm) + (Ezxf + Ezxfm))

    if upwind:
        Sx = _upwind_sign(Fx_rho, sign_rel_tol)[I, J]
        Sxm = _upwind_sign(Fx_rho, sign_rel_tol)[I, Jm]
        dEdyN = (1.0 - Sx) * (EzNE - Ezyf) + (1.0 + Sx) * (EzNW - Ezyfm)
        dEdyS = (1.0 - Sxm) * (Ezyf - EzSE) + (1.0 + Sxm) * (Ezyfm - EzSW)
        dEdy = 0.125 * (dEdyS - dEdyN)

        Sy = _upwind_sign(Fy_rho, sign_rel_tol)[I, J]
        Sym = _upwind_sign(Fy_rho, sign_rel_tol)[Im, J]
        dEdxE = (1.0 - Sy) * (EzNE - Ezxf) + (1.0 + Sy) * (EzSE - Ezxfm)
        dEdxW = (1.0 - Sym) * (Ezxf - EzNW) + (1.0 + Sym) * (Ezxfm - EzSW)
        dEdx = 0.125 * (dEdxW - dEdxE)

        Ez[I, J] = Eavg + (dEdy + dEdx)
    else:
        Ez[I, J] = Eavg
    return Ez


def ct_rhs(Ez: torch.Tensor, dx: float, dy: float
           ) -> tuple[torch.Tensor, torch.Tensor]:
    """``(dBxf/dt, dByf/dt) = (-dEz/dy, +dEz/dx)`` on the staggered faces."""
    dBxf = (Ez[:, :-1] - Ez[:, 1:]) / dy      # (nxt+1, nyt)
    dByf = (Ez[1:, :] - Ez[:-1, :]) / dx      # (nxt,   nyt+1)
    return dBxf, dByf


def ct_update(Bxf: torch.Tensor, Byf: torch.Tensor, Ez: torch.Tensor,
              dt: float, dx: float, dy: float, coef: float = 1.0
              ) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance the staggered field by ``coef*dt`` times the curl of ``Ez``."""
    dBxf, dByf = ct_rhs(Ez, dx, dy)
    return Bxf + coef * dt * dBxf, Byf + coef * dt * dByf


def div_b(Bxf: torch.Tensor, Byf: torch.Tensor, dx: float, dy: float
          ) -> torch.Tensor:
    """Cell-centred ``div B``; shape ``(nxt, nyt)``."""
    return (Bxf[1:, :] - Bxf[:-1, :]) / dx + (Byf[:, 1:] - Byf[:, :-1]) / dy
