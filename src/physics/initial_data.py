"""
Shocktube initial conditions for 1D SR-MHD.

Each problem returns (primL, primR) dicts with scalar tensors,
plus an interface position x_interface.

Available problems:
    brio_wu       — classic SR-MHD Brio-Wu variant
    komissarov_1  — Komissarov (1999) shock tube 1
    komissarov_2  — Komissarov (1999) shock tube 2
    generic       — user-specified dict

Reference: Mignone & Bodo (2006), Komissarov (1999).
"""

import torch
from typing import Tuple


Prim = dict[str, torch.Tensor]


def _prim(rho, vx, vy, vz, p, eps, Bx, By, Bz,
          dtype=torch.float64, device="cpu") -> Prim:
    """Pack scalars into a primitive state dict."""
    vals = dict(rho=rho, vx=vx, vy=vy, vz=vz, p=p, eps=eps,
                Bx=Bx, By=By, Bz=Bz)
    return {k: torch.tensor(v, dtype=dtype, device=device) for k, v in vals.items()}


def brio_wu(
    dtype=torch.float64, device="cpu"
) -> Tuple[Prim, Prim, float]:
    """
    SR-MHD Brio-Wu shocktube (Mignone & Bodo 2006, Table 1, problem 1).
    gamma = 2, K = 1 (cold polytrope used here as seed; p = K rho^gamma).
    Bx = 0.5 (constant), By flips sign at interface.
    """
    pL = _prim(rho=1.0,  vx=0.0, vy=0.0, vz=0.0,
               p=1.0,    eps=1.0, Bx=0.5, By= 1.0, Bz=0.0,
               dtype=dtype, device=device)
    pR = _prim(rho=0.125, vx=0.0, vy=0.0, vz=0.0,
               p=0.1,    eps=0.8, Bx=0.5, By=-1.0, Bz=0.0,
               dtype=dtype, device=device)
    return pL, pR, 0.5


def komissarov_1(
    dtype=torch.float64, device="cpu"
) -> Tuple[Prim, Prim, float]:
    """
    Komissarov (1999) shock tube 1.
    """
    pL = _prim(rho=1.0,  vx=0.0, vy=0.0, vz=0.0,
               p=1.0,    eps=1.5, Bx=1.0, By=1.0, Bz=0.0,
               dtype=dtype, device=device)
    pR = _prim(rho=0.1,  vx=0.0, vy=0.0, vz=0.0,
               p=0.1,    eps=1.5, Bx=1.0, By=-1.0, Bz=0.0,
               dtype=dtype, device=device)
    return pL, pR, 0.5


def komissarov_2(
    dtype=torch.float64, device="cpu"
) -> Tuple[Prim, Prim, float]:
    """
    Komissarov (1999) shock tube 2 — collision problem.
    """
    pL = _prim(rho=1.0,  vx= 0.9, vy=0.0, vz=0.0,
               p=1.0,    eps=1.5,  Bx=10.0, By=5.0, Bz=0.0,
               dtype=dtype, device=device)
    pR = _prim(rho=1.0,  vx=-0.9, vy=0.0, vz=0.0,
               p=1.0,    eps=1.5,  Bx=10.0, By=5.0, Bz=0.0,
               dtype=dtype, device=device)
    return pL, pR, 0.5


def generic(
    primL: dict, primR: dict,
    x_interface: float = 0.5,
    dtype=torch.float64, device="cpu",
    eos=None,
) -> Tuple[Prim, Prim, float]:
    """
    User-supplied left/right state dicts (plain Python floats / numpy scalars).
    Required keys: rho, vx, vy, vz, p, Bx, By, Bz.
    Optional key:  eps — computed from p and rho via eos if not provided.
    """
    pL = {k: torch.tensor(v, dtype=dtype, device=device) for k, v in primL.items()}
    pR = {k: torch.tensor(v, dtype=dtype, device=device) for k, v in primR.items()}
    if eos is not None:
        for prim in (pL, pR):
            if "eps" not in prim:
                prim["eps"] = eos.eps__press_rho(prim["p"], prim["rho"])
    return pL, pR, x_interface


# ── Grid initialisation ───────────────────────────────────────────────────────

def init_shocktube(
    grid,
    primL: Prim,
    primR: Prim,
    x_interface: float,
) -> dict[str, torch.Tensor]:
    """
    Fill a Grid1D with a shocktube initial condition.

    Parameters
    ----------
    grid        : Grid1D instance.
    primL, primR: scalar primitive state dicts (left / right of interface).
    x_interface : x coordinate of the discontinuity.

    Returns
    -------
    prims : dict of (ntotal,) tensors — primitives on full domain.
    """
    x   = grid.x                              # (ntotal,)
    mask = x < x_interface                    # True on the left side

    prims: dict[str, torch.Tensor] = {}
    for k in primL:
        arr = torch.where(mask, primL[k].expand_as(mask),
                                primR[k].expand_as(mask))
        prims[k] = arr.to(dtype=primL[k].dtype, device=primL[k].device)

    return prims