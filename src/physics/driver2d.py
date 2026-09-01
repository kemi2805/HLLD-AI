"""
2D SR-MHD driver: directionally-unsplit finite volume.

Scheme
------
    dU/dt = -(F^x_{i+1/2} - F^x_{i-1/2})/dx - (F^y_{j+1/2} - F^y_{j-1/2})/dy

with both sweeps evaluated from the SAME state and summed (unsplit), rather
than applied one after the other (operator split).  Time integration is SSP
Runge-Kutta 2 or 3.

Constrained transport is not applied here; ``compute_rhs_2d`` evolves the
magnetic field as ordinary conserved variables, which is correct only where
the solution has no transverse variation (e.g. a shock tube run along one
axis).  CT replaces the Bx/By update in ct.py.
"""

from __future__ import annotations

import torch

from .c2p import conservative_to_primitive
from .hlld import hlld_flux, primitive_to_conserved
from .state import flat, unflat
from .sweep import flux_divergence, sweep

# All eight conserved variables, i.e. the no-CT configuration.
CONS_KEYS_2D = ["D", "Sx", "Sy", "Sz", "tau", "Bx", "By", "Bz"]


def compute_rhs_2d(
    prims: dict[str, torch.Tensor],
    grid,
    eos,
    flux_fn=hlld_flux,
    limiter: str = "mc",
    vel_var: str = "Wv",
    keys=None,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Unsplit 2D spatial residual on the full ghost-inclusive block."""
    keys = list(CONS_KEYS_2D if keys is None else keys)

    Fx, ax = sweep(prims, grid, eos, 0, flux_fn, limiter, vel_var)
    Fy, ay = sweep(prims, grid, eos, 1, flux_fn, limiter, vel_var)

    rx = flux_divergence(Fx, 0, grid.dx, keys)
    ry = flux_divergence(Fy, 1, grid.dy, keys)
    rhs = {k: rx[k] + ry[k] for k in keys}
    return rhs, {"x": ax, "y": ay}


def compute_dt_2d(prims: dict, grid, eos, cfl: float,
                  speed_of_light: bool = False) -> float:
    """Unsplit CFL limit.

    For a directionally-unsplit update the stable step uses the SUM of the
    per-direction inverse crossing times, not the minimum:

        dt = cfl / ( max(|vx| + cf)/dx + max(|vy| + cf)/dy )
    """
    ph = grid.phys
    rho, eps = prims["rho"][ph], prims["eps"][ph]
    vx, vy, vz = prims["vx"][ph], prims["vy"][ph], prims["vz"][ph]
    Bx, By, Bz = prims["Bx"][ph], prims["By"][ph], prims["Bz"][ph]

    p, cs2 = eos.press_and_cs2(eps, rho)
    v2 = vx**2 + vy**2 + vz**2
    W = 1.0 / torch.sqrt(torch.clamp(1.0 - v2, min=1e-10))
    B2 = Bx**2 + By**2 + Bz**2
    vdotB = vx * Bx + vy * By + vz * Bz
    b2 = (B2 + vdotB**2) / W**2
    h = 1.0 + eps + p / rho
    v02 = cs2 + (b2 / (rho * h + b2)) * (1.0 - cs2)
    cf = torch.sqrt(torch.clamp(v02, min=0.0, max=1.0))

    if speed_of_light:
        sx = sy = torch.ones((), dtype=rho.dtype, device=rho.device)
    else:
        sx = torch.clamp((vx.abs() + cf).max(), min=1e-14)
        sy = torch.clamp((vy.abs() + cf).max(), min=1e-14)

    return float(cfl / (sx / grid.dx + sy / grid.dy))


def cons_to_prims_2d(cons: dict, grid, eos, atmo_rho: float = 1e-10,
                     return_status: bool = False):
    """Cell-centred c2p.  Flattens for KastaunC2P, which needs ``(N,)``."""
    out = conservative_to_primitive(flat(cons), eos, atmo_rho=atmo_rho,
                                    return_status=return_status)
    if return_status:
        prims, status = out
        return (unflat(prims, grid.shape),
                {k: v.reshape(*grid.shape) for k, v in status.items()})
    return unflat(out, grid.shape)


def prims_to_cons_2d(prims: dict, grid) -> dict:
    return unflat(primitive_to_conserved(flat(prims)), grid.shape)


# ── SSP Runge-Kutta ──────────────────────────────────────────────────────────

# Shu-Osher SSP-RK3, written as (weight on U^n, weight on the stage input, dt).
_RK3 = ((1.0, 0.0, 1.0), (0.75, 0.25, 0.25), (1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0))
_RK2 = ((1.0, 0.0, 1.0), (0.5, 0.5, 0.5))

_SCHEMES = {"rk2": _RK2, "rk3": _RK3}


def rk_step(
    cons: dict,
    prims: dict,
    grid,
    eos,
    dt: float,
    *,
    scheme: str = "rk3",
    bc_x: str = "outflow",
    bc_y: str = "outflow",
    flux_fn=hlld_flux,
    limiter: str = "mc",
    vel_var: str = "Wv",
    atmo_rho: float = 1e-10,
    keys=None,
) -> tuple[dict, dict, dict]:
    """One SSP-RK step.

    Each stage is ``U <- a*U^n + b*U^stage + c*dt*L(U^stage)``, applied with
    identical coefficients to every evolved variable.
    """
    stages = _SCHEMES[scheme]
    keys = list(CONS_KEYS_2D if keys is None else keys)

    U0 = {k: cons[k].clone() for k in keys}
    U = {k: cons[k] for k in keys}
    P = prims
    diag: dict = {}

    for a, b, c in stages:
        rhs, diag = compute_rhs_2d(P, grid, eos, flux_fn, limiter, vel_var, keys)
        U = {k: a * U0[k] + b * U[k] + c * dt * rhs[k] for k in keys}
        # NOTE: Bx and By are NOT held fixed here.  In 2D each is advanced by
        # the TRANSVERSE sweep -- F^y(Bx) = Bx*vy - By*vx and
        # F^x(By) = By*vx - Bx*vy -- while its own normal flux is zeroed
        # inside sweep().  Freezing them here would kill the induction
        # equation entirely.  (For a problem uniform along y the transverse
        # flux difference vanishes on its own, so Bx stays constant without
        # being forced.)
        grid.apply_bc_cc(U, bc_x, bc_y, U_ref=U0)
        P = cons_to_prims_2d(U, grid, eos, atmo_rho)
        for k in ("Bx", "By", "Bz"):
            if k in U:
                P[k] = U[k]
        grid.apply_bc_cc(P, bc_x, bc_y)

    return U, P, diag
