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


# ── constrained-transport driver ─────────────────────────────────────────────


def ct_stage(state, grid, eos, *, flux_fn=hlld_flux, limiter="mc",
             vel_var="Wv", emf_mode="solver", upwind=True, recorder=None):
    """One spatial-residual evaluation with constrained transport.

    Returns ``(rhs_cons, dBxf, dByf, diag)``.  ``state.prims`` must already
    be consistent with ``state.cons`` and with the staggered field.
    """
    from . import ct

    Bcx, Bcy = ct.cell_centred_B(state.Bxf, state.Byf)
    prims = dict(state.prims)
    prims["Bx"], prims["By"] = Bcx, Bcy

    Ec = ct.emf_cell(Bcx, Bcy, prims["vx"], prims["vy"])

    want_hll = emf_mode == "hll"
    Fx, ax = sweep(prims, grid, eos, 0, flux_fn, limiter, vel_var,
                   Bnf=state.Bxf, want_hll_aux=want_hll, recorder=recorder)
    Fy, ay = sweep(prims, grid, eos, 1, flux_fn, limiter, vel_var,
                   Bnf=state.Byf, want_hll_aux=want_hll, recorder=recorder)

    Efx = ct.face_emf_x(Fx, ax, emf_mode)
    Efy = ct.face_emf_y(Fy, ay, emf_mode)
    Ez = ct.corner_emf(Ec, Efx, Efy, Fx["D"], Fy["D"], upwind=upwind)

    from .state import EVOLVED_KEYS
    rx = flux_divergence(Fx, 0, grid.dx, EVOLVED_KEYS)
    ry = flux_divergence(Fy, 1, grid.dy, EVOLVED_KEYS)
    rhs = {k: rx[k] + ry[k] for k in EVOLVED_KEYS}

    dBxf, dByf = ct.ct_rhs(Ez, grid.dx, grid.dy)
    return rhs, dBxf, dByf, {"x": ax, "y": ay}


def sync_state(state, grid, eos, bc_x="outflow", bc_y="outflow",
               atmo_rho=1e-10, return_status=False):
    """Make ``prims`` consistent with ``cons`` + staggered B, and apply BCs.

    Order matters: the staggered field is the primary variable, so the
    cell-centred B fed to the inversion is always re-derived from it.
    """
    from . import ct

    grid.apply_bc_face(state.Bxf, state.Byf, bc_x, bc_y)
    grid.apply_bc_cc(state.cons, bc_x, bc_y)

    Bcx, Bcy = ct.cell_centred_B(state.Bxf, state.Byf)
    cons = dict(state.cons)
    cons["Bx"], cons["By"] = Bcx, Bcy

    out = cons_to_prims_2d(cons, grid, eos, atmo_rho, return_status)
    prims, status = out if return_status else (out, None)
    prims["Bx"], prims["By"] = Bcx, Bcy
    prims["Bz"] = cons["Bz"]
    grid.apply_bc_cc(prims, bc_x, bc_y)
    state.prims = prims
    return (state, status) if return_status else state


def rk_step_ct(state, grid, eos, dt, *, scheme="rk3", bc_x="outflow",
               bc_y="outflow", flux_fn=hlld_flux, limiter="mc",
               vel_var="Wv", emf_mode="solver", upwind=True,
               atmo_rho=1e-10, recorder=None):
    """One SSP-RK step with constrained transport.

    The cell-centred conserved variables and the staggered field are
    combined with IDENTICAL Runge-Kutta coefficients (via ``state.combine``),
    which is what keeps every stage divergence-free without any dedicated
    CT/RK machinery.
    """
    from .state import EVOLVED_KEYS, State2D, combine

    stages = _SCHEMES[scheme]
    S0 = State2D(cons={k: state.cons[k].clone() for k in EVOLVED_KEYS},
                 Bxf=state.Bxf.clone(), Byf=state.Byf.clone(),
                 prims=state.prims)
    S = state
    diag: dict = {}

    for a, b, c in stages:
        rhs, dBxf, dByf, diag = ct_stage(
            S, grid, eos, flux_fn=flux_fn, limiter=limiter, vel_var=vel_var,
            emf_mode=emf_mode, upwind=upwind, recorder=recorder)
        L = State2D(cons=rhs, Bxf=dBxf, Byf=dByf)
        S = combine([S0, S, L], [a, b, c * dt])
        S = sync_state(S, grid, eos, bc_x, bc_y, atmo_rho)

    return S, diag
