"""
1D SR-MHD shocktube driver.

Scheme:
    - Spatial:  PLM reconstruction (MC limiter) + HLLD fluxes
    - Temporal: 2nd-order SSP Runge-Kutta (Shu & Osher)
    - C2P:      Kastaun vectorised scheme
    - Grid:     outflow boundary conditions

All heavy work is in vectorised PyTorch — no Python loops over cells.
"""

import functools
from pathlib import Path

import numpy as np
import torch

from src.physics import grid

from .c2p import conservative_to_primitive
from .eos import hybrid_eos
from .grid import Grid1D
from .hlld import (
    compute_srmhd_fluxes,
    hllc_flux,
    hlld_ai_flux,
    hlld_flux,
    hlle_flux,
    primitive_to_conserved,
)
from .initial_data import init_shocktube
from .reconstruction import get_interface_states

# ── conserved-variable dict helpers ──────────────────────────────────────────
#
# B^i is identical in the primitive and conserved variable sets (it is not
# evolved via the fluid update — in 1-D it is frozen, in 3-D it is updated
# by the induction equation separately).  We therefore keep it in a dedicated
# list and compose _CONS_KEYS / _PRIM_KEYS from the non-B fluid keys plus B,
# so that B never appears twice when both lists are iterated.

_HYDRO_CONS_KEYS = ["D", "Sx", "Sy", "Sz", "tau"]
_HYDRO_PRIM_KEYS = ["rho", "vx", "vy", "vz", "p", "eps"]
_MAG_KEYS = ["Bx", "By", "Bz"]

_CONS_KEYS = _HYDRO_CONS_KEYS + _MAG_KEYS  # full conserved set
_PRIM_KEYS = _HYDRO_PRIM_KEYS + _MAG_KEYS  # full primitive set


def _check_nan(label, d):
    for k, v in d.items():
        if torch.any(torch.isnan(v)):
            print(f"  NaN in {label}[{k}]")


def prims_to_cons_grid(prims: dict, grid: Grid1D) -> dict:
    """
    Convert primitives → conserved on all ntotal cells.
    Calls primitive_to_conserved cell-wise (already vectorised).
    """
    return primitive_to_conserved(prims)


def cons_to_prims_grid(
    cons: dict,
    eos: hybrid_eos,
    atmo_rho: float = 1e-10,
) -> dict:
    """Conservative → primitive on all cells (vectorised C2P)."""
    return conservative_to_primitive(cons, eos, atmo_rho=atmo_rho)


# ── CFL time-step ─────────────────────────────────────────────────────────────


def compute_dt(
    prims: dict,
    eos: hybrid_eos,
    dx: float,
    cfl: float,
    ng: int,
    ncells: int,
    speed_of_light: bool = False,
) -> torch.Tensor:
    """
    Estimate maximum stable time-step via CFL condition.
    Uses the fast magnetosonic speed in the physical domain.

    speed_of_light=True restores the old hardcoded ``dt = cfl*dx`` (max
    signal speed pinned to c).  Returns a 0-d tensor on the same device and
    dtype as the input, so the caller does not silently get a float64 CPU
    tensor back from a GPU run.
    """
    phys = slice(ng, ng + ncells)
    rho = prims["rho"][phys]
    eps = prims["eps"][phys]
    vx = prims["vx"][phys]
    vy = prims["vy"][phys]
    vz = prims["vz"][phys]
    Bx = prims["Bx"][phys]
    By = prims["By"][phys]
    Bz = prims["Bz"][phys]

    p, cs2 = eos.press_and_cs2(eps, rho)
    v2 = vx**2 + vy**2 + vz**2
    W = 1.0 / torch.sqrt(torch.clamp(1.0 - v2, min=1e-10))
    B2 = Bx**2 + By**2 + Bz**2
    vdotB = vx * Bx + vy * By + vz * Bz
    b2 = (B2 + vdotB**2) / W**2

    h = 1.0 + eps + p / rho
    v02 = cs2 + (b2 / (rho * h + b2)) * (1.0 - cs2)
    # approximate max wave speed: |vx| + v_fast
    v_fast = torch.sqrt(torch.clamp(v02, min=0.0, max=1.0))
    max_speed = (vx.abs() + v_fast).max()
    max_speed = torch.clamp(max_speed, min=1e-14)

    if speed_of_light:
        # Legacy behaviour: assume the max signal speed is c = 1, i.e.
        # dt = cfl*dx.  Safe in SR (all speeds <= c) but needlessly small,
        # and it silently discards the wave speed computed above.  In 2D the
        # CFL limit must combine both directions, so this cannot be the
        # default there; kept as a switch for bisecting scheme problems.
        max_speed = torch.ones_like(max_speed)

    return (cfl * dx) / max_speed


# ── RHS: compute dU/dt ────────────────────────────────────────────────────────

_SOLVERS = {"hlld": hlld_flux, "hlle": hlle_flux, "hllc": hllc_flux}


def compute_rhs(
    cons: dict,
    prims: dict,
    grid: Grid1D,
    eos: hybrid_eos,
    flux_fn=hlld_flux,
    limiter: str = "pcm",
) -> dict:
    """
    Compute the spatial RHS  −(F_{i+1/2} − F_{i-1/2}) / dx
    for all physical cells.

    Parameters
    ----------
    flux_fn : callable
        Riemann solver used at each interface.
    limiter : "pcm" | "minmod" | "mc"
        Reconstruction.  NOTE the default is "pcm" (piecewise constant,
        1st order) purely to preserve historical behaviour — this was
        hardcoded here, so every published 1D run so far is 1st order
        despite the module docstring claiming PLM.  Set it from config.

    Returns a dict of (ntotal,) tensors where only the physical
    region [ng : ng+N] is meaningful (ghosts are zero).
    """
    ng = grid.ng
    ncells = grid.ncells
    dx = grid.dx

    # Reconstruct at interfaces (ncells+1 of them)
    primL, primR = get_interface_states(prims, ng, ncells, limiter=limiter)

    # Riemann flux at each interface
    fHLLD, _, _ = flux_fn(primL, primR, eos, idir=0)

    # In 1-D Bx is constant — override flux to enforce exactly
    fHLLD["Bx"] = torch.zeros_like(fHLLD["Bx"])

    # Assemble RHS:  rhs[i] = -(F[i+1/2] - F[i-1/2]) / dx
    # fHLLD has shape (ncells+1,); fHLLD[j] is the flux at interface j
    # (between physical cell j-1 and j for j=0..ncells)
    rhs: dict = {}
    for k in _CONS_KEYS:
        f = fHLLD[k]  # (ncells+1,)
        dF = (f[:-1] - f[1:]) / dx  # (ncells,)

        # Place in full array (ghosts stay zero)
        arr = torch.zeros(grid.ntotal, dtype=dF.dtype, device=dF.device)
        arr[ng : ng + ncells] = +dF
        rhs[k] = arr

    return rhs


# ── RK2 update ───────────────────────────────────────────────────────────────


def rk2_step(
    cons: dict,
    prims: dict,
    grid: Grid1D,
    eos: hybrid_eos,
    dt: torch.Tensor,
    atmo_rho: float,
    bc: str = "outflow",
    flux_fn=hlld_flux,
    limiter: str = "pcm",
) -> tuple[dict, dict]:
    """
    One SSP-RK2 (Heun) step:
        U^(1) = U^n + dt * L(U^n)
        U^{n+1} = 0.5 * (U^n + U^(1) + dt * L(U^(1)))

    Returns updated (cons, prims).
    """

    def _bc(U, U_old):
        if bc == "outflow":
            return grid.apply_outflow_bc(U)
        elif bc == "constant":
            return grid.apply_constant_bc(U, U_old)
        return grid.apply_periodic_bc(U)

    # Stage 1
    k1 = compute_rhs(cons, prims, grid, eos, flux_fn, limiter)
    cons1 = {k: cons[k] + dt * k1[k] for k in _CONS_KEYS}
    cons1 = _bc(cons1, cons)
    # Enforce Bx = constant (copy from initial)
    cons1["Bx"] = cons["Bx"].clone()

    prims1 = cons_to_prims_grid(cons1, eos, atmo_rho)
    prims1 = _bc(prims1, prims)

    # Stage 2
    k2 = compute_rhs(cons1, prims1, grid, eos, flux_fn, limiter)
    cons_new = {k: 0.5 * (cons[k] + cons1[k] + dt * k2[k]) for k in _CONS_KEYS}
    cons_new = _bc(cons_new, cons)
    cons_new["Bx"] = cons["Bx"].clone()
    prims_new = cons_to_prims_grid(cons_new, eos, atmo_rho)
    prims_new = _bc(prims_new, prims)

    return cons_new, prims_new


# NOTE: rk2_midpoint_step was removed here.  It was dead code and broken:
# it passed its `solver: str` argument into compute_rhs's `flux_fn` slot,
# which expects a callable, so it would have raised on first use.  SSP-RK3
# for the 2D driver is implemented in driver2d.py instead.


def save_snapshot(
    path: Path,
    t: float,
    grid: Grid1D,
    prims: dict,
    cons: dict,
):
    """Save physical-region data to a compressed numpy archive."""
    ng = grid.ng
    ncells = grid.ncells
    phys = slice(ng, ng + ncells)

    data = {"t": t, "x": grid.x_phys.cpu().numpy()}
    # Fluid primitives (no B — B is saved once below)
    for k in _HYDRO_PRIM_KEYS:
        data[f"prim_{k}"] = prims[k][phys].cpu().numpy()
    # Fluid conserved (no B)
    for k in _HYDRO_CONS_KEYS:
        data[f"cons_{k}"] = cons[k][phys].cpu().numpy()
    # Magnetic field — identical in prim and cons; save once
    for k in _MAG_KEYS:
        data[k] = cons[k][phys].cpu().numpy()

    np.savez_compressed(path, **data)


# ── Machine Learning ─────────────────────────────────────────────────────────────


def _load_ai_solver(cfg: dict, device: str):
    """Load ML model and norm stats for hlld_ai solver."""
    import numpy as np

    from src.models.network import PressureNet

    ml_cfg = cfg.get("ml", {})
    ckpt_path = ml_cfg.get("checkpoint", "checkpoints/default-mignone.pt")
    stats_path = ml_cfg.get(
        "norm_stats", "data/processed/default-mignone/norm_stats.npz"
    )

    from src.physics.ai_features import FEATURE_VERSION, N_FEATURES

    model = PressureNet(
        n_input=ml_cfg.get("n_input", N_FEATURES),
        hidden=ml_cfg.get("hidden", [256, 256, 256, 128, 64]),
        activation=ml_cfg.get("activation", "tanh"),
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    norm_stats = dict(np.load(stats_path))

    # Refuse a checkpoint whose features were built by a different
    # definition.  A mismatch is invisible at run time -- the network happily
    # consumes any 15 numbers and returns a plausible-looking p* -- so it has
    # to be caught here rather than debugged later from bad physics.
    stamped = norm_stats.get("feature_version")
    stamped = (str(stamped) if stamped is not None
               and not hasattr(stamped, "size") else
               (str(stamped.item()) if stamped is not None else None))
    if stamped is None:
        print(f"  WARNING: {stats_path} predates feature versioning; assuming "
              f"it matches {FEATURE_VERSION!r}. Regenerate to remove this "
              f"warning.")
    elif stamped != FEATURE_VERSION:
        raise ValueError(
            f"feature-version mismatch: checkpoint stats {stats_path} were "
            f"built with {stamped!r} but this code produces "
            f"{FEATURE_VERSION!r}. Retrain, or check out the matching commit."
        )

    n_in = int(norm_stats["mu"].shape[0]) if "mu" in norm_stats else N_FEATURES
    if n_in != N_FEATURES:
        raise ValueError(
            f"{stats_path} has {n_in} features but ai_features defines "
            f"{N_FEATURES}"
        )
    return model, norm_stats


# ── Main ──────────────────────────────────────────────────────────────────────


def run(cfg: dict):
    """
    Run a 1D SR-MHD shocktube simulation.

    Parameters
    ----------
    cfg : dict parsed from a YAML config file.
          Expected top-level keys: grid, eos, run, output.
    """
    # ── Config ───────────────────────────────────────────────────────────────
    gc = cfg["grid"]
    ec = cfg["eos"]
    rc = cfg["run"]
    oc = cfg["output"]

    device = rc.get("device", "cpu")
    dtype = torch.float64

    # ── EOS ──────────────────────────────────────────────────────────────────
    eos = hybrid_eos(
        K=ec["K"],
        gamma=ec["gamma"],
        gamma_th=ec.get("gamma_th", None),
    )

    # ── Grid ─────────────────────────────────────────────────────────────────
    grid = Grid1D(
        xmin=gc["xmin"],
        xmax=gc["xmax"],
        ncells=gc["ncells"],
        ng=gc.get("ng", 2),
        device=device,
    )

    # ── Initial data ─────────────────────────────────────────────────────────
    from . import initial_data as id_mod

    problem = rc.get("problem", "brio_wu")
    id_func = getattr(id_mod, problem)

    if problem == "generic":
        primL_cfg = rc["primL"]
        primR_cfg = rc["primR"]
        x_int = rc.get("x_interface", 0.5)
        primL, primR, x_int = id_func(
            primL_cfg, primR_cfg, x_int, dtype=dtype, device=device, eos=eos
        )
    else:
        primL, primR, x_int = id_func(dtype=dtype, device=device)

    prims = init_shocktube(grid, primL, primR, x_int)
    cons = prims_to_cons_grid(prims, grid)
    grid.apply_outflow_bc(prims)
    grid.apply_outflow_bc(cons)
    # cons  = prims_to_cons_grid(prims, grid)

    # ── Output directory ─────────────────────────────────────────────────────
    out_dir = Path(oc.get("dir", "output"))
    out_dir.mkdir(parents=True, exist_ok=True)

    t_end = rc["t_end"]
    cfl = rc.get("cfl", 0.4)
    atmo_rho = rc.get("atmo_rho", 1e-10)
    solver = rc.get("solver", "hlld")
    out_every = oc.get("every_n_steps", 100)
    out_dt_max = oc.get("dt_snapshot", None)

    t = 0.0
    step = 0
    t_next_snap = 0.0

    # ── Solver dispatch ───────────────────────────────────────────────────────
    if solver == "hlld_ai":
        model, norm_stats = _load_ai_solver(cfg, device)
        flux_fn = functools.partial(hlld_ai_flux, model=model, norm_stats=norm_stats)
    else:
        flux_fn = {"hlld": hlld_flux, "hlle": hlle_flux, "hllc": hllc_flux}.get(
            solver, hlld_flux
        )

    print(
        f"Starting {problem}  ncells={grid.ncells}  t_end={t_end}  solver={solver}  device={device}"
    )

    # Reconstruction + dt options.  Defaults reproduce the historical
    # behaviour exactly (1st-order PCM, dt = cfl*dx) so existing configs are
    # unaffected; set them in the `run:` block to opt in.
    limiter = rc.get("limiter", "pcm")
    dt_c = bool(rc.get("dt_speed_of_light", True))

    while t < t_end:
        dt = compute_dt(
            prims, eos, grid.dx, cfl, grid.ng, grid.ncells, speed_of_light=dt_c
        ).item()
        dt = min(dt, t_end - t)
        if dt <= 0.0:
            break

        if step == 0:
            fname = out_dir / f"snap_{step:06d}.npz"
            save_snapshot(fname, t, grid, prims, cons)
            print(f"  step {step:6d}  t={t:.4e}  dt={dt:.3e}  → {fname.name}")

        cons, prims = rk2_step(
            cons,
            prims,
            grid,
            eos,
            torch.tensor(dt, dtype=dtype, device=device),
            atmo_rho,
            flux_fn=flux_fn,
            limiter=limiter,
        )
        t += dt
        step += 1

        do_snap = (step % out_every == 0) or (t >= t_end)
        if out_dt_max is not None:
            do_snap = do_snap or (t >= t_next_snap)

        if do_snap:
            fname = out_dir / f"snap_{step:06d}.npz"
            save_snapshot(fname, t, grid, prims, cons)
            t_next_snap = t + (out_dt_max or 0.0)
            print(f"  step {step:6d}  t={t:.4e}  dt={dt:.3e}  → {fname.name}")

    print("Done.")
    return t, grid, prims, cons
