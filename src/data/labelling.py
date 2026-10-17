"""
Label Riemann problems with TODAY's HLLD star pressure, row by row.

The label is exactly what ``hlld.hlld_flux`` computes -- same bracket
(``_PSTAR_BRACKET``), same tolerance (``_PSTAR_TOL`` = 1e-10), same
root-finder, same B_n = 0 and wave-order checks -- read through its
``row_diag`` hook, so nothing here re-implements the solver.  Per row:

    p_star          what hlld_flux returns (negative where it fell back)
    p_root          the root-finder's p* before that fallback overwrote it
    resid           |HLLD residual| at p_root
    n_iter          secant iterations (30 = gave up)
    converged       err == 0, i.e. |residual| < 1e-10
    wave_order_ok   not (vaR > cmax | vaL < -cmin | vc < laL | vc > laR)
    bn_zero         |B_n| < 1e-14 on both sides (HLLD is singular there)
    finite          p_root and every flux component finite
    valid           all of the above and p_star > 0: a row HLLD solves.
                    Training uses only these; the rest are kept, flagged.

plus the raw feature matrices of every requested feature version, built by
``ai_features.pstar_features`` -- the call hlld_ai_flux makes at inference.
"""

from __future__ import annotations

import numpy as np
import torch

from src.physics import ai_features as ai
from src.physics.eos import hybrid_eos
from src.physics.hlld import compute_srmhd_fluxes, energy_form, hlld_flux

PRIM_KEYS = ("rho", "vx", "vy", "vz", "p", "eps", "Bx", "By", "Bz")
FLAG_KEYS = ("converged", "wave_order_ok", "bn_zero", "finite", "valid")


def make_eos(gamma: float) -> hybrid_eos:
    """The ideal-gas EOS every 2D problem in this repo uses (K = 0)."""
    return hybrid_eos(K=0.0, gamma=gamma, gamma_th=gamma)


def to_prims(state: dict, eos: hybrid_eos, need_eps: bool = True) -> dict:
    """numpy/torch dict -> float64 torch dict with eps from (p, rho)."""
    s = {k: torch.as_tensor(np.asarray(v), dtype=torch.float64)
         for k, v in state.items() if k in PRIM_KEYS}
    if need_eps or "eps" not in s:
        s["eps"] = eos.eps__press_rho(s["p"], s["rho"])
    return s


def label(sL: dict, sR: dict, eos: hybrid_eos, idir: int = 0,
          versions=(ai.FEATURE_VERSION, ai.FEATURE_VERSION_V3)) -> dict:
    """HLLD labels, flags and features for one batch (torch float64 dicts)."""
    rd: dict = {}
    fD, _, p_star = hlld_flux(sL, sR, eos, idir=idir, row_diag=rd)
    finite_flux = torch.ones_like(p_star, dtype=torch.bool)
    for v in fD.values():
        finite_flux &= torch.isfinite(v)
    p_root = rd["p_root"]
    out = dict(
        p_star=p_star,
        p_root=p_root,
        resid=rd["resid"].abs(),
        p_hll=rd["p_hll"],
        n_iter=rd["n_iter"].to(torch.int16),
        converged=rd["err"] == 0,
        wave_order_ok=~rd["wave_order_bad"],
        bn_zero=rd["bn_zero"],
        finite=torch.isfinite(p_root) & finite_flux,
    )
    out["valid"] = (out["converged"] & out["wave_order_ok"] & ~out["bn_zero"]
                    & out["finite"] & (p_star > 0))

    uL, uR, fL, fR, cmax, cmin = compute_srmhd_fluxes(sL, sR, eos, idir)
    UL, FL = energy_form(uL, fL)
    UR, FR = energy_form(uR, fR)
    for v in versions:
        out[f"X_{v}"] = ai.pstar_features(v, sL, sR, uL, uR, UL, FL, UR, FR,
                                          cmin, cmax, idir)
    return {k: v.numpy() for k, v in out.items()}


def label_batched(sL: dict, sR: dict, eos: hybrid_eos, idir=0,
                  batch: int = 100_000, versions=(ai.FEATURE_VERSION,
                                                  ai.FEATURE_VERSION_V3)):
    """`label` in batches (the secant runs every row of a batch until the
    slowest converges, so moderate batches waste less), concatenated.
    ``idir`` may be an int or a per-row integer array (recorded 2D faces)."""
    n = int(np.asarray(sL["rho"]).shape[0])
    idir_arr = (np.full(n, int(idir)) if np.ndim(idir) == 0
                else np.asarray(idir).astype(int))
    parts: list = []
    order: list = []
    for d in np.unique(idir_arr):
        rows = np.flatnonzero(idir_arr == d)
        for i0 in range(0, rows.size, batch):
            r = rows[i0:i0 + batch]
            tl = {k: torch.as_tensor(np.asarray(v)[r]) for k, v in sL.items()}
            tr = {k: torch.as_tensor(np.asarray(v)[r]) for k, v in sR.items()}
            parts.append(label(tl, tr, eos, int(d), versions))
            order.append(r)
    idx = np.concatenate(order)
    inv = np.empty_like(idx)
    inv[idx] = np.arange(idx.size)
    return {k: np.concatenate([p[k] for p in parts], axis=0)[inv]
            for k in parts[0]}


def invalid_reasons(lab: dict) -> dict:
    """Counts of the invalid rows by FIRST reason, in the order HLLD's own
    checks matter: B_n = 0, not converged, wave order, non-finite, p* <= 0."""
    inv = ~lab["valid"]
    left = inv.copy()
    out = {}
    for name, bad in (("bn_zero", lab["bn_zero"]),
                      ("not_converged", ~lab["converged"]),
                      ("wave_order", ~lab["wave_order_ok"]),
                      ("non_finite", ~lab["finite"])):
        hit = left & bad
        out[name] = int(hit.sum())
        left &= ~hit
    out["other"] = int(left.sum())
    out["total_invalid"] = int(inv.sum())
    # overlaps, for the record: every reason that applies, not the first
    out["any_not_converged"] = int((inv & ~lab["converged"]).sum())
    out["any_wave_order"] = int((inv & ~lab["wave_order_ok"]).sum())
    return out
