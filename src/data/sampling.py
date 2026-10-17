"""
Left/right interface states for the p* network, sampled over the UNION of
the measured rotor and Orszag-Tang envelopes (the 2026-10 retrain).

Why not the archived generator (``generate.py``) with wider ranges
------------------------------------------------------------------
It samples L and R independently and uniformly, with uniformly random
transverse directions.  Measured on the real interfaces (the September
envelope recordings, docs/rotor_envelope.md, docs/ot_envelope.md):

* the faces are overwhelmingly SMALL jumps: the largest relative jump
  between L and R has median 2e-11 (rotor) / 1.5e-3 (OT) and 99th
  percentile 0.23 / 0.53.  Independent L and R essentially never land
  there, so a mixture component draws R as a perturbation of L with a
  log-uniform size (``jump``);
* every face of a 2D problem is COPLANAR (v_t and B_t on one line, both
  sides; 100% of the recorded OT faces have cos(v_t, B_t) = +-1).
  Uniform directions never produce that, so half the rows put every
  transverse vector on one random line, with random signs (``*_planar``);
* small |B_n| is common (OT: 29% of faces below 0.1, 3.4% below 0.01) and
  so is a vanishing transverse field (rotor: 73% of faces |B_t| < 1e-6):
  log-uniform tails for both;
* the box's corners are far outside the measured JOINT envelope:
  sigma = b^2/(rho h) <= 7.85 measured (rotor), log10(p/rho) in
  [-2.0, 1.05] measured (both), while independent ranges reach sigma ~1e4
  and log10(p/rho) in [-4.5, 4].  Rows past ``caps`` (about twice the
  measured sigma, half a decade past the measured temperatures) are
  redrawn.

The box itself (``box``) is the union of the two documented envelope boxes:
the rotor's recommended ranges (docs/rotor_envelope.md) and the OT box of
rmhd_final's scripts/envelopes.sh.  W is sampled as W = 1 + 10^u.

All rows have the normal along x (idir = 0); the states carry no eps -- the
labeller computes it from (p, rho) with the run's EOS.  Everything is numpy
float64 drawn from one ``np.random.Generator``, so a chunk is reproducible
from its seed.
"""

from __future__ import annotations

import numpy as np

DEFAULT_CFG = dict(
    box=dict(lrho=[-2.0, 1.5], lpress=[-3.0, 2.0], wm1_log=[-4.0, 1.0],
             Bn=[0.0, 7.5], Bt=[0.0, 6.0]),
    small_Bn=dict(frac=0.25, log10=[-6.0, 0.0]),
    small_Bt=dict(frac=0.15, log10=[-6.0, -1.0]),
    caps=dict(sigma_max=16.0, log_T=[-2.5, 1.55]),
    components=dict(indep=0.25, indep_planar=0.25, jump=0.25, jump_planar=0.25),
    jump_log10=[-5.0, 0.0],
)

COMPONENTS = ("indep", "indep_planar", "jump", "jump_planar")
STATE_KEYS = ("rho", "vx", "vy", "vz", "p", "Bx", "By", "Bz")


def merged_cfg(cfg: dict | None) -> dict:
    """DEFAULT_CFG with the given (possibly partial) sampling config on top."""
    out = {k: (dict(v) if isinstance(v, dict) else v)
           for k, v in DEFAULT_CFG.items()}
    for k, v in (cfg or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = v
    return out


def _u(rng, lo_hi, n):
    lo, hi = lo_hi
    return lo + (hi - lo) * rng.random(n)


def _mixture_mag(rng, n, uniform_range, small):
    """|x| from U(range) with probability 1 - small.frac, else 10^U(log10)."""
    mag = _u(rng, uniform_range, n)
    tail = rng.random(n) < small["frac"]
    mag[tail] = 10.0 ** _u(rng, small["log10"], int(tail.sum()))
    return mag


def _side(rng, n, box, small_Bt, planar, et):
    """One side's (rho, p, v, B_t): v as (v_n, v_t1, v_t2), B_t as (t1, t2).

    ``planar`` rows put v_t and B_t on the line ``et`` (unit 2-vectors,
    shared by both sides of a row), each with a random sign."""
    rho = 10.0 ** _u(rng, box["lrho"], n)
    p = 10.0 ** _u(rng, box["lpress"], n)
    W = 1.0 + 10.0 ** _u(rng, box["wm1_log"], n)
    vmag = np.sqrt(1.0 - 1.0 / W ** 2)

    # 3D: isotropic direction; planar: an angle in the (n, et) plane
    cos_t = _u(rng, (-1.0, 1.0), n)
    sin_t = np.sqrt(np.clip(1.0 - cos_t ** 2, 0.0, None))
    phi = _u(rng, (0.0, 2.0 * np.pi), n)
    alpha = _u(rng, (0.0, 2.0 * np.pi), n)
    vn = np.where(planar, vmag * np.cos(alpha), vmag * cos_t)
    vt_signed = vmag * np.sin(alpha)
    vt1 = np.where(planar, vt_signed * et[0], vmag * sin_t * np.cos(phi))
    vt2 = np.where(planar, vt_signed * et[1], vmag * sin_t * np.sin(phi))

    bt = _mixture_mag(rng, n, box["Bt"], small_Bt)
    phiB = _u(rng, (0.0, 2.0 * np.pi), n)
    sgnB = np.where(rng.random(n) < 0.5, -1.0, 1.0)
    bt1 = np.where(planar, sgnB * bt * et[0], bt * np.cos(phiB))
    bt2 = np.where(planar, sgnB * bt * et[1], bt * np.sin(phiB))
    return rho, p, np.stack([vn, vt1, vt2]), np.stack([bt1, bt2])


def _perturb(rng, n, L, Bn, planar, et, jump_log10):
    """R as a jump of log-uniform relative size away from L."""
    rho, p, v, bt = L
    d = 10.0 ** _u(rng, jump_log10, n)
    xi = lambda: _u(rng, (-1.0, 1.0), n)
    rhoR = rho * 10.0 ** (d * xi())
    pR = p * 10.0 ** (d * xi())

    # perturb the 4-velocity u = W v: always subluminal afterwards
    W = 1.0 / np.sqrt(1.0 - (v ** 2).sum(axis=0))
    u = W * v
    us = d * (1.0 + np.sqrt((u ** 2).sum(axis=0)))
    du_n, du_1, du_2 = us * xi(), us * xi(), us * xi()
    du_t = us * xi()                   # planar: along et only
    du_1 = np.where(planar, du_t * et[0], du_1)
    du_2 = np.where(planar, du_t * et[1], du_2)
    uR = u + np.stack([du_n, du_1, du_2])
    vR = uR / np.sqrt(1.0 + (uR ** 2).sum(axis=0))

    # field jumps on the scale of |B| plus the thermal equivalent sqrt(2p)
    bs = d * (np.sqrt(Bn ** 2 + (bt ** 2).sum(axis=0)) + np.sqrt(2.0 * p))
    db_1, db_2, db_t = bs * xi(), bs * xi(), bs * xi()
    db_1 = np.where(planar, db_t * et[0], db_1)
    db_2 = np.where(planar, db_t * et[1], db_2)
    btR = bt + np.stack([db_1, db_2])
    return rhoR, pR, vR, btR


def _in_box(box, rho, p, v, bt):
    W = 1.0 / np.sqrt(np.clip(1.0 - (v ** 2).sum(axis=0), 1e-300, None))
    lr, lp = np.log10(rho), np.log10(p)
    btm = np.sqrt((bt ** 2).sum(axis=0))
    return ((lr >= box["lrho"][0]) & (lr <= box["lrho"][1])
            & (lp >= box["lpress"][0]) & (lp <= box["lpress"][1])
            & (W <= 1.0 + 10.0 ** box["wm1_log"][1])
            & (btm <= box["Bt"][1]))


def _caps_ok(caps, gamma, rho, p, v, Bn, bt):
    B = np.stack([Bn, bt[0], bt[1]])
    v2 = (v ** 2).sum(axis=0)
    W2 = 1.0 / (1.0 - v2)
    vB = (v * B).sum(axis=0)
    b2 = (B ** 2).sum(axis=0) / W2 + vB ** 2
    rh = rho + gamma / (gamma - 1.0) * p          # rho*h, ideal gas
    lT = np.log10(p / rho)
    return ((b2 / rh <= caps["sigma_max"])
            & (lT >= caps["log_T"][0]) & (lT <= caps["log_T"][1]))


def _draw(rng, n, cfg, c):
    """n candidate rows of component ``c`` (not yet capped): (L, R, Bn)."""
    box = cfg["box"]
    name = COMPONENTS[c]
    planar = np.full(n, name.endswith("_planar"))
    phi0 = _u(rng, (0.0, 2.0 * np.pi), n)
    et = np.stack([np.cos(phi0), np.sin(phi0)])
    Bn = (_mixture_mag(rng, n, box["Bn"], cfg["small_Bn"])
          * np.where(rng.random(n) < 0.5, -1.0, 1.0))
    L = _side(rng, n, box, cfg["small_Bt"], planar, et)
    if name.startswith("jump"):
        R = _perturb(rng, n, L, Bn, planar, et, cfg["jump_log10"])
    else:
        R = _side(rng, n, box, cfg["small_Bt"], planar, et)
    return L, R, Bn


def component_counts(n: int, weights: dict) -> np.ndarray:
    """Exact per-component row counts for n rows (largest remainder)."""
    w = np.array([float(weights.get(c, 0.0)) for c in COMPONENTS])
    if w.sum() <= 0:
        raise ValueError("component weights sum to zero")
    q = n * w / w.sum()
    k = np.floor(q).astype(int)
    k[np.argsort(-(q - k))[: n - k.sum()]] += 1
    return k


def sample_pairs(n: int, cfg: dict | None, rng: np.random.Generator,
                 gamma: float, max_rounds: int = 60) -> dict:
    """``n`` accepted L/R pairs, in the EXACT component proportions.

    Each component is filled by rejection on its own (the caps reject the
    independent components far more often than the jump ones, so a shared
    rejection loop would skew the mixture), then the rows are shuffled.
    Returns ``{"L": {...}, "R": {...}, "comp": int8 array}`` with numpy
    float64 arrays under ``STATE_KEYS`` (normal = x; B_x shared by L and R)
    and ``comp`` indexing ``COMPONENTS``; ``"accept"`` is the fraction of
    candidates that passed the box and the caps, ``"accept_by_comp"`` the
    same per component.
    """
    cfg = merged_cfg(cfg)
    caps, box = cfg["caps"], cfg["box"]
    counts = component_counts(n, cfg["components"])
    cols = {t: {k: [] for k in ("rho", "p", "v", "bt")} for t in "LR"}
    Bns, comps, acc = [], [], {}
    tried_all = 0
    for c, nc in enumerate(counts):
        have = tried = 0
        for _ in range(max_rounds):
            if have >= nc:
                break
            m = max(2 * (nc - have), 1024)
            L, R, Bn = _draw(rng, m, cfg, c)
            ok = (_in_box(box, *R) & _caps_ok(caps, gamma, *L[:3], Bn, L[3])
                  & _caps_ok(caps, gamma, *R[:3], Bn, R[3]))
            idx = np.flatnonzero(ok)[: nc - have]
            for t, S in (("L", L), ("R", R)):
                cols[t]["rho"].append(S[0][idx])
                cols[t]["p"].append(S[1][idx])
                cols[t]["v"].append(S[2][:, idx])
                cols[t]["bt"].append(S[3][:, idx])
            Bns.append(Bn[idx])
            comps.append(np.full(idx.size, c, dtype=np.int8))
            have += idx.size
            tried += m
        if have < nc:
            raise RuntimeError(f"{COMPONENTS[c]}: only {have}/{nc} rows "
                               f"accepted: the caps exclude almost everything")
        acc[COMPONENTS[c]] = have / max(tried, 1)
        tried_all += tried

    perm = rng.permutation(n)
    out = {}
    for t in "LR":
        rho = np.concatenate(cols[t]["rho"])[perm]
        p = np.concatenate(cols[t]["p"])[perm]
        v = np.concatenate(cols[t]["v"], axis=1)[:, perm]
        bt = np.concatenate(cols[t]["bt"], axis=1)[:, perm]
        out[t] = dict(rho=rho, p=p, vx=v[0], vy=v[1], vz=v[2], By=bt[0],
                      Bz=bt[1])
    Bn = np.concatenate(Bns)[perm]
    out["L"]["Bx"] = Bn
    out["R"]["Bx"] = Bn.copy()
    out["comp"] = np.concatenate(comps)[perm]
    out["accept"] = n / max(tried_all, 1)
    out["accept_by_comp"] = acc
    return out
