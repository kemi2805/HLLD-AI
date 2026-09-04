#!/usr/bin/env python3
"""
The 1D shock-tube gate: exact flux against HLLD, both against the truth.

This is the plan's GO/NO-GO test.  Everything downstream -- MPI, the rotor
production runs -- is gated on it, because it answers the one question the
solver engineering cannot: is an exact flux actually STABLE AND BENEFICIAL
inside a shock-capturing scheme, or merely computable?

The reference is not a converged high-resolution run.  A shock tube IS a
single Riemann problem, so its exact solution at time ``t`` is the solution
sampled along the ray ``xi = (x - x0)/t`` -- available analytically from the
same exact solver, at every cell, with no resolution study and no
discretisation error of its own.  That makes the L1 comparison a measurement
against truth rather than against another approximation.

Reported per tube:

  L1 error per variable, exact vs HLLD.  The plan's bar is "no worse".
  Contact width in cells.  The plan's bar is "sharper" -- this is where an
    exact solver should win, since HLLD linearises the contact away.
  Fallback fraction, because a run that fell back everywhere is HLLD wearing
    a different name, and the comparison would be meaningless.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_RMHD = "/Users/miler/Codes/rmhd_final"
if _RMHD not in sys.path:
    sys.path.append(_RMHD)

from src.physics.driver import run                      # noqa: E402
from src.physics.eos import hybrid_eos                  # noqa: E402
from src.physics.hlld import LAST_DIAG                  # noqa: E402

VARS = ("rho", "p", "vx", "vy", "vz", "By", "Bz")


def exact_profile(cfg, x, t):
    """The analytic solution at time ``t``, sampled at cell centres ``x``.

    One Riemann problem, read along the ray.  Everything here runs in the
    solver's own frame (normal component first, TOTAL pressure).
    """
    from batched.ray_scalar import claim_rmhd_namespace
    claim_rmhd_namespace()
    from eos import set_eos
    set_eos("ideal")
    import ml_guess as mg
    from batched import api as API
    from batched import ml_b as MB
    from batched import rarefaction_b as RB
    from batched import ray_b as RAY
    from batched import wave_speeds_b as WB

    rc = cfg["run"]
    g = float(cfg["eos"]["gamma"])
    L, R = rc["primL"], rc["primR"]
    x0 = float(rc["x_interface"])

    def seven(P):
        v2 = P["vx"] ** 2 + P["vy"] ** 2 + P["vz"] ** 2
        W2 = 1.0 / (1.0 - v2)
        eta = P["Bx"] * P["vx"] + P["By"] * P["vy"] + P["Bz"] * P["vz"]
        b2 = (P["Bx"] ** 2 + P["By"] ** 2 + P["Bz"] ** 2) / W2 + eta ** 2
        return [P["rho"], P["p"] + 0.5 * b2, P["vx"], P["vy"], P["vz"],
                P["By"], P["Bz"]], P["Bx"]

    l7, Bn = seven(L)
    r7, _ = seven(R)
    n = x.size

    # Solve the Riemann problem ONCE.  A shock tube is a single problem read
    # at many rays, so the wave structure is shared by every cell and only xi
    # differs -- solving it per cell would be N times the work for one answer.
    #
    # The batch dimension is used for a MULTISTART instead: lane 0 takes the
    # ML seed, the rest take jitters of it.  The retry ladder cannot serve
    # here, because it derives its jitter from each lane's own features and
    # every lane is the same problem -- identical features, identical keys,
    # identical "retries".  The mechanism that makes the ladder work degenerates
    # exactly when the interfaces are copies of one another.
    #
    # A multistart also covers the gamma mismatch: the checkpoint is trained at
    # gamma = 5/3, so on a gamma = 2 or 4/3 tube the ML seed is out of
    # distribution and the spread of starts is what finds the root.
    # Escalating multistart, not a flat 32.  The cost of a flat sweep is the
    # starts that never converge running the full iteration budget: 8 minutes
    # for what is fundamentally ONE Riemann problem.  Cheap rungs first, and
    # stop the moment any start lands -- the same first-convergent-wins ladder
    # the rest of this project uses.
    model, scaler = mg.load(f"{_RMHD}/data/ml_guess_gamma53.pt")
    rng = np.random.default_rng(0)
    res1 = None
    diag = {}
    for K, mi, sigma in ((4, 40, 0.25), (16, 60, 0.5), (48, 100, 1.0)):
        left1 = [np.full(K, c) for c in l7]
        right1 = [np.full(K, c) for c in r7]
        Bn1 = np.full(K, Bn)
        s0, _ = MB.predict_unk6(model, scaler, np.stack(left1, axis=1),
                                np.stack(right1, axis=1), Bn1)
        seeds = []
        for c in range(6):
            col = np.full(K, s0[c][0])
            col[1:] += sigma * rng.standard_normal(K - 1)
            seeds.append(col)
        r, diag = API.make_solver(g)(left1, right1, Bn1, seed6=seeds,
                                     accuracy=1e-9, max_iter=mi)
        if r["converged"].any():
            res1 = r
            break
    if res1 is None:
        raise RuntimeError(
            f"the reference Riemann solve did not converge from any start.\n"
            f"  gamma={g}.  The ML checkpoint covers gamma=5/3, "
            f"Bx in [0.01, 1.01], Bt in [0, 1], W in [1, 2]; this tube is\n"
            f"  outside it, so the seed is a cold start and the multistart is\n"
            f"  doing the work.  classes: "
            f"{ {k: v for k, v in diag.items() if k.startswith('n_') and v} }")
    good = np.flatnonzero(res1["converged"])
    j = int(good[0])

    # broadcast the one solution back over the cells, and read it at each ray
    left = [np.full(n, c) for c in l7]
    right = [np.full(n, c) for c in r7]
    Bnv = np.full(n, Bn)
    zones = [[np.full(n, z[k][j]) for k in range(7)] for z in res1["zones"]]
    VsLv = [np.full(n, v[j]) for v in res1["VsLv"]]
    VsRv = [np.full(n, v[j]) for v in res1["VsRv"]]

    idx = {"LF": 0, "LS": 1, "RS": 2, "RF": 3}

    def xi_fn(state, switch, B, gg=g):
        eig, _, _, ok = WB.xi_all(*state, B, gg)
        return np.where(ok, eig[:, idx[switch]], np.nan)

    fan_p, fan_n = RB.make_integrators(g, lambda s, sw, B, gg: xi_fn(s, sw, B))
    xi = (x - x0) / t
    st, region, err = RAY.state_at_xi(left, right, zones, VsLv, VsRv, Bnv, g,
                                      xi_fn, fan_p, fan_n, xi_target=xi)
    rho, Ptot, vn, vt1, vt2, Bt1, Bt2 = st
    v2 = vn ** 2 + vt1 ** 2 + vt2 ** 2
    W2 = 1.0 / np.maximum(1.0 - v2, 1e-300)
    eta = Bnv * vn + Bt1 * vt1 + Bt2 * vt2
    b2 = (Bnv ** 2 + Bt1 ** 2 + Bt2 ** 2) / W2 + eta ** 2
    return dict(rho=rho, p=Ptot - 0.5 * b2, vx=vn, vy=vt1, vz=vt2,
                By=Bt1, Bz=Bt2), region, err


def contact_width(x, rho, x_contact, frac=0.8):
    """Cells spanned by the central ``frac`` of the density jump at the CD.

    A sharper contact is the specific place an exact solver should beat HLLD,
    which linearises the contact away.
    """
    i = int(np.argmin(np.abs(x - x_contact)))
    w = max(4, len(x) // 12)
    lo, hi = max(0, i - w), min(len(x), i + w + 1)
    seg = rho[lo:hi]
    jump = seg.max() - seg.min()
    if jump <= 1e-12:
        return float("nan")
    band = (seg > seg.min() + 0.5 * (1 - frac) * jump) & \
           (seg < seg.max() - 0.5 * (1 - frac) * jump)
    return float(band.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--ncells", type=int, default=200)
    ap.add_argument("--solvers", default="hlld,exact")
    args = ap.parse_args()

    for cpath in args.configs:
        cfg0 = yaml.safe_load(open(cpath))
        name = Path(cpath).stem
        t_end = float(cfg0["run"]["t_end"])
        print(f"\n{'='*72}\n{name}   ncells={args.ncells}  t_end={t_end}")

        out = {}
        for solver in args.solvers.split(","):
            cfg = yaml.safe_load(open(cpath))
            cfg["grid"]["ncells"] = args.ncells
            cfg["run"]["solver"] = solver
            cfg["output"]["dir"] = f"/tmp/tube_{name}_{solver}"
            cfg["output"]["every_n_steps"] = 10 ** 9
            t0 = time.perf_counter()
            _t, grid, prims, _cons = run(cfg)
            dt = time.perf_counter() - t0
            d = dict(LAST_DIAG)
            out[solver] = dict(prims=prims, grid=grid, wall=dt,
                               frac_exact=d.get("frac_exact", 0.0))
            print(f"  {solver:6s} {dt:7.1f}s   "
                  f"exact fraction {d.get('frac_exact', 0.0)*100:5.1f}%")

        g = out[list(out)[0]]["grid"]
        ng = getattr(g, "ng", 2)
        x = g.x[ng:-ng].numpy()
        ref, region, rerr = exact_profile(cfg0, x, t_end)

        print(f"\n  {'variable':<8}" + "".join(f"{s:>14}" for s in out))
        for v in VARS:
            row = f"  {v:<8}"
            for s in out:
                p = out[s]["prims"][v][ng:-ng].numpy()
                row += f"{np.abs(p - ref[v]).mean():14.4e}"
            print(row)
        # the contact sits where the exact solution's CD is
        xc = x[int(np.argmax(np.abs(np.diff(ref["rho"]))))] if len(x) > 1 else 0.0
        row = f"  {'CD cells':<8}"
        for s in out:
            p = out[s]["prims"]["rho"][ng:-ng].numpy()
            row += f"{contact_width(x, p, xc):14.1f}"
        print(row + "     (fewer is sharper)")


if __name__ == "__main__":
    main()
