#!/usr/bin/env python
"""Replay a 64^2 run's Newton-refused faces (diag_4q/pop_sweeps.py) one sweep
at a time, as the run calls the exact flux, and keep per lane the rung that
answered and the planar seven-wave rung's verdict, stage and iterations, and
per sweep the wall of each stage (the seven-wave pipeline and each rescue
rung, timed from the main process like tpatch.py).

    python replay_sweeps.py POP.npz OUT.npz [--every K]   (every K-th sweep)

The env decides the code path (RMHD_* switches, kernels.env, the networks,
the pool); the call is the run's: tau_weak 1e-2, tau_bt 1e-9, 2 retries,
40 iterations."""
import argparse
import collections
import os
import sys
import time

import numpy as np
import torch

HLLD = os.environ.get("HLLD_DIR", "/mnt/rafast/miler/codes/rotor2d/HLLD")
sys.path.insert(0, HLLD)
from rmhd.eos import set_eos                                     # noqa: E402
set_eos("ideal")
from src.physics import exact_flux as EF                         # noqa: E402
from src.physics import exact_pool as XP                         # noqa: E402
from src.physics.eos import hybrid_eos                           # noqa: E402
from src.physics.hlld import LAST_DIAG                           # noqa: E402

T = collections.defaultdict(float)
_map_lanes, _pipeline = XP.map_lanes, EF._run_pipeline


def map_lanes(name, gamma, sL, sR, Bn, **kw):
    t0 = time.perf_counter()
    try:
        return _map_lanes(name, gamma, sL, sR, Bn, **kw)
    finally:
        T[name] += time.perf_counter() - t0


def _run_pipeline(*a, **kw):
    t0 = time.perf_counter()
    try:
        return _pipeline(*a, **kw)
    finally:
        T["_run_pipeline"] += time.perf_counter() - t0


XP.map_lanes, EF._run_pipeline = map_lanes, _run_pipeline
STAGES = ("_run_pipeline", "_planar_and_ray", "_planar_ladder", "_coplanar7_and_ray",
          "_planar4_and_ray", "_continuation_and_ray")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pop")
    ap.add_argument("out")
    ap.add_argument("--every", type=int, default=1)
    a = ap.parse_args()
    print("env:", {k: v for k, v in sorted(os.environ.items()) if k.startswith("RMHD_")}, flush=True)
    p = np.load(a.pop, allow_pickle=True)
    UL, UR, SW = p["UL"], p["UR"], p["sweep"]
    gam = float(p["gamma"])
    eos = hybrid_eos(K=0.0, gamma=gam, gamma_th=gam)

    def lab(U):
        rho, Pt, vx, vy, vz, By, Bz, Bx = (U[:, j] for j in range(8))
        W2 = 1.0 / (1.0 - (vx * vx + vy * vy + vz * vz))
        eta = Bx * vx + By * vy + Bz * vz
        b2 = (Bx * Bx + By * By + Bz * Bz) / W2 + eta * eta
        t = lambda v: torch.tensor(np.ascontiguousarray(v), dtype=torch.float64)
        d = {"rho": t(rho), "vx": t(vx), "vy": t(vy), "vz": t(vz),
             "p": t(Pt - 0.5 * b2), "Bx": t(Bx), "By": t(By), "Bz": t(Bz)}
        d["eps"] = eos.eps__press_rho(d["p"], d["rho"])
        return d

    sweeps = np.unique(SW)[::a.every]
    keep = np.isin(SW, sweeps)
    idx = np.flatnonzero(keep)
    n = idx.size
    out = dict(lane=idx, sweep_of=SW[idx], exact=np.zeros(n, bool), src=np.zeros(n, np.int8),
               c7_reason=np.full(n, -9, np.int8), c7_stage=np.zeros(n, np.int8),
               c7_n_iter=np.zeros(n, np.int64), c7_rot=np.zeros((2, n), bool))
    wall = np.zeros((sweeps.size, len(STAGES) + 1))
    pos = {k: i for i, k in enumerate(idx)}
    t_all = time.time()
    for si, s in enumerate(sweeps):
        j = np.flatnonzero(SW == s)
        before = dict(T)
        t0 = time.perf_counter()
        EF.exact_flux_batched(lab(UL[j]), lab(UR[j]), eos, idir=0, tau_weak=1e-2,
                              tau_bt=1e-9, n_retries=2, max_iter=40)
        wall[si, -1] = time.perf_counter() - t0
        for q, k in enumerate(STAGES):
            wall[si, q] = T.get(k, 0.0) - before.get(k, 0.0)
        d = dict(LAST_DIAG)
        sel = np.asarray(d["sel"])
        ex = np.asarray(d["exact_mask"].numpy() if hasattr(d["exact_mask"], "numpy") else d["exact_mask"])
        exs = ex[sel] if ex.size == len(j) else ex
        k = np.array([pos[i] for i in j[sel]], dtype=int)
        src = np.zeros(sel.size, np.int8)
        src[np.asarray(d["reason7"]) == 0] = 1
        src[np.asarray(d["planar5_mask"], bool)] = 2
        if "planar4_mask" in d:
            src[np.asarray(d["planar4_mask"], bool)] = 3
        if "coplanar7_late_mask" in d:
            c7l = np.asarray(d["coplanar7_late_mask"], bool)
            src[c7l] = 4
            out["c7_reason"][k] = np.asarray(d["coplanar7_late_reason"])
            out["c7_stage"][k] = np.asarray(d["coplanar7_late_stage"])
            out["c7_n_iter"][k] = np.asarray(d["coplanar7_late_n_iter"])
            out["c7_rot"][:, k] = np.asarray(d["coplanar7_late_rot"])
        src[~exs] = 0
        out["src"][k] = src
        out["exact"][k] = exs
        if si % 10 == 0:
            print("  sweep %d (%d/%d): %d lanes, %.1f s; total %.0f s" % (
                s, si + 1, sweeps.size, len(j), wall[si, -1], time.time() - t_all), flush=True)
    out.update(sweeps=sweeps, wall=wall, stages=np.array(STAGES + ("total",)))
    np.savez_compressed(a.out, **out)
    tot = wall.sum(axis=0)
    print("wall: " + ", ".join("%s %.0f s" % (k, v) for k, v in zip(STAGES + ("total",), tot) if v > 0))
    c = collections.Counter(int(x) for x in out["src"])
    print("answered by: lost %d seven %d five-wave %d four-unknown %d planar-seven %d" % (
        c[0], c[1], c[2], c[3], c[4]))
    if (out["c7_reason"] != -9).any():
        off = out["c7_reason"] != 8
        off &= out["c7_reason"] != -9
        print("planar-seven offered %d: answered %d, by stage %s; n_iter of the answered: median %d, max %d; "
              "of the refused: median %d" % (
                  off.sum(), (off & (out["src"] == 4)).sum(),
                  dict(collections.Counter(int(x) for x in out["c7_stage"][off & (out["src"] == 4)])),
                  np.median(out["c7_n_iter"][off & (out["src"] == 4)]) if (off & (out["src"] == 4)).any() else -1,
                  out["c7_n_iter"][off & (out["src"] == 4)].max() if (off & (out["src"] == 4)).any() else -1,
                  np.median(out["c7_n_iter"][off & (out["src"] != 4)]) if (off & (out["src"] != 4)).any() else -1))
    print("done in %.0f s" % (time.time() - t_all))


if __name__ == "__main__":
    main()
