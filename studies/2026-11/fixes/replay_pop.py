#!/usr/bin/env python
"""Replay a failure-ledger population through the production exact-flux path
and keep, per lane, the verdict, the rung that answered, p* and the flux.

    python replay_pop.py POP.npz OUT.npz [--lanes a:b]

The env decides the code path (RMHD_* switches, kernels.env, the networks);
run it twice with one switch changed for a paired comparison.  The lab-frame
states are rebuilt with x as the normal, as failure_ledger.collect_harvest
does, and the call is the run's: tau_weak 1e-2, tau_bt 1e-9, 2 retries,
40 iterations.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

HLLD = os.environ.get("HLLD_DIR", "/mnt/rafast/miler/codes/rotor2d/HLLD")
sys.path.insert(0, HLLD)
from rmhd.eos import set_eos                                     # noqa: E402
set_eos("ideal")
import rmhd                                                      # noqa: E402
from src.physics import exact_flux as EF                         # noqa: E402
from src.physics.eos import hybrid_eos                           # noqa: E402
from src.physics.hlld import LAST_DIAG                           # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pop")
    ap.add_argument("out")
    ap.add_argument("--lanes", default=None)
    ap.add_argument("--chunk", type=int, default=256)
    a = ap.parse_args()
    print("rmhd from", os.path.dirname(rmhd.__file__), " HLLD from", HLLD, flush=True)
    print("env:", {k: v for k, v in os.environ.items() if k.startswith("RMHD_") or k == "RETRIES"}, flush=True)
    p = np.load(a.pop, allow_pickle=True)
    UL, UR = p["UL"], p["UR"]
    idx = np.arange(len(UL))
    if a.lanes:
        lo, hi = (int(x) for x in a.lanes.split(":"))
        idx = idx[lo:hi]
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

    n = len(idx)
    out = dict(lane=idx, exact=np.zeros(n, bool), attempted=np.zeros(n, bool),
               src=np.zeros(n, np.int8), reason7=np.full(n, -9, np.int8),
               reason5=np.full(n, -9, np.int8), pstar=np.full(n, np.nan),
               iters7=np.zeros(n, np.int64), iters5=np.zeros(n, np.int64),
               c7=np.zeros(n, bool), c7_stage=np.zeros(n, np.int8),
               c7_rot=np.zeros((2, n), bool))
    F_all = None
    t0 = time.time()
    for c0 in range(0, n, a.chunk):
        j = idx[c0:c0 + a.chunk]
        sL, sR = lab(UL[j]), lab(UR[j])
        F, U, ps = EF.exact_flux_batched(sL, sR, eos, idir=0, tau_weak=1e-2,
                                         tau_bt=1e-9, n_retries=2, max_iter=40)
        d = dict(LAST_DIAG)
        sel = np.asarray(d["sel"])
        ex = np.asarray(d["exact_mask"].numpy() if hasattr(d["exact_mask"], "numpy") else d["exact_mask"])
        exs = ex[sel] if ex.size == len(j) else ex
        k = c0 + sel
        out["attempted"][k] = True
        out["exact"][k] = exs
        src = np.zeros(sel.size, np.int8)
        src[np.asarray(d["reason7"]) == 0] = 1
        src[np.asarray(d["planar5_mask"], bool)] = 2
        if "planar4_mask" in d:
            src[np.asarray(d["planar4_mask"], bool)] = 3
        if "coplanar7_mask" in d:
            c7 = np.asarray(d["coplanar7_mask"], bool) & (np.asarray(d["reason7"]) == 0)
            src[c7] = 4                       # the planar form of the seven-wave stage
            out["c7"][k] = c7 & exs
            out["c7_stage"][k] = np.asarray(d["coplanar7_stage"])
            out["c7_rot"][:, k] = np.asarray(d["coplanar7_rot"])
        if "coplanar7_late_mask" in d:
            # RMHD_COPLANAR7=late: the same solver as a rung after the five-wave one
            c7l = np.asarray(d["coplanar7_late_mask"], bool)
            src[c7l] = 4
            out["c7"][k] |= c7l & exs
            out["c7_stage"][k] = np.where(c7l, np.asarray(d["coplanar7_late_stage"]), out["c7_stage"][k])
            out["c7_rot"][:, k] = np.where(c7l, np.asarray(d["coplanar7_late_rot"]), out["c7_rot"][:, k])
        src[~exs] = 0
        out["src"][k] = src
        out["reason7"][k] = np.asarray(d["reason7"])
        out["reason5"][k] = np.asarray(d["reason5"])
        out["iters7"][k] = np.asarray(d["seven_n_iter_total"])
        out["iters5"][k] = np.asarray(d["planar_n_iter"])
        out["pstar"][c0:c0 + len(j)] = ps.reshape(-1).numpy()
        if F_all is None:
            F_all = {kk: np.full(n, np.nan) for kk in F}
        for kk, v in F.items():
            F_all[kk][c0:c0 + len(j)] = v.reshape(-1).numpy()
        print("  %5d / %d: attempted %d exact %d  (%.0f s)" % (
            min(c0 + a.chunk, n), n, sel.size, exs.sum(), time.time() - t0), flush=True)
    out.update({"F_" + k: v for k, v in F_all.items()})
    np.savez_compressed(a.out, **out)
    grp = p["group"][idx]
    print("failing lanes exact: %d / %d;  controls exact: %d / %d;  wall %.0f s" % (
        out["exact"][grp == 1].sum(), (grp == 1).sum(), out["exact"][grp == 0].sum(),
        (grp == 0).sum(), time.time() - t0))


if __name__ == "__main__":
    main()
