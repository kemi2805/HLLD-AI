"""Interim paired counts while the treatment runs are still going (2026-11-03):
physical faces attempted / exact / lost and the rung that answered, summed over
the steps BOTH runs have closed (the treatment's diag.csv), plus the same in
four equal step windows.  Same face bookkeeping as adm_compare64.load_run
(lost = attempted & ~exact & ~linear_fallback, physical faces only).

    python interim_adm64.py runs_adm64 [r2d ot rotor]
"""
import glob
import json
import os
import sys

import numpy as np

from adm_compare64 import unpack

KEYS = ("attempted", "exact", "lost", "seven", "planar5", "planar4")


def per_step(d, nsteps):
    meta = json.load(open(os.path.join(d, "run_meta.json")))
    s0 = np.load(sorted(glob.glob(os.path.join(d, "snap_*.npz")))[0], allow_pickle=True)
    nx, ny = s0["rho"].shape
    ng = int(meta.get("ng", 2))
    nxt, nyt = nx + 2 * ng, ny + 2 * ng
    shapes = {0: ((nxt + 1, nyt), (slice(ng, ng + nx + 1), slice(ng, ng + ny))),
              1: ((nxt, nyt + 1), (slice(ng, ng + nx), slice(ng, ng + ny + 1)))}
    out = np.zeros((nsteps, len(KEYS)), np.int64)
    for f in sorted(glob.glob(os.path.join(d, "harvest", "coverage_*.npz"))):
        z = np.load(f)
        for r in range(z["sweep"].shape[0]):
            s, idir, n = int(z["sweep"][r]), int(z["idir"][r]), int(z["n"][r])
            if s // 6 >= nsteps:
                continue
            shp, sl = shapes[idir]
            m = {k: unpack(z[k][r], n).reshape(shp)[sl]
                 for k in ("attempted", "exact", "seven", "planar5", "planar4", "linear_fallback")}
            m["lost"] = m["attempted"] & ~m["exact"] & ~m["linear_fallback"]
            out[s // 6] += [int(m[k].sum()) for k in KEYS]
    return out


def steps(d):
    return int(np.atleast_1d(np.genfromtxt(os.path.join(d, "diag.csv"), delimiter=",", names=True)["step"]).size)


def main():
    root = sys.argv[1]
    for p in sys.argv[2:] or ("r2d", "ot", "rotor"):
        C, T = os.path.join(root, p + "_ctl"), os.path.join(root, p + "_adm")
        n = min(steps(C), steps(T))
        c, t = per_step(C, n), per_step(T, n)
        tt = np.atleast_1d(np.genfromtxt(os.path.join(T, "diag.csv"), delimiter=",", names=True)["t"])
        print("== %s: steps 1-%d (t <= %.4f) of the control's %d" % (p, n, tt[n - 1], steps(C)))
        print("   %-9s %10s %10s %7s %8s %8s %8s %8s" % ("run", "attempted", "exact", "lost", "solved%",
                                                      "seven", "planar5", "planar4"))
        for nm, a in (("control", c), ("filter", t)):
            s = a.sum(0)
            print("   %-9s %10d %10d %7d %8.3f %8d %8d %8d" % (nm, s[0], s[1], s[2], 100.0 * s[1] / s[0],
                                                             s[3], s[4], s[5]))
        d = t.sum(0) - c.sum(0)
        print("   %-9s %+10d %+10d %+7d %8s %+8d %+8d %+8d" % ("diff", d[0], d[1], d[2], "", d[3], d[4], d[5]))
        edges = np.linspace(0, n, 5).astype(int)
        print("   lost by step window: " + "   ".join(
            "%d-%d: %d -> %d" % (edges[k] + 1, edges[k + 1], c[edges[k]:edges[k + 1], 2].sum(),
                                 t[edges[k]:edges[k + 1], 2].sum()) for k in range(4)))
        first = np.flatnonzero((c != t).any(axis=1))
        if first.size:
            k = first[0]
            print("   first step whose counts differ: %d  control %s  filter %s" % (
                k + 1, dict(zip(KEYS, c[k].tolist())), dict(zip(KEYS, t[k].tolist()))))


if __name__ == "__main__":
    main()
