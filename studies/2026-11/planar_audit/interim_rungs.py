"""Interim: at the faces the filter run loses MORE often than the control
(rotor/OT/4Q, common steps), what did the control answer them with?"""
import glob, json, os, sys
import numpy as np
from adm_compare64 import unpack
from interim_adm64 import steps

K = ("attempted", "exact", "seven", "planar5", "planar4", "linear_fallback")


def maps(d, nsteps):
    meta = json.load(open(os.path.join(d, "run_meta.json")))
    s0 = np.load(sorted(glob.glob(os.path.join(d, "snap_*.npz")))[0], allow_pickle=True)
    nx, ny = s0["rho"].shape
    ng = int(meta.get("ng", 2))
    nxt, nyt = nx + 2 * ng, ny + 2 * ng
    shapes = {0: ((nxt + 1, nyt), (slice(ng, ng + nx + 1), slice(ng, ng + ny))),
              1: ((nxt, nyt + 1), (slice(ng, ng + nx), slice(ng, ng + ny + 1)))}
    M = {k: {0: 0, 1: 0} for k in K + ("lost",)}
    for f in sorted(glob.glob(os.path.join(d, "harvest", "coverage_*.npz"))):
        z = np.load(f)
        for r in range(z["sweep"].shape[0]):
            s, idir, n = int(z["sweep"][r]), int(z["idir"][r]), int(z["n"][r])
            if s // 6 >= nsteps:
                continue
            shp, sl = shapes[idir]
            m = {k: unpack(z[k][r], n).reshape(shp)[sl].astype(np.int64) for k in K}
            m["lost"] = m["attempted"] * (1 - m["exact"]) * (1 - m["linear_fallback"])
            for k in m:
                M[k][idir] = M[k][idir] + m[k]
    return {k: np.concatenate([M[k][0].ravel(), M[k][1].ravel()]) for k in M}


for p in sys.argv[2:] or ("r2d", "ot", "rotor"):
    C, T = os.path.join(sys.argv[1], p + "_ctl"), os.path.join(sys.argv[1], p + "_adm")
    n = min(steps(C), steps(T))
    c, t = maps(C, n), maps(T, n)
    d = t["lost"] - c["lost"]
    for nm, sel in (("filter loses more", d > 0), ("filter loses fewer", d < 0)):
        print("== %s, %s at %d faces (%+d sweeps)" % (p, nm, sel.sum(), d[sel].sum()))
        for run, m in (("control", c), ("filter ", t)):
            print("   %s there: attempted %6d  lost %5d  seven %6d  planar5 %5d  planar4 %5d" % (
                run, m["attempted"][sel].sum(), m["lost"][sel].sum(), m["seven"][sel].sum(),
                m["planar5"][sel].sum(), m["planar4"][sel].sum()))
