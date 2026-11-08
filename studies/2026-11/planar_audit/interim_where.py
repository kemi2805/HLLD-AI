"""Interim: WHERE the paired runs' lost faces differ, over the steps both have
closed.  Per physical face, the number of sweeps in which it was lost
(control, filter); how concentrated the difference is (a face lost in every
sweep for 100 steps counts 600), and a map of it.

    python interim_where.py runs_adm64 OUTDIR [r2d ot rotor]
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from adm_compare64 import contours, extent, unpack
from interim_adm64 import steps


def lost_map(d, nsteps):
    meta = json.load(open(os.path.join(d, "run_meta.json")))
    s0 = np.load(sorted(glob.glob(os.path.join(d, "snap_*.npz")))[0], allow_pickle=True)
    nx, ny = s0["rho"].shape
    ng = int(meta.get("ng", 2))
    nxt, nyt = nx + 2 * ng, ny + 2 * ng
    shapes = {0: ((nxt + 1, nyt), (slice(ng, ng + nx + 1), slice(ng, ng + ny))),
              1: ((nxt, nyt + 1), (slice(ng, ng + nx), slice(ng, ng + ny + 1)))}
    L = {0: np.zeros((nx + 1, ny), np.int64), 1: np.zeros((nx, ny + 1), np.int64)}
    for f in sorted(glob.glob(os.path.join(d, "harvest", "coverage_*.npz"))):
        z = np.load(f)
        for r in range(z["sweep"].shape[0]):
            s, idir, n = int(z["sweep"][r]), int(z["idir"][r]), int(z["n"][r])
            if s // 6 >= nsteps:
                continue
            shp, sl = shapes[idir]
            m = {k: unpack(z[k][r], n).reshape(shp)[sl] for k in ("attempted", "exact", "linear_fallback")}
            L[idir] += m["attempted"] & ~m["exact"] & ~m["linear_fallback"]
    return L, s0


def main():
    root, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    probs = sys.argv[3:] or ("r2d", "ot", "rotor")
    fig, ax = plt.subplots(1, len(probs), figsize=(4.6 * len(probs), 4.3), constrained_layout=True, squeeze=False)
    for j, p in enumerate(probs):
        C, T = os.path.join(root, p + "_ctl"), os.path.join(root, p + "_adm")
        n = min(steps(C), steps(T))
        Lc, s0 = lost_map(C, n)
        Lt, _ = lost_map(T, n)
        d = np.concatenate([(Lt[0] - Lc[0]).ravel(), (Lt[1] - Lc[1]).ravel()])
        tot_c = Lc[0].sum() + Lc[1].sum()
        tot_t = Lt[0].sum() + Lt[1].sum()
        srt = np.sort(d)
        print("== %s (steps 1-%d): lost %d -> %d (%+d); faces lost more %d (+%d), fewer %d (%d)" % (
            p, n, tot_c, tot_t, tot_t - tot_c, (d > 0).sum(), d[d > 0].sum(), (d < 0).sum(), d[d < 0].sum()))
        print("   largest per-face increases: %s" % srt[::-1][:12].tolist())
        print("   largest per-face decreases: %s" % srt[:12].tolist())
        for k in (5, 10, 20, 50):
            print("   top %2d faces carry %+d of the +%d increase" % (k, srt[::-1][:k].sum(), d[d > 0].sum()))
        nx, ny = Lc[1].shape[0], Lc[0].shape[1]
        M = np.zeros((2 * nx + 1, 2 * ny + 1))
        M[0::2, 1::2] = Lt[0] - Lc[0]
        M[1::2, 0::2] = Lt[1] - Lc[1]
        dm = max(1, np.abs(M).max())
        im = ax[0, j].imshow(np.ma.masked_equal(M, 0).T, origin="lower", extent=extent(s0), cmap="RdBu_r",
                             norm=TwoSlopeNorm(0, -dm, dm), interpolation="nearest")
        snaps = sorted(glob.glob(os.path.join(T, "snap_*.npz")))
        contours(ax[0, j], np.load(snaps[-1], allow_pickle=True))
        ax[0, j].set_title("%s, steps 1-%d: filter - control lost\n%d -> %d (%+d)" % (
            p, n, tot_c, tot_t, tot_t - tot_c), fontsize=10)
        ax[0, j].set_xticks([])
        ax[0, j].set_yticks([])
        fig.colorbar(im, ax=ax[0, j], shrink=0.8, pad=0.01).set_label("sweeps lost: more (red) / fewer (blue)")
    fig.savefig(os.path.join(out, "interim_lost_diff.png"), dpi=140)
    print("wrote", os.path.join(out, "interim_lost_diff.png"))


if __name__ == "__main__":
    main()
