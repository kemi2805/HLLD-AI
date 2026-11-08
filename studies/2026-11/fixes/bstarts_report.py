#!/usr/bin/env python
"""Summarise b_starts shards: recovered lost faces by loss type, which start
answered, the steering, shocks and their entropy; controls' agreement."""
import glob
import sys
import numpy as np

D = sys.argv[1]
pop = np.load("/Users/miler/Codes/claude_scratch/r2d_retrain/anatomy/data/ledger/population.npz", allow_pickle=True)
lt = np.load("/Users/miler/Codes/claude_scratch/r2d_retrain/anatomy/data/ledger/lanes_typed.npz")
cols = {}
for f in sorted(glob.glob(D + "/shard_*.npz")):
    z = np.load(f, allow_pickle=True)
    for k in z.files:
        if k == "tried":
            continue
        v = z[k]
        cols.setdefault(k, []).append(v if v.ndim < 2 or k not in ("kinds", "shocks") else v.T)
z = {k: np.concatenate(v) for k, v in cols.items()}
lane = z["lane"]


def beta(U):
    rho, Pt, vx, vy, vz, By, Bz, Bx = (U[:, j] for j in range(8))
    W2 = 1 / (1 - (vx * vx + vy * vy + vz * vz)); eta = Bx * vx + By * vy + Bz * vz
    b2 = (Bx * Bx + By * By + Bz * Bz) / W2 + eta * eta
    return 2 * (Pt - 0.5 * b2) / b2


bmin = np.minimum(beta(pop["UL"][lane]), beta(pop["UR"][lane]))
typ = lt["typ"][lane]
kind = np.where(typ == "A shear/degenerate", "A", np.where(bmin >= 1, "B weak", "B strong"))
lost = ~z["ctrl"]
print("%-9s %5s %9s   by start" % ("type", "lost", "recovered"))
for k in ("A", "B weak", "B strong"):
    m = lost & (kind == k)
    ok = m & z["ok"]
    hw = {}
    for h in z["how"][ok]:
        hw[h] = hw.get(h, 0) + 1
    print("%-9s %5d %5d %4.0f%%   %s" % (k, m.sum(), ok.sum(), 100 * ok.sum() / max(m.sum(), 1), dict(sorted(hw.items(), key=lambda t: -t[1]))))
ok = lost & z["ok"]
sh = z["shocks"][ok]
print("recovered: answers with a shock %d of %d; entropy-falling shock (K ratio < 1 - 1e-9) on %d" % (
    sh.any(axis=1).sum(), ok.sum(), (z["smin"][ok] < 1 - 1e-9).sum()))
kd = z["kinds"][ok]
print("steering of the recovered (L,R): ", {str(tuple(r)): int(((kd == r).all(axis=1)).sum()) for r in np.unique(kd, axis=0)})
c = z["ctrl"]
dp = np.abs(z["p_ans"][c & z["ok"]] - z["p_prod"][c & z["ok"]]) / np.abs(z["p_prod"][c & z["ok"]])
print("controls: %d, answered %d, p* within 1e-6 of production on %d, other root (> 1e-3) on %d, max diff %.1e" % (
    c.sum(), (c & z["ok"]).sum(), (dp < 1e-6).sum(), (dp > 1e-3).sum(), dp.max() if dp.size else 0))
np.save(D + "/recovered_lanes.npy", lane[ok])
