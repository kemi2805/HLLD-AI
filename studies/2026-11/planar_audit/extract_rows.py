#!/usr/bin/env python
"""Extract the rows r2d_audit_bad.py flagged (audit_bad_1103.npz) plus control
samples from it1/all.npz (3000 seven-wave, 3000 planar five-wave rows), so the
jump conditions can be checked wave by wave off the cluster (2026-11-03).

    python extract_rows.py RETRAIN_DIR OUT.npz
"""
import sys

import numpy as np

D, OUT = sys.argv[1], sys.argv[2]
rows = np.load(D + "/audit_bad_1103.npz", allow_pickle=True)["rows"]
out = {}
for k in ("src", "att", "t", "res", "crossed", "fcomp", "row"):
    out["meta_" + k] = np.array([r[k] for r in rows])
out["meta_file"] = np.array([r["file"] for r in rows])
UL, UR, Z, S = [], [], [], []
for r in rows:
    z = np.load(D + "/" + r["file"], mmap_mode=None)
    i = int(r["row"])
    UL.append(np.asarray(z["U_L"][i], float))
    UR.append(np.asarray(z["U_R"][i], float))
    Z.append(np.asarray(z["zones"][i], float))
    S.append(np.asarray(z["speeds"][i], float))
out.update(bad_U_L=np.array(UL), bad_U_R=np.array(UR), bad_zones=np.array(Z),
           bad_speeds=np.array(S))
za = np.load(D + "/it1/all.npz")
src = np.asarray(za["source"])
rng = np.random.default_rng(1103)
for s in (0, 1):
    i = np.flatnonzero(src == s)
    pick = np.sort(rng.choice(i, 3000, replace=False))
    for k in ("U_L", "U_R", "zones", "speeds", "attempts", "t"):
        out["all%d_%s" % (s, k)] = np.asarray(za[k])[pick]
    out["all%d_row" % s] = pick
np.savez_compressed(OUT, **out)
print("wrote", OUT, {k: v.shape for k, v in out.items()})
