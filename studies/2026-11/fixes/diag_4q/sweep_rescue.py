#!/usr/bin/env python3
"""Per sweep of a 64^2 run, in a block of steps: how many faces reach the
rungs after the five-wave solver (planar seven-wave + four-unknown + lost),
and in how many sweeps each rung has lanes at all.

    python3 sweep_rescue.py RUN STEP_FIRST STEP_LAST"""
import glob, sys
import numpy as np
R = "/mnt/rafast/miler/codes/rotor2d/HLLD/results"
run, s0, s1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
PER = 6
K = ("attempted", "exact", "coplanar7", "planar4")
rows = {}
for f in sorted(glob.glob(R + "/" + run + "/harvest/coverage_*.npz")):
    z = np.load(f)
    for j in range(len(z["sweep"])):
        sw = int(z["sweep"][j])
        if not (s0 <= sw // PER + 1 <= s1):
            continue
        n = int(z["n"][j])
        u = {k: np.unpackbits(z[k][j])[:n].astype(bool) if k in z.files else np.zeros(n, bool) for k in K}
        rows[sw] = (u["coplanar7"].sum(), u["planar4"].sum(), (u["attempted"] & ~u["exact"]).sum())
a = np.array([rows[k] for k in sorted(rows)])
late = a.sum(axis=1)
print("%s steps %d-%d: %d sweeps" % (run, s0, s1, len(a)))
print("  faces after the five-wave solver per sweep: mean %.1f, median %d, max %d; sweeps with none %d" % (
    late.mean(), np.median(late), late.max(), (late == 0).sum()))
print("  sweeps with planar-seven answers %d, with four-unknown answers %d, with lost %d" % (
    (a[:, 0] > 0).sum(), (a[:, 1] > 0).sum(), (a[:, 2] > 0).sum()))
print("  sweeps where some face got past the planar-seven rung (four-unknown or lost): %d" % (
    ((a[:, 1] + a[:, 2]) > 0).sum()))
h = np.bincount(np.minimum(late, 20))
print("  histogram of faces per sweep (0..20+):", h.tolist())
