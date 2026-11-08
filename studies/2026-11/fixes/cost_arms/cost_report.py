#!/usr/bin/env python3
"""The timing arms side by side: wall of the steps, wall per solver stage
(tpatch.py), faces per rung and lost (coverage maps).

    python3 cost_report.py DIR [DIR ...]      (DIR = results/fixes/cost_arms/<arm>)"""
import glob, json, os, re, sys
import numpy as np
K = ("attempted", "exact", "seven", "planar5", "coplanar7", "planar4")
NAMES = {"_run_pipeline": "seven-wave Newton", "_planar_and_ray": "five-wave", "_planar_ladder": "five-wave ladder",
         "_coplanar7_and_ray": "planar seven-wave", "_planar4_and_ray": "four-unknown",
         "_continuation_and_ray": "continuation"}


def steps(d):
    logs = [os.path.join(d, "run.log")] + glob.glob("/mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_cost-%s_*.out" % os.path.basename(d))
    for f in logs:
        if not os.path.exists(f):
            continue
        w = {}
        for line in open(f, errors="ignore"):
            m = re.match(r"\s+step\s+(\d+) .*\((\d+)s\)", line)
            if m:
                w[int(m.group(1))] = int(m.group(2))
        if w:
            return w
    return {}


def cov(d):
    tot = np.zeros(7, dtype=int)
    for f in sorted(glob.glob(d + "/harvest/coverage_*.npz")):
        z = np.load(f)
        for j in range(len(z["sweep"])):
            n = int(z["n"][j])
            u = {k: np.unpackbits(z[k][j])[:n].astype(bool) for k in K if k in z.files}
            row = [u[k].sum() if k in u else 0 for k in K] + [(u["attempted"] & ~u["exact"]).sum()]
            tot += np.array(row)
    return tot


rows = []
for d in sys.argv[1:]:
    w = steps(d)
    t = json.load(open(d + "/rung_times.json")) if os.path.exists(d + "/rung_times.json") else {}
    c = cov(d)
    rows.append((os.path.basename(d), w, t, c))
print("%-14s %8s %9s %9s   %s" % ("arm", "steps", "1st step", "the rest", "faces: attempted / seven / five-wave / planar7 / four-unknown / lost"))
for a, w, t, c in rows:
    if w:
        s0 = min(w)                      # the first step's wall includes the start-up
        print("%-14s %3d-%3d %9d %9d   %s" % (a, s0, max(w), w[s0], w[max(w)] - w[s0],
                                            " / ".join(str(int(x)) for x in c[[0, 2, 3, 4, 5, 6]])))
names = sorted({k for _, _, t, _ in rows for k in t}, key=lambda k: list(NAMES).index(k) if k in NAMES else 99)
print()
print("%-22s" % "stage (wall s, calls, lanes)" + "".join("%26s" % a for a, _, _, _ in rows))
for k in names:
    line = "%-22s" % NAMES.get(k, k)
    for _, _, t, _ in rows:
        v = t.get(k)
        line += "%26s" % ("%.0f s  %d  %d" % (v["wall"], v["calls"], v["lanes"]) if v else "-")
    print(line)
