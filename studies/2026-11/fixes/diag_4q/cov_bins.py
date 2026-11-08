#!/usr/bin/env python3
"""Faces answered by each rung per block of steps (summed over the block's
sweeps), for the 64^2 runs named on the command line:
attempted / seven-wave Newton / five-wave / planar seven-wave (late) /
four-unknown / lost.

    python3 cov_bins.py PROBLEM BLOCK TAG [TAG ...]"""
import glob, re, sys
import numpy as np
R = "/mnt/rafast/miler/codes/rotor2d/HLLD"
s, B, TAGS = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
CTRL = {"r2d": "r2d_64_exact_ctl1102", "ot": "ot_64_exact_ssretry", "rotor": "rotor_64_exact_ssretry"}
K = ("attempted", "seven", "planar5", "coplanar7", "planar4")


def sweeps(run):
    out = {}
    for f in sorted(glob.glob(R + "/results/%s/harvest/coverage_*.npz" % run)):
        z = np.load(f, allow_pickle=True)
        for j in range(len(z["sweep"])):
            n = int(z["n"][j])
            u = {k: np.unpackbits(z[k][j])[:n].astype(bool) for k in K + ("exact",) if k in z.files}
            row = [u[k].sum() if k in u else 0 for k in K] + [(u["attempted"] & ~u["exact"]).sum()]
            out[int(z["sweep"][j])] = row        # a restart rewrites sweeps: keep the last
    return np.array([out[k] for k in sorted(out)])


steps = 0
for f in sorted(glob.glob(R + "/logs/%s_2*.log" % CTRL[s])):
    for line in open(f, errors="ignore"):
        m = re.match(r"\s+step\s+(\d+) .*divB", line)
        if m:
            steps = max(steps, int(m.group(1)))
data = {t: sweeps("%s_64_exact_%s" % (s, t)) for t in TAGS}
ctl = sweeps(CTRL[s])
per = int(round(len(ctl) / max(steps, 1)))
print("%d sweeps per step; columns att/seven/p5/c7/p4/lost" % per)
n = max(len(d) for d in data.values()) // per
print("steps    " + "".join("%34s" % t for t in TAGS))
for a in range(0, n, B):
    row = "%3d-%3d  " % (a + 1, a + B)
    for t in TAGS:
        d = data[t][a * per:(a + B) * per]
        row += "%34s" % ("/".join(str(int(x)) for x in d.sum(axis=0)) if len(d) == B * per else "-")
    print(row)
