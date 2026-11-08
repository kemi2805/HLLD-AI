#!/usr/bin/env python3
"""Faces attempted, answered by each rung and lost, summed over the steps every
run of a problem has finished (from the coverage maps), for the control, the
rotation run and the planar-stage short runs.

    python3 cov_sum.py PROBLEM [NSTEPS]"""
import glob, re, sys
import numpy as np
R = "/mnt/rafast/miler/codes/rotor2d/HLLD"
s = sys.argv[1]
CTRL = {"r2d": "r2d_64_exact_ctl1102", "ot": "ot_64_exact_ssretry", "rotor": "rotor_64_exact_ssretry"}
RUNS = [("control", CTRL[s])] + [(n, "%s_64_exact_%s" % (s, t)) for n, t in
        (("rotation", "finalrotgate"), ("first", "finalrotc7_shortgate"),
         ("after+b", "finalrotc7ab_shortgate"), ("late+b", "finalrotc7lb_shortgate"))]
KEYS = ("attempted", "exact", "seven", "planar5", "coplanar7", "planar4")


def sweeps(run):
    out = []
    for f in sorted(glob.glob(R + "/results/%s/harvest/coverage_*.npz" % run)):
        z = np.load(f, allow_pickle=True)
        for j in range(len(z["sweep"])):
            n = int(z["n"][j])
            u = {k: (np.unpackbits(z[k][j])[:n].astype(bool) if k in z.files else np.zeros(n, bool)) for k in KEYS}
            out.append((int(z["sweep"][j]),) + tuple(int(u[k].sum()) for k in KEYS)
                       + (int((u["attempted"] & ~u["exact"]).sum()),))
    out.sort()
    return np.array(out)


steps = 0
for f in sorted(glob.glob(R + "/logs/%s_2*.log" % CTRL[s])):
    for line in open(f, errors="ignore"):
        m = re.match(r"\s+step\s+(\d+) .*divB", line)
        if m:
            steps = max(steps, int(m.group(1)))
data = {name: sweeps(run) for name, run in RUNS}
per = int(round(len(data["control"]) / max(steps, 1)))
have = {k: len(v) // per for k, v in data.items() if len(v)}
n = int(sys.argv[2]) if len(sys.argv) > 2 else min(have.values())
print("%s: steps 1-%d (%d sweeps per step); runs with fewer steps are left out" % (s, n, per))
print("%-9s %9s %9s %8s %8s %8s %6s %8s" % ("run", "attempted", "seven", "planar5", "planar7", "planar4", "lost", "lost %"))
for name, d in data.items():
    if have.get(name, 0) < n:
        continue
    a = d[:n * per, 1:].sum(axis=0)
    att, ex, sev, p5, c7, p4, lost = a
    print("%-9s %9d %9d %8d %8d %8d %6d %8.3f" % (name, att, sev, p5, c7, p4, lost, 100.0 * lost / max(att, 1)))
