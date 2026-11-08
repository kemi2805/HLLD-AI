#!/usr/bin/env python3
"""Per step of the 64^2 runs (control, rotation run, planar-stage short
runs): faces attempted and answered by each rung (from the coverage maps),
summed over the step's sweeps: seven-wave Newton (with the planar stage's
answers when it runs inside solve_batch), five-wave planar, the planar
seven-wave rung (RMHD_COPLANAR7=late), four-unknown planar, lost.

    python3 cov_steps.py PROBLEM [NSTEPS] [FIRST]"""
import glob, sys
import numpy as np
R = "/mnt/rafast/miler/codes/rotor2d/HLLD"
s = sys.argv[1] if len(sys.argv) > 1 else "r2d"
nmax = int(sys.argv[2]) if len(sys.argv) > 2 else 18
first = int(sys.argv[3]) if len(sys.argv) > 3 else 1
CTRL = {"r2d": "r2d_64_exact_ctl1102", "ot": "ot_64_exact_ssretry", "rotor": "rotor_64_exact_ssretry"}
RUNS = [("control", CTRL[s])] + [(n, "%s_64_exact_%s" % (s, t)) for n, t in
        (("rot", "finalrotgate"), ("first", "finalrotc7_shortgate"),
         ("after", "finalrotc7a_shortgate"), ("after+b", "finalrotc7ab_shortgate"),
         ("late+b", "finalrotc7lb_shortgate"))]


def sweeps(run):
    out = []
    for f in sorted(glob.glob(R + "/results/%s/harvest/coverage_*.npz" % run)):
        z = np.load(f, allow_pickle=True)
        keys = [k for k in ("attempted", "exact", "seven", "planar5", "planar4", "coplanar7") if k in z.files]
        for j in range(len(z["sweep"])):
            n = int(z["n"][j])
            u = {k: np.unpackbits(z[k][j])[:n].astype(bool) for k in keys}
            c7 = u["coplanar7"].sum() if "coplanar7" in u else 0
            out.append((int(z["sweep"][j]), u["attempted"].sum(), u["seven"].sum(), u["planar5"].sum(),
                        c7, u["planar4"].sum(), (u["attempted"] & ~u["exact"]).sum()))
    out.sort()
    return np.array(out)


z = np.load(sorted(glob.glob(R + "/results/%s/harvest/coverage_*.npz" % CTRL[s]))[0], allow_pickle=True)
print("coverage keys:", z.files)
data = {name: sweeps(run) for name, run in RUNS}
for name, d in data.items():
    print("%-8s %d sweeps" % (name, len(d)))
# sweeps per step (RK stages x 2 directions), from the control's log
import re
steps = 0
for f in sorted(glob.glob(R + "/logs/%s_2*.log" % CTRL[s])):
    for line in open(f, errors="ignore"):
        m = re.match(r"\s+step\s+(\d+) .*divB", line)
        if m:
            steps = max(steps, int(m.group(1)))
per = int(round(len(data["control"]) / max(steps, 1)))
print("%d sweeps per step (control: %d sweeps, %d steps)" % (per, len(data["control"]), steps))
print("step  " + "".join("%28s" % n for n, _ in RUNS))
print("      " + "".join("%28s" % "att/seven/p5/c7/p4/lost" for _ in RUNS))
for i in range(first - 1, nmax):
    row = "%4d  " % (i + 1)
    for name, _ in RUNS:
        d = data[name]
        rows = d[i * per:(i + 1) * per]
        if len(rows) < per:
            row += "%28s" % "-"
            continue
        a = rows[:, 1:].sum(axis=0)
        row += "%28s" % ("%d/%d/%d/%d/%d/%d" % tuple(a))
    print(row)
