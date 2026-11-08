#!/usr/bin/env python3
"""Where the extra wall of the loss-fix runs goes: per step, from the coverage
maps, the faces attempted, answered by each rung, and lost (attempted, not
exact) -- every lost face has run the whole rescue ladder -- against the wall
per step from the run log, for a run and its control.

    python3 cov_cost.py PROBLEM TAG [TAG ...]       (e.g. ot fixfinal2 finalgate)
"""
import glob, re, sys
import numpy as np
R = "/mnt/rafast/miler/codes/rotor2d/HLLD"
CTRL = {"r2d": "r2d_64_exact_ctl1102", "ot": "ot_64_exact_ssretry", "rotor": "rotor_64_exact_ssretry"}


def wall(pat):
    w = {}
    for f in sorted(glob.glob(pat)):
        for line in open(f, errors="ignore"):
            m = re.match(r"\s+step\s+(\d+) .*\((\d+)s\)", line)
            if m:
                w[int(m.group(1))] = int(m.group(2))
    return w


def coverage(run):
    recs = []
    for f in sorted(glob.glob(R + "/results/%s/harvest/coverage_*.npz" % run)):
        z = np.load(f, allow_pickle=True)
        for k in sorted(z.files):
            pass
        recs.append(z)
    return recs


def sweeps(run):
    """per sweep: (sweep, attempted, exact, planar4, seven, planar5, lost)"""
    out = []
    for f in sorted(glob.glob(R + "/results/%s/harvest/coverage_*.npz" % run)):
        z = np.load(f, allow_pickle=True)
        for j in range(len(z["sweep"])):
            n = int(z["n"][j])
            u = lambda k: np.unpackbits(z[k][j])[:n].astype(bool)
            att, ex = u("attempted"), u("exact")
            out.append((int(z["sweep"][j]), att.sum(), ex.sum(), u("planar4").sum(), u("seven").sum(),
                        u("planar5").sum(), (att & ~ex).sum()))
    out.sort()
    return np.array(out)


prob = sys.argv[1]
runs = {"control": CTRL[prob]}
for t in sys.argv[2:]:
    runs[t] = "%s_64_exact_%s" % (prob, t)
data = {}
for name, run in runs.items():
    s = sweeps(run)
    logs = sorted(glob.glob(R + "/logs/%s_2*.log" % run)) if name == "control" else sorted(glob.glob(R + "/logs/%s_2*.log" % run))
    w = wall(R + "/logs/%s_2*.log" % run)
    data[name] = (s, w)
    print("%-10s sweeps %d  steps %d" % (name, len(s), max(w) if w else 0))
# sweeps per step from the control
s0, w0 = data["control"]
per = max(1, int(round(len(s0) / max(w0))))
print("sweeps per step ~", per)
n = min(max(w) for _, w in data.values())
print("%-10s %6s %8s %8s %8s %8s %9s" % ("run", "steps", "attempt", "seven", "planar4", "lost", "wall s"))
for lo, hi in ((1, 25), (25, 50), (50, 100), (100, 200), (200, n)):
    if hi > n or lo >= hi:
        continue
    for name, (s, w) in data.items():
        rows = s[(lo - 1) * per: hi * per]
        if not len(rows) or lo not in w or hi not in w:
            continue
        print("%-10s %3d-%3d %8d %8d %8d %8d %9d" % (name, lo, hi, rows[:, 1].sum(), rows[:, 4].sum(), rows[:, 3].sum(),
                                                rows[:, 6].sum(), w[hi] - w[lo]))
