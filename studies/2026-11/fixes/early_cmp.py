#!/usr/bin/env python3
"""Treatment vs control so far (login host, numpy only): wall per step window
from the run logs, and the HLLD (lost) fraction per step from diag.csv."""
import glob, re, sys
import numpy as np
R = "/mnt/rafast/miler/codes/rotor2d/HLLD"
TAG = sys.argv[1] if len(sys.argv) > 1 else "fix2"
PAIRS = {"r2d": "r2d_64_exact_ctl1102", "ot": "ot_64_exact_ssretry", "rotor": "rotor_64_exact_ssretry"}


def wall(logpat):
    fs = sorted(glob.glob(logpat))
    w = {}
    for f in fs:
        for line in open(f, errors="ignore"):
            m = re.match(r"\s+step\s+(\d+) .*\((\d+)s\)", line)
            if m:
                w[int(m.group(1))] = int(m.group(2))
    return w


def diag(d):
    rows = [l.strip().split(",") for l in open(d + "/diag.csv")]
    h = rows[0]
    a = np.array([[float(x) if x not in ("", "nan") else np.nan for x in r] for r in rows[1:]])
    return {k: a[:, i] for i, k in enumerate(h)}


for s, c in PAIRS.items():
    wt = wall(R + "/logs/%s_64_exact_%s_*.log" % (s, TAG))
    wc = wall(R + "/logs/%s_*.log" % c.replace("_64_exact_", "_64_exact_")) or wall(R + "/logs/*%s*.log" % c.split("_64_exact_")[1])
    dt, dc = diag(R + "/results/%s_64_exact_%s" % (s, TAG)), diag(R + "/results/" + c)
    n = int(min(dt["step"].max(), dc["step"].max()))
    print("== %s: %d steps in both" % (s, n))
    for a, b in ((1, 10), (10, 25), (25, 50), (50, 100), (100, 200), (200, 300), (300, n)):
        if b > n or a >= b or a not in wt or b not in wt:
            continue
        msg = "  steps %3d-%3d: treatment %5.0f s/step" % (a, b, (wt[b] - wt[a]) / (b - a))
        if a in wc and b in wc:
            msg += "  control %5.0f s/step  ratio %.2f" % ((wc[b] - wc[a]) / (b - a), (wt[b] - wt[a]) / max(wc[b] - wc[a], 1))
        print(msg)
    for k in ("hlle_frac_x", "hlle_frac_y"):
        if k in dt and k in dc:
            t_ = dt[k][:n]; c_ = dc[k][:n]
            print("  %s mean over steps 1-%d: control %.5f treatment %.5f (ratio %.2f)" % (k, n, np.nanmean(c_), np.nanmean(t_), np.nanmean(t_) / max(np.nanmean(c_), 1e-30)))
