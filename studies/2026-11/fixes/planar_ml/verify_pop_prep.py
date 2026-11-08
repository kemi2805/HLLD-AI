#!/usr/bin/env python3
"""Chunk files for verify_rmhd.py from the planar-network replay's population.

    python3 verify_pop_prep.py OUTDIR --population RUN [RUN ...]   production's answers
    python3 verify_pop_prep.py OUTDIR --dir LABEL PATH [LABEL PATH ...]   a replay's harvest

--population: the late-rung (source 2) and four-unknown (source 3) rows of
RUN's harvest in the sweeps replay_net.py replays (t/t_end >= 0.8), split per
sweep by the coverage counts exactly as replay_net.sweeps_of does -> label
<run>_prod.  --dir: every source-2/3 row of a harvest directory written by
replay_net.py --harvest (the replay's own answers).  Chunks are named
<label>_g43|g53_s<source>_<k>.npz so the caller can give verify_rmhd.py the
right gamma (SP_POP)."""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import planar_data as PD                                         # noqa: E402

CHUNK = 4000


def gtag(g):
    return "g43" if abs(g - 4.0 / 3.0) < 1e-9 else "g53"


def write(out, label, g, src, cat):
    n = len(cat["U_L"])
    for c, b in enumerate(range(0, n, CHUNK)):
        np.savez(os.path.join(out, "%s_%s_s%d_%03d.npz" % (label, gtag(g), src, c)), gamma=g,
                 **{k: v[b:b + CHUNK] for k, v in cat.items()})
    print("%-40s source %d: %d answers" % (label, src, n), flush=True)


def population(out, run, t_from=0.8):
    t_of, t_end = PD.run_times(run)
    H = os.path.join(PD.R, run, "harvest")
    rows = {2: [], 3: []}
    g = None
    for f in sorted(glob.glob(os.path.join(H, "coverage_r*_*.npz"))):
        tag = os.path.basename(f)[len("coverage_"):-len(".npz")]
        z = np.load(f)
        sw = z["sweep"]
        t = np.array([t_of.get(int(x) // PD.PER + 1, np.nan) for x in sw])
        if not (np.nan_to_num(t) / t_end >= t_from).any():
            continue
        cnt = lambda key: np.array([int(np.unpackbits(z[key][j])[:int(z["n"][j])].sum()) for j in range(len(sw))])
        n7, n4 = cnt("coplanar7"), cnt("planar4")
        s = np.load(os.path.join(H, "solved_" + tag + ".npz"))
        g = float(np.asarray(s["gamma"]).reshape(-1)[0])
        i2, i3 = np.flatnonzero(s["source"] == 2), np.flatnonzero(s["source"] == 3)
        o2, o3 = np.concatenate([[0], np.cumsum(n7)]), np.concatenate([[0], np.cumsum(n4)])
        keep = np.array([t[j] / t_end >= t_from for j in range(len(sw))])
        for src, ii, oo in ((2, i2, o2), (3, i3, o3)):
            sel = np.concatenate([ii[oo[j]:oo[j + 1]] for j in range(len(sw)) if keep[j]] or [np.zeros(0, int)])
            if sel.size:
                rows[src].append({k: s[k][sel] for k in ("U_L", "U_R", "zones", "speeds", "stage", "steer", "rot")})
    for src, parts in rows.items():
        if parts:
            write(out, run.split("_")[0] + "_prod", g, src, {k: np.concatenate([p[k] for p in parts]) for k in parts[0]})


def harvest_dir(out, label, path):
    rows = {2: [], 3: []}
    g = None
    for f in sorted(glob.glob(os.path.join(path, "solved_r*_*.npz"))):
        s = np.load(f)
        g = float(np.asarray(s["gamma"]).reshape(-1)[0])
        for src in (2, 3):
            k = np.flatnonzero(s["source"] == src)
            if k.size:
                rows[src].append({key: s[key][k] for key in ("U_L", "U_R", "zones", "speeds", "stage", "steer", "rot")})
    for src, parts in rows.items():
        if parts:
            write(out, label, g, src, {k: np.concatenate([p[k] for p in parts]) for k in parts[0]})


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    if sys.argv[2] == "--population":
        for run in sys.argv[3:]:
            population(out, run)
    elif sys.argv[2] == "--dir":
        a = sys.argv[3:]
        for label, path in zip(a[0::2], a[1::2]):
            harvest_dir(out, label, path)
    else:
        raise SystemExit(__doc__)
