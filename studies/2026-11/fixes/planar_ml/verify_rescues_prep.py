#!/usr/bin/env python3
"""Chunk files for verify_rmhd.py: every late-rung (source 2) and
four-unknown (source 3) answer the harvest runs recorded.

    python3 verify_rescues_prep.py OUTDIR RUN [RUN ...] [--chunk 4000]

-> OUTDIR/<run>_s<source>_<k>.npz (U_L, U_R, zones, speeds, gamma, shard,
row, stage, steer, rot)."""
import glob
import os
import sys

import numpy as np

R = os.environ.get("PLANAR_RESULTS", "/mnt/rafast/miler/codes/rotor2d/HLLD/results")
args = sys.argv[1:]
chunk = 4000
if "--chunk" in args:
    i = args.index("--chunk")
    chunk = int(args[i + 1])
    del args[i:i + 2]
out, runs = args[0], args[1:]
os.makedirs(out, exist_ok=True)
for run in runs:
    rows = {2: [], 3: []}
    gamma = None
    for f in sorted(glob.glob(os.path.join(R, run, "harvest", "solved_r*_*.npz"))):
        s = np.load(f)
        gamma = float(np.asarray(s["gamma"]).reshape(-1)[0])
        shard = int(os.path.basename(f)[-9:-4])
        for src in (2, 3):
            k = np.flatnonzero(s["source"] == src)
            if k.size:
                rows[src].append({key: s[key][k] for key in ("U_L", "U_R", "zones", "speeds", "stage", "steer", "rot")}
                                 | {"shard": np.full(k.size, shard, np.int32), "row": k.astype(np.int32)})
    for src, parts in rows.items():
        if not parts:
            continue
        cat = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
        n = len(cat["U_L"])
        for c, b in enumerate(range(0, n, chunk)):
            np.savez(os.path.join(out, "%s_s%d_%03d.npz" % (run, src, c)), gamma=gamma,
                     **{key: v[b:b + chunk] for key, v in cat.items()})
        print("%-34s source %d: %d answers in %d chunks" % (run, src, n, -(-n // chunk)), flush=True)
