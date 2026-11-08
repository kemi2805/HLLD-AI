#!/usr/bin/env python
"""A uniform sample of a run's harvested exact answers (seven-wave and planar
rows: U_L, U_R, zones, speeds, source) for verify_rmhd.py, written in chunks.

    python sample_run_harvest.py HARVEST_DIR OUTDIR NTOTAL NCHUNKS
"""
import glob
import os
import sys
import numpy as np
hd, out, ntot, nch = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
files = sorted(glob.glob(os.path.join(hd, "solved_*.npz")))
sizes = np.array([np.load(f, mmap_mode="r")["U_L"].shape[0] if False else len(np.load(f)["source"]) for f in files])
tot = sizes.sum()
rng = np.random.default_rng(0)
take = np.sort(rng.choice(tot, size=min(ntot, tot), replace=False))
starts = np.concatenate([[0], np.cumsum(sizes)])
parts = {k: [] for k in ("U_L", "U_R", "zones", "speeds", "source")}
gamma = None
for fi, f in enumerate(files):
    m = (take >= starts[fi]) & (take < starts[fi + 1])
    if not m.any():
        continue
    z = np.load(f)
    gamma = float(z["gamma"])
    rows = take[m] - starts[fi]
    for k in parts:
        parts[k].append(z[k][rows])
allr = {k: np.concatenate(v) for k, v in parts.items()}
os.makedirs(out, exist_ok=True)
n = len(allr["source"])
for c, ix in enumerate(np.array_split(np.arange(n), nch)):
    np.savez(os.path.join(out, "chunk_%03d.npz" % c), gamma=gamma, **{k: v[ix] for k, v in allr.items()})
print("%d solved rows in %d shards; sampled %d (source counts %s) into %d chunks" % (
    tot, len(files), n, np.bincount(allr["source"]).tolist(), nch))
