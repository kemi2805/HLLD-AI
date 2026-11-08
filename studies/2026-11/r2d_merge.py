#!/usr/bin/env python
"""Merge trainer-schema datasets, each repeated k times (2026-11-02, retrain loop).

    python r2d_merge.py OUT.npz IN.npz[:k] [IN.npz[:k] ...]

Keeps U_L, U_R, zones, speeds; refuses mixed gammas; shuffles the rows.
An input that does not exist (a harvest that solved nothing) is skipped.
"""
import os
import sys

import numpy as np

KEYS = ("U_L", "U_R", "zones", "speeds")
out, parts = sys.argv[1], sys.argv[2:]
cols, gamma = {k: [] for k in KEYS}, None
for p in parts:
    path, k = (p.rsplit(":", 1) + ["1"])[:2] if ":" in p else (p, "1")
    if not os.path.exists(path):          # a harvest that solved nothing writes no file
        print("  %-70s absent (nothing solved), skipped" % path)
        continue
    z = np.load(path)
    g = float(z["gamma"])
    if gamma is not None and abs(g - gamma) > 1e-12:
        sys.exit("%s: gamma %g != %g" % (path, g, gamma))
    gamma = g
    for key in KEYS:
        cols[key].extend([z[key]] * int(k))
    print("  %-70s %8d rows x %s" % (path, z["U_L"].shape[0], k))
merged = {k: np.concatenate(v) for k, v in cols.items()}
perm = np.random.default_rng(3).permutation(merged["U_L"].shape[0])
np.savez_compressed(out, gamma=gamma, **{k: v[perm] for k, v in merged.items()})
print("wrote %s: %d rows" % (out, perm.size))
