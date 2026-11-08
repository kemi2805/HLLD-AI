#!/usr/bin/env python3
"""Aggregate verify_rmhd.py's output over the chunks of verify_rescues_prep.py.

    python3 verify_rescues_summary.py OUTDIR"""
import glob
import os
import sys
from collections import defaultdict

import numpy as np

NAMES = ("LF", "LA", "LS", "CD", "RS", "RA", "RF")
d = sys.argv[1]
groups = defaultdict(list)
for f in sorted(glob.glob(os.path.join(d, "*_s[23]_[0-9][0-9][0-9]_verify.npz"))):
    key = os.path.basename(f).rsplit("_", 2)[0]               # <run>_s<src>
    groups[key].append(f)
for key, fs in sorted(groups.items()):
    n = weak = adm = order = n_int = 0
    fail = np.zeros(7, int)
    inter = np.zeros(7, int)
    efall = np.zeros(7, int)
    stage_fail = defaultdict(lambda: [0, 0])
    for f in fs:
        r = np.load(f, allow_pickle=True)
        src = np.load(f.replace("_verify.npz", ".npz"))
        k = r["kind"]
        n += k.shape[0]
        weak += int(r["weak"].sum())
        adm += int(r["admissible"].sum())
        order += int((~r["order_ok"]).sum())
        fail += (k == "FAIL").sum(axis=0)
        inter += r["inter"].sum(axis=0)
        n_int += int(r["inter"][:, [2, 4]].any(axis=1).sum())
        efall += ((k == "disc") & (r["dS"] < -1e-9)).sum(axis=0)
        for st, w in zip(src["stage"], r["weak"]):
            stage_fail[int(st)][0] += 1
            stage_fail[int(st)][1] += int(not w)
    print("== %s: %d answers; weak solutions %d (%.2f%%), admissible %d (%.2f%%), out of order %d" % (
        key, n, weak, 100.0 * weak / max(1, n), adm, 100.0 * adm / max(1, n), order))
    print("   FAIL by wave %s" % dict(zip(NAMES, fail.tolist())))
    print("   intermediate shocks by wave %s; answers with an intermediate slow shock %d (%.1f%%)" % (
        dict(zip(NAMES, inter.tolist())), n_int, 100.0 * n_int / max(1, n)))
    print("   entropy-falling jumps by wave %s" % dict(zip(NAMES, efall.tolist())))
    print("   not weak by stage %s" % {s: "%d/%d" % (v[1], v[0]) for s, v in sorted(stage_fail.items())})
