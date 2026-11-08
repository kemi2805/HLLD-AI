#!/usr/bin/env python
"""Rows of a trainer-schema dataset whose face is in none of the others
(2026-11-02, retrain loop: the ledger faces a new network wins that the
training set does not hold yet).

    python r2d_newly_won.py OUT.npz NEW.npz OLD.npz [OLD.npz ...]

A face is its (U_L, U_R) bytes; zones and speeds may differ between two solves
of the same face at the Newton tolerance, so they are not compared.  Keeps
gamma, U_L, U_R, zones, speeds (r2d_merge.py's schema).
"""
import os
import sys

import numpy as np

KEYS = ("U_L", "U_R", "zones", "speeds")
out, new, olds = sys.argv[1], sys.argv[2], sys.argv[3:]
key = lambda z: [a.tobytes() + b.tobytes() for a, b in zip(z["U_L"], z["U_R"])]
seen = set()
for p in olds:
    if os.path.exists(p):                 # a harvest that solved nothing writes no file
        seen |= set(key(np.load(p)))
    else:
        print("  %s absent (nothing solved): no faces from it" % p)
if not os.path.exists(new):
    sys.exit("%s absent: nothing solved, so nothing newly won" % new)
z = np.load(new)
m = np.array([k not in seen for k in key(z)], bool)
np.savez_compressed(out, gamma=float(z["gamma"]), **{k: z[k][m] for k in KEYS})
print("%s: %d rows, %d of them on faces in none of %s -> %s" % (new, m.size, m.sum(), ", ".join(olds), out))
