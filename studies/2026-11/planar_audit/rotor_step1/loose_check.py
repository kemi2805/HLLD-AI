"""The filter run's flagged rows (first rotor shard): re-solve each flagged
slow shock alone from its stored ahead state and CD-side field, filter off
and on (numpy ladder), and say whether the answer is a root (RH certificate)
and whether it is the stored one.  Tests the loose-fallback hypothesis
(2026-11-04).

    RMHD_SLOWSHOCK=numpy PYTHONPATH=<rmhd worktree> python loose_check.py FLAGGED.npz
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wavecheck as WC  # noqa: E402
from rmhd.batched import slow_shock_b as SSB  # noqa: E402

z = np.load(sys.argv[1])
g = float(z["gamma"])
Bn = z["U_L"][:, 7]
Z = z["zones"]
for w, ka, kb, col in (("LS", 2, 3, 2), ("RS", 5, 4, 4)):
    i = np.flatnonzero((z[w + "_solver_kind"] == 1) & ((z[w + "_K"] < 1 - 1e-6) | (z[w + "_lax_min"] < -1e-4)))
    if i.size == 0:
        continue
    ahead = [np.ascontiguousarray(Z[i, ka, j]) for j in range(7)]
    behind = [Z[i, kb, j] for j in range(7)]
    print("== %s: %d flagged rows; stored Vs %s" % (w, i.size, np.round(z["speeds"][i, col], 6)))
    print("   stored: K %s rh %s" % (np.round(z[w + "_K"][i], 5), ["%.1e" % x for x in z[w + "_rh"][i]]))
    for flt in (False, True):
        SSB.SLOW_ADMISSIBLE = flt
        solve = SSB.make_solver(g)
        Vs, bh, err = solve(np.asarray(behind[5]), np.asarray(behind[6]), ahead, w, Bn[i])
        rh, _ = WC.rh_cert(ahead, bh, Vs, Bn[i], g)
        Kb = WC.pgas(bh, Bn[i]) / bh[0] ** g
        Ka = WC.pgas(ahead, Bn[i]) / ahead[0] ** g
        same = np.abs(Vs - z["speeds"][i, col]) < 1e-8
        print("   filter %-5s err %s Vs %s rh %s K %s same-as-stored %s" % (
            flt, err.astype(int).tolist(), np.round(Vs, 6), ["%.1e" % x for x in rh],
            np.round(Kb / Ka, 5), same.astype(int).tolist()))
