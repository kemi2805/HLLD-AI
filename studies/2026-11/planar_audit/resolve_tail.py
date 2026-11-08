"""Accepted exact answers whose slow shock is no root (rh_tail_adm64/*_tail.npz,
extract_rh_tail.py; 2026-11-04).  Re-solve each such slow shock ALONE from its
stored ahead state and CD-side field with slow_shock_b's numpy ladder, filter
off and on, and classify it:
  REPRO   the ladder returns the stored speed (to 1e-8) although it is no
          root: the stored shock IS the ladder's output for these inputs
          (err False) -- the loose fallback, a grid point
  ROOT    the ladder finds a root (rh < 1e-8) that differs from the stored one
  FAIL    the ladder reports failure
Per class the re-solved root's entropy ratio and Lax margin.

    RMHD_SLOWSHOCK=numpy PYTHONPATH=<rmhd> python resolve_tail.py TAIL.npz LO HI [NMAX]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wavecheck as WC  # noqa: E402
from rmhd.batched import slow_shock_b as SSB  # noqa: E402

z = np.load(sys.argv[1])
lo, hi = float(sys.argv[2]), float(sys.argv[3])
nmax = int(sys.argv[4]) if len(sys.argv) > 4 else 200
g = float(z["gamma"])
rng = np.random.default_rng(0)
print("== %s: rows with a slow shock rh in (%.0e, %.0e]" % (os.path.basename(sys.argv[1]), lo, hi))
for w, ka, kb, col in (("LS", 2, 3, 2), ("RS", 5, 4, 4)):
    m = (z[w + "_solver_kind"] == 1) & (z[w + "_rh"] > lo) & (z[w + "_rh"] <= hi)
    i = np.flatnonzero(m)
    if i.size == 0:
        continue
    src_all = z["source"][i]
    if i.size > nmax:
        i = np.sort(rng.choice(i, nmax, replace=False))
    Z = z["zones"][i]
    Bn = np.ascontiguousarray(z["U_L"][i, 7])
    ahead = [np.ascontiguousarray(Z[:, ka, j]) for j in range(7)]
    behind = [Z[:, kb, j] for j in range(7)]
    Vst = z["speeds"][i, col]
    print("  %s: %d shocks (seven %d, planar5 %d), %d re-solved; stored K<1-1e-6 %d, stored lax<-1e-4 %d" % (
        w, m.sum(), (src_all == 0).sum(), (src_all == 1).sum(), i.size,
        (z[w + "_K"][i] < 1 - 1e-6).sum(), (z[w + "_lax_min"][i] < -1e-4).sum()))
    for flt in (False, True):
        SSB.SLOW_ADMISSIBLE = flt
        Vs, bh, err = SSB.make_solver(g)(np.ascontiguousarray(behind[5]), np.ascontiguousarray(behind[6]),
                                         ahead, w, Bn)
        rh, _ = WC.rh_cert(ahead, bh, Vs, Bn, g)
        K = (WC.pgas(bh, Bn) / bh[0] ** g) / (WC.pgas(ahead, Bn) / ahead[0] ** g)
        same = np.abs(Vs - Vst) < 1e-8
        repro = ~err & same
        root = ~err & ~same & (rh < 1e-8)
        fail = err
        other = ~(repro | root | fail)
        def q(x):
            return "-" if x.size == 0 else "p50 %.2e min %.2e" % (np.median(x), x.min())
        print("    filter %-5s REPRO %3d (rh %s) | ROOT %3d (1-K %s) | FAIL %3d | other %3d" % (
            flt, repro.sum(), q(rh[repro]), root.sum(), q(1 - K[root]), fail.sum(), other.sum()))
