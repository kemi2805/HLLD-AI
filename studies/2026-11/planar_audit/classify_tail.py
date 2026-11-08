"""Accepted exact answers whose slow shock is no root, split by the sub-solver's
OWN residual (2026-11-04).  slow_shock_b's inner Newton converges when
|f_hug| and |f_416| < 1e-10 (1 + |P_ahead|); the ladder's loose fallback is a
grid point with |f_hug| + |f_416| < 0.1 (1 + |P_ahead|).  So at the stored
(Vs, behind) of every slow shock in rh_tail_adm64/*_tail.npz (RH certificate
> 1e-5): r = max(|f_hug|, |f_416|) / (1 + |P_ahead|), and
  root   r <= 1e-8   a converged root whose RH certificate is still > 1e-5
                     (eq. 4.17, closed by the OUTER Newton, carries the rest)
  loose  r >  1e-6   the loose fallback, no root
Counts per run, source and certificate band.

    PYTHONPATH=<rmhd> python classify_tail.py rh_tail_adm64
"""
import glob
import os
import sys

import numpy as np

from rmhd.batched import slow_shock_b as SSB

BANDS = ((1e-5, 1e-4), (1e-4, 1e-3), (1e-3, np.inf))


def main():
    d = sys.argv[1]
    for f in sorted(glob.glob(os.path.join(d, "*_tail.npz"))):
        z = np.load(f)
        g = float(z["gamma"])
        hug, e416, e417 = SSB.make_hug(g), SSB.make_eq416(g), SSB.make_eq417(g)
        Zs, S, Bn, src = z["zones"], z["speeds"], z["U_L"][:, 7], z["source"]
        rows_loose = np.zeros(Zs.shape[0], bool)
        tab = {}
        for w, ka, kb, col in (("LS", 2, 3, 2), ("RS", 5, 4, 4)):
            m = (z[w + "_solver_kind"] == 1) & (z[w + "_rh"] > 1e-5)
            i = np.flatnonzero(m)
            if i.size == 0:
                continue
            ah = [np.ascontiguousarray(Zs[i, ka, j]) for j in range(7)]
            bh = [np.ascontiguousarray(Zs[i, kb, j]) for j in range(7)]
            Vs = np.ascontiguousarray(S[i, col])
            with np.errstate(all="ignore"):
                f1, _ = hug(Vs, bh, ah, Bn[i])
                f2, _ = e416(Vs, bh, ah, Bn[i])
                f3, _ = e417(Vs, bh, ah, Bn[i])
            sc = 1.0 + np.abs(ah[1])
            r = np.maximum(np.abs(f1), np.abs(f2)) / sc
            r = np.where(np.isfinite(r), r, np.inf)
            rows_loose[i[r > 1e-6]] = True
            for s in (0, 1):
                for lo, hi in BANDS:
                    k = (src[i] == s) & (z[w + "_rh"][i] > lo) & (z[w + "_rh"][i] <= hi)
                    key = (s, lo)
                    a = tab.setdefault(key, [0, 0, 0, 0, []])
                    a[0] += int(k.sum())
                    a[1] += int((k & (r <= 1e-8)).sum())
                    a[2] += int((k & (r > 1e-6)).sum())
                    a[3] += int((k & (r > 1e-8) & (r <= 1e-6)).sum())
                    a[4].append(np.abs(f3[k]) / sc[k])
        name = os.path.basename(f).replace("_tail.npz", "")
        print("== %s: rows with a loose slow shock: seven %d, planar5 %d" % (
            name, int((rows_loose & (src == 0)).sum()), int((rows_loose & (src == 1)).sum())))
        for (s, lo), (n, nroot, nloose, nmid, e) in sorted(tab.items()):
            if n == 0:
                continue
            e = np.concatenate(e)
            print("   %-7s rh in (%.0e, %s]: %6d shocks  root %6d  loose %5d  between %4d   |eq417|/scale p50 %.1e" % (
                ("seven", "planar5")[s], lo, "1e%d" % np.log10(lo * 10) if np.isfinite(lo * 10) and lo < 1e-3 else "inf",
                n, nroot, nloose, nmid, np.median(e)))


if __name__ == "__main__":
    main()
