"""The loose-fallback fix (2026-11-04): re-solve every LOOSE slow shock that a
filter run accepted (flagged_adm64/<run>_admiss.npz, r = max(|f_hug|,|f_416|)
/(1+|P_a|) > 1e-6 at the stored shock) with the CURRENT slow_shock_b, and
classify:
  REPRO  returns the stored speed again (to 1e-8): the leak is still open
  FAIL   err True
  ROOT   a converged root (r <= 1e-8), with its entropy ratio K
  OTHER  anything else (a different non-root answer)
for: filter off; filter on with the ladder only; filter on with the retry.

    PYTHONPATH=<rmhd> python fixcheck_loose.py FLAGGED_DIR [NMAX] [run ...]
"""
import glob
import os
import sys

import numpy as np

from rmhd.batched import slow_shock_b as SB

np.seterr(all="ignore")


def ent(s, Bn, g):
    rho, P, vx, vy, vz, By, Bz = s
    v2 = vx * vx + vy * vy + vz * vz
    eta = Bn * vx + By * vy + Bz * vz
    b2 = (Bn * Bn + By * By + Bz * Bz) * (1.0 - v2) + eta * eta
    return (P - 0.5 * b2) / rho ** g


def main():
    d = sys.argv[1]
    nmax = int(sys.argv[2]) if len(sys.argv) > 2 else 10 ** 9
    runs = sys.argv[3:] or [os.path.basename(f)[:-4] for f in sorted(glob.glob(os.path.join(d, "*_admiss.npz")))]
    rng = np.random.default_rng(0)
    for run in runs:
        z = np.load(os.path.join(d, run + ".npz"))
        g = float(z["gamma"])
        hug, e416 = SB.make_hug(g), SB.make_eq416(g)
        Zs, S, Bn_all = z["zones"], z["speeds"], z["U_L"][:, 7]
        print("== %s (%d flagged rows, gamma %.4f)" % (run, Zs.shape[0], g))
        for w, ka, kb, col in (("LS", 2, 3, 2), ("RS", 5, 4, 4)):
            i = np.flatnonzero(z[w + "_solver_kind"] == 1)
            if i.size == 0:
                continue
            ah = [np.ascontiguousarray(Zs[i, ka, j]) for j in range(7)]
            bh = [np.ascontiguousarray(Zs[i, kb, j]) for j in range(7)]
            Vst = np.ascontiguousarray(S[i, col])
            Bn = np.ascontiguousarray(Bn_all[i])
            f1, _ = hug(Vst, bh, ah, Bn)
            f2, _ = e416(Vst, bh, ah, Bn)
            r = np.maximum(np.abs(f1), np.abs(f2)) / (1.0 + np.abs(ah[1]))
            loose = ~(r <= 1e-6)
            k = np.flatnonzero(loose)
            if k.size > nmax:
                k = np.sort(rng.choice(k, nmax, replace=False))
            print("  %s: %d slow shocks, %d loose (r > 1e-6), %d re-solved" % (w, i.size, loose.sum(), k.size))
            if k.size == 0:
                continue
            a = [c[k] for c in ah]
            Byt, Bzt, B, Vs0 = bh[5][k], bh[6][k], Bn[k], Vst[k]
            for name, flt, rt in (("off", False, True), ("on, ladder only", True, False), ("on, with retry", True, True)):
                SB.SLOW_ADMISSIBLE, SB.SLOW_RETRY = flt, rt
                V, b, e = SB.make_solver(g)(Byt, Bzt, a, w, B)
                g1, _ = hug(V, b, a, B)
                g2, _ = e416(V, b, a, B)
                rr = np.maximum(np.abs(g1), np.abs(g2)) / (1.0 + np.abs(a[1]))
                repro = ~e & (np.abs(V - Vs0) < 1e-8)
                root = ~e & ~repro & (rr <= 1e-8)
                fail = e
                other = ~(repro | root | fail)
                K = ent(b, B, g) / ent(a, B, g)
                print("    filter %-16s REPRO %4d | FAIL %4d | ROOT %4d (1-K p50 %s, entropy falls > 1e-6: %d) | OTHER %d" % (
                    name, repro.sum(), fail.sum(), root.sum(),
                    "%.1e" % np.median(1 - K[root]) if root.any() else "-",
                    int((root & (K < 1 - 1e-6)).sum()), other.sum()))
    SB.SLOW_ADMISSIBLE, SB.SLOW_RETRY = False, True


if __name__ == "__main__":
    main()
