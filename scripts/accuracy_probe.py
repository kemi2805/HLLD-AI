"""How exact are the exact fluxes, and in which convergence regime?

The x-faces of a snapshot (cell-average pairs, B_n averaged) solved twice
through production's path: at production's tolerance (residual 1e-8, 40
iterations) and at 1e-12 (60 iterations).  The tight answer stands in for
the true root.  Per face, by the solver that answered at 1e-8:

  * the flux error of the production answer, |F_8 - F_12| / |F_12|;
  * the extra iterations 1e-12 cost -- ~1 in Newton's quadratic regime
    (each step squares the error), many in a linear one (a near-singular
    Jacobian, a stall);
  * faces exact at 1e-8 that 1e-12 cannot reach (the precision floor), and
    faces whose tight answer is another root (flux moved > 1e-4);
  * iterations and retries at production's tolerance.

    python scripts/accuracy_probe.py SNAP GAMMA

with production's environment (kernels.env, the ML checkpoints, RMHD_POOL);
calea job 41731 ran it on results/ot_64_exact_pool/snap_00{2,4,8}.npz (4/3)
and results/r2d_64_exact_pool/snap_004.npz (5/3).
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.physics.eos import hybrid_eos                     # noqa: E402
from src.physics import exact_flux as EF                   # noqa: E402
from src.physics.hlld import LAST_DIAG                     # noqa: E402


def faces(snap, eos):
    z = np.load(snap)
    t = lambda a: torch.tensor(np.ascontiguousarray(a).reshape(-1), dtype=torch.float64)
    f = {k: z[k] for k in ("rho", "p", "vx", "vy", "Bx", "By")}
    f["vz"] = np.zeros_like(f["rho"]); f["Bz"] = np.zeros_like(f["rho"])
    L = dict(f)
    R = {k: np.roll(v, -1, axis=0) for k, v in f.items()}
    L["Bx"] = R["Bx"] = 0.5 * (L["Bx"] + R["Bx"])
    st = lambda a: dict({k: t(v) for k, v in a.items()},
                        eps=eos.eps__press_rho(t(a["p"]), t(a["rho"])))
    return float(z["t"]), st(L), st(R)


def solve(sL, sR, eos, acc, mi):
    F, _, _ = EF.exact_flux_batched(sL, sR, eos, idir=0, tau_weak=1e-2, tau_bt=1e-9,
                                    n_retries=2, max_iter=mi, accuracy=acc)
    d = dict(LAST_DIAG)
    sel = np.asarray(d["sel"])
    n = d["exact_mask"].numel()
    path = np.full(n, -1, np.int8)                  # -1 not attempted
    ex = d["exact_mask"].numpy()
    path[sel] = 0                                   # HLLD
    a = lambda k: np.asarray(d.get(k, np.zeros(sel.size, bool)), bool)
    seven = np.asarray(d["reason7"]) == 0
    path[sel[seven]] = 1
    path[sel[a("planar5_mask")]] = 2
    path[sel[a("planar4_mask")]] = 3
    path[sel[a("degenerate_mask")]] = 4
    path[(path > 0) & ~ex] = 0
    it7 = np.zeros(n, int); it7[sel] = np.asarray(d["seven_n_iter_total"])
    itp = np.zeros(n, int); itp[sel] = np.asarray(d["planar_n_iter"])
    att = np.zeros(n, int); att[sel] = np.asarray(d["seven_attempts"])
    return F, ex, path, it7, itp, att


def q(x):
    return "p50 %.1e p90 %.1e max %.1e" % (np.median(x), np.percentile(x, 90), x.max()) if x.size else "-"


def main():
    snap, gam = sys.argv[1], float(sys.argv[2])
    eos = hybrid_eos(K=0.0, gamma=gam, gamma_th=gam)
    tt, sL, sR = faces(snap, eos)
    F8, e8, p8, i8, ip8, a8 = solve(sL, sR, eos, 1e-8, 40)
    F12, e12, p12, i12, ip12, a12 = solve(sL, sR, eos, 1e-12, 60)
    rel = (sum((F8[k] - F12[k]).abs() for k in F8) / sum(F12[k].abs() for k in F12)).numpy()
    print("== %s t=%.3f: %d faces, attempted %d, exact at 1e-8 %d, at 1e-12 %d"
          % (snap.split("/")[-2], tt, p8.size, (p8 >= 0).sum(), e8.sum(), e12.sum()))
    names = {1: "seven-wave", 2: "planar", 3: "four-unknown", 4: "degenerate"}
    for k, name in names.items():
        m = p8 == k
        if not m.any():
            continue
        both = m & e12
        it = i8 if k == 1 else ip8
        it_t = i12 if k == 1 else ip12
        d_it = (it_t - it)[both]
        print("  %-12s %6d | iterations at 1e-8 p50 %d p90 %d max %d | extra for 1e-12: "
              "+0/1 %4.1f%%  +2 %4.1f%%  +3..5 %4.1f%%  >5 %4.1f%%"
              % (name, m.sum(), np.median(it[m]), np.percentile(it[m], 90), it[m].max(),
                 100 * np.mean(d_it <= 1), 100 * np.mean(d_it == 2),
                 100 * np.mean((d_it >= 3) & (d_it <= 5)), 100 * np.mean(d_it > 5)))
        print("  %-12s        | flux error of the 1e-8 answer: %s | > 1e-6 on %d, > 1e-4 (another root) on %d | lost at 1e-12: %d"
              % ("", q(rel[both]), (rel[both] > 1e-6).sum(), (rel[both] > 1e-4).sum(), (m & ~e12).sum()))
        if k == 1:
            print("  %-12s        | seven-wave attempts (retries+1): 1 %.1f%%  2 %.1f%%  3 %.1f%%"
                  % ("", 100 * np.mean(a8[m] == 1), 100 * np.mean(a8[m] == 2), 100 * np.mean(a8[m] >= 3)))
    m = (p8 == 0) & e12
    print("  HLLD at 1e-8 %d, of them exact at 1e-12: %d" % ((p8 == 0).sum(), m.sum()), flush=True)


if __name__ == "__main__":     # the pool spawns workers that re-import this
    main()
