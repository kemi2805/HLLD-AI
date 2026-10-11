"""Solve rate per decade of the weak-gate jump measure, from an exact run's
harvest -- for a run with the gate opened (``TAU_WEAK=1e-12``), where every
face with a real jump was attempted.

A solved face is a row of ``solved_*``; a LOST face is a row of
``unsolved_*`` with no planar-solved twin (``source == 1``) in its shard or
the next one (harvests written before 2026-10-11 could flush between a
call's seven-wave and planar records; see ``src/physics/harvest.py``).  The
four-unknown and degenerate rungs' answers are not harvested as solved rows,
so a four-unknown answer counts as lost here: the rates are lower bounds by
the four-unknown share.  The jump is exact_flux's own ``relative_jump`` on
lab-frame states rebuilt with x as the normal (it is rotation-invariant).

Only flushes before the last one that wrote solved rows are read: past the
solved cap (5M rows with ``--harvest-all``) an unsolved row has no twin to
look up, and that last flush may have been cut by the cap.  ``--max-sweep``
keeps the flushes whose sweeps all lie at or below it, so two runs compare
over the same stretch of the evolution (read the range off one run's output
and pass it to the other).

    python scripts/harvest_by_jump.py results/ot_64_exact_open/harvest
    python scripts/harvest_by_jump.py results/ot_64_exact/harvest --max-sweep 1808
"""
import argparse
import bisect
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.physics.eos import hybrid_eos                             # noqa: E402
from src.physics.exact_flux import relative_jump                   # noqa: E402

EDGES = np.array([0.0, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, np.inf])
NAMES = ["< 1e-8", "1e-8 .. 1e-6", "1e-6 .. 1e-4", "1e-4 .. 1e-3",
         "1e-3 .. 1e-2", ">= 1e-2 (the gate)"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("harvest", help="a run's harvest directory")
    ap.add_argument("--max-sweep", type=int, default=None)
    a = ap.parse_args()
    H = a.harvest

    shard = lambda f: os.path.basename(f).split("_", 1)[1]   # rNNN_SSSSS.npz
    solved_files = sorted(glob.glob(os.path.join(H, "solved_*.npz")))
    if not solved_files:
        raise SystemExit("no solved_*.npz in %s" % H)
    gam = float(np.load(solved_files[0])["gamma"])
    eos = hybrid_eos(K=0.0, gamma=gam, gamma_th=gam)
    last = shard(solved_files[-1])
    cov = {shard(f): np.load(f)["sweep"]
           for f in glob.glob(os.path.join(H, "coverage_*.npz"))}
    keep = {s for s, sw in cov.items()
            if s < last and (a.max_sweep is None or sw.max() <= a.max_sweep)}
    files = [f for f in sorted(glob.glob(os.path.join(H, "unsolved_*.npz")))
             if shard(f) in keep]
    if not files:
        raise SystemExit("no paired flush in range")
    sw = np.concatenate([cov[shard(f)] for f in files])
    print("gamma %.6g; %d flushes, sweeps %d .. %d"
          % (gam, len(files), sw.min(), sw.max()), flush=True)

    key = lambda A, B: [x.tobytes() + y.tobytes() for x, y in zip(A, B)]

    def planar(fs):
        if not fs or not os.path.exists(fs):
            return set()
        z = np.load(fs)
        p = z["source"] == 1
        return set(key(z["U_L"][p], z["U_R"][p]))

    def after(fs):
        j = bisect.bisect_right(solved_files, fs)
        return solved_files[j] if j < len(solved_files) else ""

    def lab(U):
        rho, Pt, vx, vy, vz, By, Bz, Bx = (U[:, j] for j in range(8))
        W2 = 1.0 / (1.0 - (vx * vx + vy * vy + vz * vz))
        eta = Bx * vx + By * vy + Bz * vz
        b2 = (Bx * Bx + By * By + Bz * Bz) / W2 + eta * eta
        t = lambda v: torch.tensor(np.ascontiguousarray(v), dtype=torch.float64)
        d = {"rho": t(rho), "vx": t(vx), "vy": t(vy), "vz": t(vz),
             "p": t(Pt - 0.5 * b2), "Bx": t(Bx), "By": t(By), "Bz": t(Bz)}
        d["eps"] = eos.eps__press_rho(d["p"], d["rho"])
        return d

    def jumps(UL, UR):
        if len(UL) == 0:
            return np.zeros(0)
        return relative_jump(lab(UL), lab(UR)).numpy().reshape(-1)

    solved = np.zeros(len(NAMES), np.int64)
    lost = np.zeros(len(NAMES), np.int64)
    for k, fu in enumerate(files):
        u = np.load(fu)
        fs = fu.replace("unsolved_", "solved_")
        s = (np.load(fs) if os.path.exists(fs)
             else dict(U_L=np.zeros((0, 8)), U_R=np.zeros((0, 8))))
        resc = planar(fs) | planar(after(fs))
        m = np.array([kk not in resc for kk in key(u["U_L"], u["U_R"])], bool)
        solved += np.histogram(jumps(s["U_L"], s["U_R"]), EDGES)[0]
        lost += np.histogram(jumps(u["U_L"][m], u["U_R"][m]), EDGES)[0]
        if k % 25 == 0:
            print("  flush %d / %d" % (k + 1, len(files)), flush=True)

    print("\n%-22s %12s %12s %10s" % ("relative jump", "solved", "lost", "solved %"))
    for name, s_, l_ in zip(NAMES, solved, lost):
        print("%-22s %12d %12d %9.2f%%" % (name, s_, l_, 100.0 * s_ / max(s_ + l_, 1)))
    A, B = solved.sum(), lost.sum()
    print("%-22s %12d %12d %9.2f%%" % ("all attempted", A, B, 100.0 * A / max(A + B, 1)))
    w = slice(0, 5)
    print("%-22s %12d %12d %9.2f%%" % ("below the gate", solved[w].sum(), lost[w].sum(),
          100.0 * solved[w].sum() / max(solved[w].sum() + lost[w].sum(), 1)))


if __name__ == "__main__":
    main()
