#!/usr/bin/env python
"""Paired comparison per time block of an r2d_eval_seeds.py result
(2026-11-03, the proper retrains of the rotor and the Orszag-Tang).

    python blocks_paired.py EVAL_RESULT.npz EVAL.npz T_END A:B [A:B ...]

EVAL.npz is the held-out file the result was drawn from (val.npz or
test.npz of split_proper.py, which keep each problem's time t).  The faces
are grouped by the nearest block centre (split_proper.py's TEST 0.4375 /
0.8125 and VAL 0.625 of t_end); for every pair: converged by A only / by B
only and the exact McNemar p-value, natural draw and complex faces.
"""
import sys

import numpy as np

from r2d_paired import mcnemar

CENTRES = (0.4375, 0.625, 0.8125)


def main():
    res, ev = np.load(sys.argv[1]), np.load(sys.argv[2])
    tend = float(sys.argv[3])
    pairs = [s.split(":") for s in sys.argv[4:]]
    t = ev["t"][res["idx"]] / tend
    blk = np.argmin(np.abs(t[:, None] - np.array(CENTRES)[None, :]), axis=1)
    for a, b in pairs:
        A, B = res[a + " conv"], res[b + " conv"]
        for sname in ("in_nat", "in_cx"):
            for k, c in enumerate(CENTRES):
                m = res[sname] & (blk == k)
                if not m.any():
                    continue
                onlyA, onlyB = int((A & ~B & m).sum()), int((B & ~A & m).sum())
                print("  %-13s vs %-13s %-7s t/t_end %.4f  n %5d  %5.1f%% vs %5.1f%%  only %4d / %4d  p %.2g"
                      % (a, b, "natural" if sname == "in_nat" else "complex", c, m.sum(),
                         100 * A[m].mean(), 100 * B[m].mean(), onlyA, onlyB, mcnemar(onlyA, onlyB)))


if __name__ == "__main__":
    main()
