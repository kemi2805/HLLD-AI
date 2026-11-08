#!/usr/bin/env python
"""Paired comparison of the arms in an r2d_eval_seeds.py result (2026-11-02).

    python r2d_paired.py EVAL_RESULT.npz EVAL.npz PROXY.npz A:B [A:B ...]

For every pair A:B, on the natural draw and on the complex faces: converged
by A only / by B only and the exact two-sided McNemar p-value, once on all
test faces and once on the test faces that are NOT in the proxy set (the
proxy picked the *_proxy checkpoints and was drawn from the same held-out
rows, so only the faces outside it test those checkpoints cleanly).  Also
the union of every pair (both starts, one attempt each).
"""
import math
import sys

import numpy as np


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / (1 << n)
    return min(1.0, 2.0 * p)


def main():
    res, ev, px = (np.load(p) for p in sys.argv[1:4])
    pairs = [s.split(":") for s in sys.argv[4:]]
    idx = res["idx"]
    key = lambda A, B: [a.tobytes() + b.tobytes() for a, b in zip(A, B)]
    proxy = set(key(px["U_L"], px["U_R"]))
    in_px = np.array([k in proxy for k in key(ev["U_L"][idx], ev["U_R"][idx])], bool)
    sets = {"natural": res["in_nat"], "complex": res["in_cx"]}
    print("%d test faces, %d of them in the proxy set (natural %d, complex %d)"
          % (idx.size, in_px.sum(), (in_px & sets["natural"]).sum(), (in_px & sets["complex"]).sum()))
    for a, b in pairs:
        A, B = res[a + " conv"], res[b + " conv"]
        for sname, m in sets.items():
            for tag, mm in (("all", m), ("no proxy", m & ~in_px)):
                onlyA, onlyB = int((A & ~B & mm).sum()), int((B & ~A & mm).sum())
                print("  %-15s vs %-15s %-8s %-9s n %5d  %s %5.1f%%  %s %5.1f%%  only-%s %4d  only-%s %4d"
                      "  p %.2g  union %5.1f%%"
                      % (a, b, sname, tag, mm.sum(), a, 100 * A[mm].mean(), b, 100 * B[mm].mean(),
                         a, onlyA, b, onlyB, mcnemar(onlyA, onlyB), 100 * (A | B)[mm].mean()))


if __name__ == "__main__":
    main()
