#!/usr/bin/env python
"""Pick the warm-start network for the next production run from a paired
test (r2d_eval_seeds.py result; 2026-11-02, retrain loop).

    python r2d_choose.py RESULT.npz OUT.txt name=path [name=path ...]

Candidates newest first.  The newest is kept unless an older one converges on
significantly more faces of the natural draw (more faces won, exact two-sided
McNemar p < 0.01); then that one is the choice, and so on down the list.  The
natural draw is the run's own mix of faces, complex ones at their real rate.
Writes the chosen checkpoint's path to OUT.txt.
"""
import math
import sys

import numpy as np


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (1 << n))


res = np.load(sys.argv[1])
out = sys.argv[2]
cands = [s.split("=", 1) for s in sys.argv[3:]]
nat = res["in_nat"]
best, best_path = cands[0]
for name, path in cands[1:]:
    A, B = res[name + " conv"][nat], res[best + " conv"][nat]
    won, lost = int((A & ~B).sum()), int((B & ~A).sum())
    p = mcnemar(won, lost)
    print("  %-6s vs %-6s natural %5.1f%% vs %5.1f%%  only-%s %d  only-%s %d  p %.2g"
          % (name, best, 100 * A.mean(), 100 * B.mean(), name, won, best, lost, p))
    if won > lost and p < 0.01:
        best, best_path = name, path
with open(out, "w") as f:
    f.write(best_path + "\n")
print("chosen: %s = %s -> %s" % (best, best_path, out))
