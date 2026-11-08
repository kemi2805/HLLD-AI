#!/usr/bin/env python
"""Pick the network on the VALIDATION faces only (2026-11-03, the proper
retrain; r2d_eval_seeds.py run on proper/val.npz).

    python r2d_choose2.py RESULT.npz OUT.txt name=path [name=path ...]

Rule, fixed before anything was run: the most faces converged on the natural
draw (one attempt); equal counts go to the most on the complex faces, then to
the candidate listed first.  The test faces are not looked at.  Writes
"name path" to OUT.txt.
"""
import sys

import numpy as np

res = np.load(sys.argv[1])
nat, cx = res["in_nat"], res["in_cx"]
cands = [s.split("=", 1) for s in sys.argv[3:]]
score = []
for i, (name, path) in enumerate(cands):
    c = res[name + " conv"]
    score.append((int(c[nat].sum()), int(c[cx].sum()), -i, name, path))
    print("%-14s natural %5d/%d (%.2f%%)   complex %5d/%d (%.2f%%)"
          % (name, c[nat].sum(), nat.sum(), 100 * c[nat].mean(), c[cx].sum(), cx.sum(), 100 * c[cx].mean()))
best = max(score)
open(sys.argv[2], "w").write("%s %s\n" % (best[3], best[4]))
print("chosen on VAL:", best[3], "->", best[4])
