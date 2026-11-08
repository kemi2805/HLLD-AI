"""Why the 4Q 128^2 run's steps got slower (2026-11-04): per step, the
wall time (slurm log) against what the solver did (harvest coverage
bitmaps: attempted, seven-wave, planar5, planar4, lost) and the time of day."""
import glob
import json
import os
import re
import sys
import time

import numpy as np

from r2d_compare64 import unpack

d = sys.argv[1]
t0 = time.mktime(time.strptime(sys.argv[2], "%Y-%m-%d %H:%M:%S"))
wall = {}
for line in open(os.path.join(d, "slurm.out"), errors="replace"):
    g = re.match(r"\s*step\s+(\d+) t=([0-9.]+).*\((\d+)s\)\s*$", line)
    if g:
        wall[int(g.group(1))] = float(g.group(3))
steps = np.array(sorted(wall))
w = np.array([wall[s] for s in steps])
per = np.diff(w)
cnt = {}
for f in sorted(glob.glob(os.path.join(d, "harvest", "coverage_*.npz"))):
    z = np.load(f)
    for r in range(z["sweep"].shape[0]):
        s, n = int(z["sweep"][r]), int(z["n"][r])
        st = s // 6 + 1
        c = cnt.setdefault(st, np.zeros(5, np.int64))
        m = {k: unpack(z[k][r], n) for k in ("attempted", "exact", "seven", "planar5", "planar4", "linear_fallback")}
        c += [m["attempted"].sum(), m["seven"].sum(), m["planar5"].sum(), m["planar4"].sum(),
              (m["attempted"] & ~m["exact"] & ~m["linear_fallback"]).sum()]
print("%5s %8s %6s %9s %8s %8s %7s %6s" % ("steps", "s/step", "clock", "attempted", "seven", "planar5", "planar4", "lost"))
for a in list(range(100, 700, 50)) + list(range(700, 738, 6)):
    b = min(a + 6 if a >= 700 else a + 20, int(steps[-1]))
    i = (steps[1:] > a) & (steps[1:] <= b)
    tt = time.strftime("%a %H:%M", time.localtime(t0 + w[steps == b][0]))
    c = sum(cnt.get(s, np.zeros(5)) for s in range(a + 1, b + 1)) / max(1, b - a)
    print("%3d-%3d %7.0f %10s %9.0f %8.0f %8.0f %7.0f %6.0f" % (a + 1, b, per[i].mean(), tt, *c))
