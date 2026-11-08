"""Cost per unit t of the 4Q exact runs (2026-11-04): does the cost grow
with physical t at the same rate at every resolution?  s per unit t,
averaged over windows of t, from the step lines (run_eta.py's format)."""
import sys
import numpy as np
from run_eta import load

WIN = [(0.005, 0.015), (0.015, 0.03), (0.03, 0.045), (0.045, 0.06), (0.1, 0.12),
       (0.2, 0.22), (0.3, 0.32), (0.37, 0.38), (0.39, 0.40)]
print("%-22s" % "t window" + "".join("%12s" % a.split("=")[0] for a in sys.argv[1:]))
runs = []
for a in sys.argv[1:]:
    name, path = a.split("=")
    s, t, dt, w = load(path).T
    runs.append((t, w))
for lo, hi in WIN:
    row = "%-22s" % ("%.3f-%.3f" % (lo, hi))
    for t, w in runs:
        m = (t >= lo) & (t <= hi)
        if m.sum() < 3:
            row += "%12s" % "-"
            continue
        tt, ww = t[m], w[m]
        row += "%12.0f" % ((ww[-1] - ww[0]) / (tt[-1] - tt[0]))
    print(row)
