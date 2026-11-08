#!/usr/bin/env python3
"""Wall per block of steps (seconds) of the 4Q 64^2 runs side by side, and
each run's ratio to the control in that block."""
import glob, re, sys
R = "/mnt/rafast/miler/codes/rotor2d/HLLD"
s = sys.argv[1] if len(sys.argv) > 1 else "r2d"
B = int(sys.argv[2]) if len(sys.argv) > 2 else 25
TAGS = sys.argv[3].split(",") if len(sys.argv) > 3 else [
    "ctl1102", "fixfinal2", "finalgate", "finalrotgate", "finalrotc7lb_shortgate", "finalrotc7lbgate"]


def wall(pat):
    w = {}
    for f in sorted(glob.glob(pat)):
        for line in open(f, errors="ignore"):
            m = re.match(r"\s+step\s+(\d+) .*\((\d+)s\)", line)
            if m:
                w[int(m.group(1))] = int(m.group(2))
    return w


runs = []
for t in TAGS:
    logs = sorted(glob.glob(R + "/logs/%s_64_exact_%s_2*.log" % (s, t)))
    runs.append(wall(logs[-1]) if logs else {})
n = max(max(w) for w in runs if w)
print("steps      " + "".join("%14s" % t[:13] for t in TAGS))
for a in range(0, n, B):
    b = min(a + B, n)
    row = "%3d-%3d    " % (a + 1, b)
    c = runs[0]
    base = (c[b] - c.get(a, 0)) if (b in c and (a in c or a == 0)) else None
    for w in runs:
        if b in w and (a in w or a == 0):
            d = w[b] - w.get(a, 0)
            row += "%7d" % d + ("  x%4.2f" % (d / base) if base else "       ")
        else:
            row += "%14s" % "-"
    print(row)
print("total      " + "".join("%14s" % ("%d@%d" % (w[max(w)], max(w)) if w else "-") for w in runs))
