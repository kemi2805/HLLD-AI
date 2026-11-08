#!/usr/bin/env python3
"""Wall per step (increment of the cumulative wall in the log) of the 4Q 64^2
control, rotation run and planar-stage short runs, side by side."""
import glob, re, sys
R = "/mnt/rafast/miler/codes/rotor2d/HLLD"
s = sys.argv[1] if len(sys.argv) > 1 else "r2d"
CTRL = {"r2d": "r2d_64_exact_ctl1102", "ot": "ot_64_exact_ssretry", "rotor": "rotor_64_exact_ssretry"}
TAGS = ["finalrotgate", "finalrotc7_shortgate", "finalrotc7a_shortgate", "finalrotc7ab_shortgate", "finalrotc7lb_shortgate"]
NAMES = ["control", "rot", "first", "after", "after+b", "late+b"]


def wall(pat):
    w = {}
    for f in sorted(glob.glob(pat)):
        for line in open(f, errors="ignore"):
            m = re.match(r"\s+step\s+(\d+) .*\((\d+)s\)", line)
            if m:
                w[int(m.group(1))] = int(m.group(2))
    return w


runs = [wall(R + "/logs/%s_2*.log" % CTRL[s])]
for t in TAGS:
    logs = sorted(glob.glob(R + "/logs/%s_64_exact_%s_2*.log" % (s, t)))
    runs.append(wall(logs[-1]) if logs else {})
n = max(max(w) if w else 0 for w in runs[2:])
print("step " + "".join("%9s" % k for k in NAMES))
for i in range(1, n + 1):
    row = "%4d " % i
    for w in runs:
        if i in w and (i - 1 in w or i == 1):
            row += "%9d" % (w[i] - w.get(i - 1, 0))
        else:
            row += "%9s" % "-"
    print(row)
