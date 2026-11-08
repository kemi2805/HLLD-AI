#!/usr/bin/env python3
"""Cumulative wall of every run at the same step relative to the control, per
problem: the latest step the REFERENCE run (the last tag) has reached; a run
that stopped earlier (cancelled) is shown at its own last step, marked.

    python3 same_step.py [TAG ...]   (default: the rotation runs and the planar-stage short runs)
"""
import glob, re, sys
R = "/mnt/rafast/miler/codes/rotor2d/HLLD"
CTRL = {"r2d": "r2d_64_exact_ctl1102", "ot": "ot_64_exact_ssretry", "rotor": "rotor_64_exact_ssretry"}
TAGS = sys.argv[1:] or ["finalrotgate", "finalrotc7_shortgate", "finalrotc7ab_shortgate", "finalrotc7lb_shortgate"]
# the full late + batched runs (81297-81299): python3 same_step.py finalrotgate finalrotc7lbgate
NAMES = {"finalrotgate": "rotation run", "finalrotc7_shortgate": "planar first",
         "finalrotc7a_shortgate": "planar after", "finalrotc7ab_shortgate": "after+batched",
         "finalrotc7lb_shortgate": "late+batched", "finalrotc7lbgate": "late+batched (full)"}


def wall(pat):
    w = {}
    for f in sorted(glob.glob(pat)):
        for line in open(f, errors="ignore"):
            m = re.match(r"\s+step\s+(\d+) .*\((\d+)s\)", line)
            if m:
                w[int(m.group(1))] = int(m.group(2))
    return w


for s, c in CTRL.items():
    runs = {"control": wall(R + "/logs/%s_2*.log" % c)}
    for tag in TAGS:
        logs = sorted(glob.glob(R + "/logs/%s_64_exact_%s_2*.log" % (s, tag)))
        if logs:
            w = wall(logs[-1])
            if w:
                runs[tag] = w
    ref = TAGS[-1] if TAGS[-1] in runs else max(runs, key=lambda k: 0 if k == "control" else max(runs[k]))
    n = min(max(runs[ref]), max(runs["control"]))
    msg = "%-5s at step %3d:" % (s, n)
    for k, w in runs.items():
        m = n if n in w else max(w)
        if m in runs["control"]:
            msg += "  %s %5.0f s (%.2fx%s)" % (NAMES.get(k, k), w[m], w[m] / runs["control"][m],
                                             "" if m == n else ", at its last step %d" % m)
    print(msg)
