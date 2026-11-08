#!/usr/bin/env python3
"""Progress and ETA of the paired 64^2 runs: steps done of the control's
total, the wall ratio to the control over the last 30 steps, and the
projected end (control's remaining wall x that ratio)."""
import glob, re, time
R = "/mnt/rafast/miler/codes/rotor2d/HLLD"
CTRL = {"r2d": "r2d_64_exact_ctl1102", "ot": "ot_64_exact_ssretry", "rotor": "rotor_64_exact_ssretry"}


def wall(pat):
    fs = sorted(glob.glob(pat))
    w = {}
    for f in fs:
        for line in open(f, errors="ignore"):
            m = re.match(r"\s+step\s+(\d+) .*\((\d+)s\)", line)
            if m:
                w[int(m.group(1))] = int(m.group(2))
    return w


now = time.time()
SHORT = {"r2d": 200, "ot": 80, "rotor": 80}           # the planar-stage short runs (MAX_STEPS)
for tag in ("finalrotgate", "finalrotc7lbgate", "finalrotc7lb_shortgate"):
    for s, c in CTRL.items():
        logs = sorted(glob.glob(R + "/logs/%s_64_exact_%s_2*.log" % (s, tag)))
        if not logs:
            continue
        wt = wall(logs[-1])
        wc = wall(R + "/logs/%s_2*.log" % c)
        if not wt or not wc:
            continue
        n, nc = max(wt), max(wc)
        total = SHORT[s] if "short" in tag else nc
        a = max(1, n - 30)
        while a not in wt and a < n:
            a += 1
        ratio = (wt[n] - wt[a]) / max(wc.get(n, wc[nc]) - wc.get(a, 0), 1)
        rem = (wc[min(total, nc)] - wc.get(n, wc[nc])) * ratio
        print("%-5s %-20s step %3d / %3d  (%3.0f%%)  ratio to control (last 30 steps) %.2f  ~%.1f h left" % (
            s, tag, n, total, 100.0 * n / total, ratio, max(rem, 0) / 3600.0))
