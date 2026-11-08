"""Bracket the finish of a running 2D exact run from its slurm step lines
(2026-11-03; 'step N t=T dt=D ... (Ss)' with S the elapsed wall in seconds).

    python run_eta.py STEPS.txt T_END [REF_STEPS.txt]

flat  = the last 20 steps' s/step and dt held to t_end;
scaled = the cost per unit t of REF (the same problem at another resolution,
further along) at each t, scaled to this run's current cost -- the cost per
step grows as the solution develops structure.
"""
import re
import sys
import time

import numpy as np

PAT = re.compile(r"step\s+(\d+)\s+t=([\d.eE+-]+)\s+dt=([\d.eE+-]+).*\((\d+)s\)")


def load(path):
    rows = [PAT.search(l) for l in open(path)]
    return np.array([[float(x) for x in m.groups()] for m in rows if m])


def main():
    a = load(sys.argv[1])
    tend = float(sys.argv[2])
    step, t, dt, wall = a.T
    now = time.time()
    fmt = lambda s: time.strftime("%a %d %H:%M", time.localtime(now + s))
    k = min(20, len(a) - 1)
    sps = (wall[-1] - wall[-1 - k]) / (step[-1] - step[-1 - k])
    dts = np.mean(dt[-k:])
    rest_steps = (tend - t[-1]) / dts
    flat = rest_steps * sps
    print("step %d  t %.4f / %.2f (%.1f%%)  wall %.1f h  last %d steps: %.0f s/step, dt %.2e"
          % (step[-1], t[-1], tend, 100 * t[-1] / tend, wall[-1] / 3600, k, sps, dts))
    print("  flat:   ~%d more steps, %.1f h -> %s" % (rest_steps, flat / 3600, fmt(flat)))
    if len(sys.argv) > 3:
        r = load(sys.argv[3])
        rs, rt, rdt, rw = r.T
        # cost per unit t of the reference at time t (smoothed over 20 steps)
        cpt = np.gradient(rw, rt)
        cpt = np.convolve(cpt, np.ones(20) / 20, mode="same")
        cur = np.interp(t[-1], rt, cpt)
        mine = sps / dts
        grid = np.linspace(t[-1], min(tend, rt[-1]), 400)
        rest = np.trapz(np.interp(grid, rt, cpt) * mine / cur, grid)
        cover = min(tend, rt[-1])
        note = "" if cover >= tend else "  (reference reaches only t %.3f; flat beyond)" % rt[-1]
        if cover < tend:
            rest += (tend - cover) * np.interp(cover, rt, cpt) * mine / cur
        print("  scaled: %.1f h -> %s%s" % (rest / 3600, fmt(rest), note))


if __name__ == "__main__":
    main()
