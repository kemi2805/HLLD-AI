"""ETA of the 4Q 512^2 exact run from the 128^2 run (2026-11-04).

The four-quadrant problem is self-similar and dx = 1/N is a power of 2, so
step n of every resolution is the same discrete state in cell units (the rho
range agrees to print precision at equal steps, 64/128/512).  The 512^2 cost
per step is then the 128^2 cost at the same step plus the constant cost of
the extra far-field cells.  Beyond the 128^2 run's last step, its per-step
cost is extrapolated (a + b sqrt(n), and a line through the last 250 steps).

    python eta_selfsim.py STEPS_512.txt STEPS_128.txt T_END
"""
import sys
import time

import numpy as np

from run_eta import load


def per_step(a):
    s, t, dt, w = a.T
    c = np.diff(w) / np.diff(s)
    return s[1:], np.convolve(c, np.ones(20) / 20, mode="same"), dt[1:]


def main():
    big, ref = load(sys.argv[1]), load(sys.argv[2])
    tend = float(sys.argv[3])
    sb, cb, dtb = per_step(big)
    sr, cr, dtr = per_step(ref)
    m = (sb >= 150) & (sb <= sb[-1] - 10)
    off = np.median(cb[m] - np.interp(sb[m], sr, cr))
    k = 20
    now = (big[-1, 3] - big[-1 - k, 3]) / (big[-1, 0] - big[-1 - k, 0])
    print("offset 512 - 128 at equal steps: %.0f s/step (median, steps 150-%d); 512 now %.0f s/step"
          % (off, sb[-1] - 10, now))
    fits = {}
    g = sr >= 100
    A = np.vstack([np.ones(g.sum()), np.sqrt(sr[g])]).T
    fits["a+b sqrt(n)"] = lambda n, c=np.linalg.lstsq(A, cr[g], rcond=None)[0]: c[0] + c[1] * np.sqrt(n)
    h = sr >= sr[-1] - 250
    p = np.polyfit(sr[h], cr[h], 1)
    fits["line, last 250"] = lambda n, p=p: np.polyval(p, n)
    ratio = np.median(dtb[-50:] / np.interp(sb[-50:], sr, dtr))    # 1/4 if the steps match
    dt_tail = np.mean(dtr[-100:]) * ratio
    t0, n0 = big[-1, 1], int(big[-1, 0])
    for name, f in fits.items():
        t, n, wall = t0, n0, 0.0
        while t < tend:
            n += 1
            inside = n <= sr[-1]
            c = (np.interp(n, sr, cr) if inside else f(n)) + off
            t += (np.interp(n, sr, dtr) * ratio) if inside else dt_tail
            wall += c
        fmt = time.strftime("%a %d %b %H:%M", time.localtime(time.time() + wall))
        print("  %-15s last step %d at %.0f s/step; %.0f h more -> %s" % (name, n, c, wall / 3600, fmt))


if __name__ == "__main__":
    main()
