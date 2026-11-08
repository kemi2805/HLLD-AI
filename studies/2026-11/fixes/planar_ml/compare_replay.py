#!/usr/bin/env python3
"""The paired comparison of replay_net.py arms (same population, same order).

    python3 compare_replay.py OFF.npz ARM.npz [ARM.npz ...]

Per run and in total, for each arm: faces, answered by the late rung (and at
its first stage), by the four-unknown solver, lost; the late rung's planar4
solves and (lane, branch) pairs per sweep; the wall (all, late rung,
four-unknown). Against the first arm: the faces whose rung changed, and the
flux difference where both are exact. The first arm against the harvest:
how often it answers with the rung the run's harvest recorded (source 2 ->
the late rung, 3 -> the four-unknown solver).
"""
import sys

import numpy as np


def load(p):
    z = np.load(p, allow_pickle=True)
    return {k: z[k] for k in z.files}


def summary(z, m_face, m_sweep):
    r = z["face_rung"][m_face]
    st = z["face_stage"][m_face]
    return dict(
        faces=int(m_face.sum()), late=int((r == 3).sum()), late_s1=int(((r == 3) & (st == 1)).sum()),
        four=int((r == 4).sum()), lost=int((r == 0).sum()), sweeps=int(m_sweep.sum()),
        solves=float(z["sweep_solves"][m_sweep].sum()), pairs=float(z["sweep_branch_lanes"][m_sweep].sum()),
        wall=float(z["sweep_wall"][m_sweep].sum()), w_late=float(z["sweep__coplanar7_and_ray"][m_sweep].sum()),
        w_four=float(z["sweep__planar4_and_ray"][m_sweep].sum()),
        w_newton=float(z["sweep__run_pipeline"][m_sweep].sum()))


def main():
    arms = [load(p) for p in sys.argv[1:]]
    def name(a):
        n = ("net/" + str(a["fallback"])) if str(a["net"]) else "off"
        return n + ("+reg" if "regular" in a and bool(a["regular"]) else "")
    names = [name(a) for a in arms]
    base = arms[0]
    for a, name in zip(arms[1:], names[1:]):
        for k in ("face_run", "face_sweep", "face_src"):
            if not np.array_equal(a[k], base[k]):
                raise SystemExit("%s: population differs from the first arm (%s)" % (name, k))
    runs = list(base["runs"])
    for ri, run in enumerate(runs + ["TOTAL"]):
        mf = (base["face_run"] == ri) if run != "TOTAL" else np.ones(base["face_run"].size, bool)
        ms = (base["sweep_run"] == ri) if run != "TOTAL" else np.ones(base["sweep_run"].size, bool)
        print("== %s" % run)
        print("  %-12s %7s %7s %9s %6s %5s | %7s %8s | %8s %8s %8s %8s" % (
            "arm", "faces", "late", "late@1", "four", "lost", "solv/sw", "pairs/sw",
            "wall", "late", "four", "newton"))
        for a, name in zip(arms, names):
            s = summary(a, mf, ms)
            print("  %-12s %7d %7d %9d %6d %5d | %7.2f %8.1f | %8.0f %8.0f %8.0f %8.0f" % (
                name, s["faces"], s["late"], s["late_s1"], s["four"], s["lost"],
                s["solves"] / max(1, s["sweeps"]), s["pairs"] / max(1, s["sweeps"]),
                s["wall"], s["w_late"], s["w_four"], s["w_newton"]))
        src = base["face_src"][mf]
        rb = base["face_rung"][mf]
        agree = ((src == 2) & (rb == 3)) | ((src == 3) & (rb == 4))
        print("  first arm vs the harvest: same rung on %d of %d faces (%.2f%%)" % (
            agree.sum(), mf.sum(), 100.0 * agree.mean() if mf.any() else 0.0))
        for a, name in zip(arms[1:], names[1:]):
            ra = a["face_rung"][mf]
            both = (rb > 0) & (ra > 0)
            fb, fa = base["face_flux"][mf][both], a["face_flux"][mf][both]
            scale = np.maximum(np.abs(fb).max(axis=1, keepdims=True), 1e-300)
            rel = (np.abs(fa - fb) / scale).max(axis=1) if both.any() else np.zeros(0)
            moved = rb != ra
            trans = {}
            for x, y in zip(rb[moved], ra[moved]):
                trans[(int(x), int(y))] = trans.get((int(x), int(y)), 0) + 1
            print("  %s vs %s: rung changed on %d faces %s; flux where both exact: median %.2g, "
                  "99%% %.2g, max %.2g (%d faces above 1e-6)" % (
                      name, names[0], moved.sum(), dict(sorted(trans.items())),
                      np.median(rel) if rel.size else 0, np.quantile(rel, 0.99) if rel.size else 0,
                      rel.max() if rel.size else 0, int((rel > 1e-6).sum())))


if __name__ == "__main__":
    main()
