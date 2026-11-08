"""Compare the replay configurations (planar acceptance study, 2026-11-03).

    python compare_replay.py REPLAY_DIR BASE OTHER [OTHER ...]

Per configuration against BASE, on the flagged faces and on the controls:
how each face was answered (seven / planar5 / planar4 / not exact) and how
far its flux moved: max_k |F_k - F_k(BASE)| / max_k |F_k(BASE)|.  Then the
admissibility of every answer in the configuration's harvest (seven-wave and
planar five-wave rows; wavecheck, independent of the code under test).
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import admiss_all as AA     # noqa: E402

D, base, others = sys.argv[1], sys.argv[2], sys.argv[3:]
RUNG = ("not exact", "seven", "planar5", "planar4", "other")


def load(name):
    return dict(np.load(os.path.join(D, name, "faces.npz")))


def harvest_flags(name):
    fs = sorted(glob.glob(os.path.join(D, name, "harvest", "solved_*.npz")))
    n = bad = ent = 0
    for f in fs:
        z = np.load(f)
        if z["U_L"].shape[0] == 0:
            continue
        fl = AA.flags(np.asarray(z["U_L"], float), np.asarray(z["U_R"], float),
                      np.asarray(z["zones"], float), np.asarray(z["speeds"], float))
        n += z["U_L"].shape[0]
        bad += int(fl["any"].sum())
        ent += int(fl["entropy"].sum())
    return n, bad, ent


A = load(base)
scale = np.max(np.abs(A["F"]), axis=1)
q = lambda x: ("p50 %.1e p90 %.1e p99 %.1e max %.1e" % tuple(np.quantile(x, [.5, .9, .99, 1.0]))
               if x.size else "-")
for name in [base] + others:
    B = load(name)
    assert (B["U_L"] == A["U_L"]).all()
    n, bad, ent = harvest_flags(name)
    print("== %s: wall %.0f s; harvested rows %d, inadmissible %d (entropy %d)" % (name, B["wall"], n, bad, ent))
    for g, gn in ((0, "flagged"), (1, "control")):
        m = A["group"] == g
        dF = np.max(np.abs(B["F"] - A["F"]), axis=1) / scale
        line = "   %-8s " % gn + "  ".join("%s %d" % (RUNG[r], ((B["rung"] == r) & m).sum()) for r in range(4))
        print(line + "   bitwise-equal flux %d / %d" % (((dF == 0) & m).sum(), m.sum()))
        if name != base:
            for ra in range(4):
                for rb in range(4):
                    k = m & (A["rung"] == ra) & (B["rung"] == rb)
                    if k.any() and (ra != rb or (dF[k] > 0).any()):
                        print("      %-9s -> %-9s %5d   |dF|/|F| %s" % (RUNG[ra], RUNG[rb], k.sum(), q(dF[k])))
