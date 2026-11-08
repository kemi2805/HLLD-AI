#!/usr/bin/env python
"""Is a slow compound wave the admissible answer where the gate refuses ours?
(user 2026-11-05: "take a look on the entropy fall, we may get this also up to
99.99%")

The refused answers (final arm exact, finalgate not) reverse B_t at a slow jump
that EXPANDS: the density falls, the characteristics diverge from it, it runs
between the Alfven speeds of its two sides, and the entropy falls by 1e-9..7e-5.
The classic admissible way to reverse the field in a coplanar problem without
a pi-rotation is the Brio-Wu compound wave: an intermediate shock that leaves
the flow behind it sonic, with a slow fan riding attached to its back
(HLLD scripts/compound_wave.py).  Here it is seeded from our own answer -- fast
pressures and the contact's B_t -- with the compound on every side whose slow
jump reverses the field, and judged by the element's certificate plus the
entropy across every jump and the order of the waves.

    PYTHONPATH=rmhd_fix python compound_refused.py LANE [LANE ...]  (numpy slow shock: RMHD_SLOWSHOCK unset)
"""
import glob
import os
import sys
import numpy as np

sys.path.insert(0, "/Users/miler/Codes/HLLD/scripts")
import compound_wave as CW                                        # noqa: E402
from rmhd.batched import planar5_b as P5                          # noqa: E402
import rmhd_ideal as RI                                           # noqa: E402

D = "/Users/miler/Codes/claude_scratch/r2d_retrain/fixes"
pop = np.load("/Users/miler/Codes/claude_scratch/r2d_retrain/anatomy/data/ledger/population.npz", allow_pickle=True)
g = float(pop["gamma"])
E = CW.make(g)
S5 = P5.make_solver(g)
key = lambda A, B: [x.tobytes() + y.tobytes() for x, y in zip(np.ascontiguousarray(A), np.ascontiguousarray(B))]
lane_of = {k: i for i, k in enumerate(key(pop["UL"], pop["UR"]))}
ans = {}
for f in sorted(glob.glob(D + "/verify/ledger/chunk_*.npz")):
    if f.endswith("_verify.npz"):
        continue
    a = np.load(f, allow_pickle=True)
    for i, k in enumerate(key(a["U_L"], a["U_R"])):
        ans[lane_of.get(k, -1)] = (a["zones"][i], a["speeds"][i])


def one(x):
    return [np.array([float(c)]) for c in x]


def entropy_ratio(A, B, Bn):
    return float(RI.entropy(B, Bn, g)[0] / RI.entropy(A, Bn, g)[0] - 1.0)


for lane in ([int(s) for s in sys.argv[1:]] if __name__ == "__main__" else []):
    UL, UR = pop["UL"][lane], pop["UR"][lane]
    Bn = np.array([UL[7]])
    L0, R0, alpha, _ = S5["to_planar"](one(UL[:7]), one(UR[:7]), Bn)
    Z, S = ans[lane]
    ca, sa = np.cos(alpha[0]), np.sin(alpha[0])
    bt = lambda z: ca * z[5] - sa * z[6]
    # which slow jumps reverse the field in our answer (planar frame)
    revL = bt(Z[2]) * bt(Z[3]) < 0
    revR = bt(Z[5]) * bt(Z[4]) < 0
    mode = (bool(revL), bool(revR))
    u0 = [np.array([np.log(Z[1][1])]), np.array([bt(Z[3])]), np.array([np.log(Z[6][1])])]
    print("== lane %d  Bn %.4f  ours: Bt_A %.4f Bt* %.4f Bt_D %.4f  compound on %s" % (
        lane, Bn[0], bt(Z[2]), bt(Z[3]), bt(Z[5]), mode))
    if not any(mode):
        print("   no reversing slow jump")
        continue
    f0, e0, st0 = E["structure"](L0, R0, u0, Bn, mode)
    print("   residual at our answer: %s err %s" % (np.abs(f0[:, 0]).max(), e0[0]))
    for side in ("L", "R"):
        if side in st0:
            Vsh, Vtl, Sx, b = st0[side]
            Ax = st0["A" if side == "L" else "D"]
            print("   %s sonic IS from our fast state: V %.6e b %.5f  IS entropy %+.2e" % (
                side, Vsh[0], b[0], entropy_ratio(Ax, Sx, Bn)))
    u, nrm = E["solve"](L0, R0, Bn, u0, mode)
    ok, rep, st = E["verify"](L0, R0, Bn, u, mode)
    print("   solve: |f| %.1e  unk %s  certificate %s" % (nrm[0], np.round([c[0] for c in u], 7), ok[0]))
    if np.isfinite(nrm[0]) and nrm[0] < 1e-6:
        # entropy across every jump of the compound answer
        A, B, C, Dd = st["A"], st["B"], st["C"], st["D"]
        msg = "   entropy: LF %+.2e RF %+.2e" % (entropy_ratio(one([c[0] for c in L0]), A, Bn),
                                                 entropy_ratio(one([c[0] for c in R0]), Dd, Bn))
        for side in ("L", "R"):
            if side in st:
                Vsh, Vtl, Sx, b = st[side]
                Ax = A if side == "L" else Dd
                msg += "  %s IS %+.2e (b %.4f, V %.6e, tail %.6e)" % (side, entropy_ratio(Ax, Sx, Bn), b[0], Vsh[0], Vtl[0])
        print(msg)
        print("   report:", {k: (float(v[0]) if np.ndim(v) else v) for k, v in rep.items()})
