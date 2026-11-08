#!/usr/bin/env python
"""planar4_b with rotations (structure/solve ``rot``) on refused lanes, batched:
seeded from the refused answer, each rotation combination, then the
independent verifier on the answer's seven-wave layout [A, A2, B, C, D2, D].

    PYTHONPATH=rmhd_fix python test_p4rot.py LANE [LANE ...]
"""
import sys
import numpy as np
import refused_data as CR
import sol_profiles as SP
import verify_rmhd as VR

g, pop, ans = CR.g, CR.pop, CR.ans
S4 = SP.S4
lanes = [int(a) for a in sys.argv[1:]]
n = len(lanes)
UL = pop["UL"][lanes]
UR = pop["UR"][lanes]
Bn = UL[:, 7].copy()
sL = [UL[:, j].copy() for j in range(7)]
sR = [UR[:, j].copy() for j in range(7)]
L0, R0, alpha, _ = S4["to_planar"](sL, sR, Bn)
ca, sa = np.cos(alpha), np.sin(alpha)
seed = [np.zeros(n) for _ in range(4)]
for k, lane in enumerate(lanes):
    Z, S = ans[lane]
    seed[0][k] = np.log(Z[1][1])
    seed[1][k] = ca[k] * Z[3][5] - sa[k] * Z[3][6]
    seed[2][k] = ca[k] * Z[4][5] - sa[k] * Z[4][6]
    seed[3][k] = np.log(Z[6][1])
kinds = np.full((2, n), 2, dtype=np.int8)
for combo in ((True, True), (True, False), (False, True)):
    rot = np.tile(np.array(combo)[:, None], (1, n))
    r = S4["solve"](sL, sR, Bn, kinds=kinds, retry=False, unk_init=[c.copy() for c in seed],
                    central=True, rot=rot, max_iter=25)
    A, B, C, D = r["zones"]
    A2, D2 = r["zones_rot"]
    Z = np.zeros((n, 8, 7))
    for j, X in enumerate((sL, A, A2, B, C, D2, D, sR)):
        Z[:, j] = np.stack(X, axis=1)
    VLA, VRA = r["speeds_rot"]
    V = np.stack([r["speeds"][0], VLA, r["speeds"][1], r["speeds"][2], r["speeds"][3], VRA, r["speeds"][4]], axis=1)
    UL8 = np.concatenate([Z[:, 0], Bn[:, None]], axis=1)
    UR8 = np.concatenate([Z[:, 7], Bn[:, None]], axis=1)
    VR.FAILWHY.clear()
    v = VR.verify(UL8, UR8, Z, V, g)
    print("== rotation L %s R %s" % combo)
    for k, lane in enumerate(lanes):
        kd = v["kind"][k]
        ent = " ".join("%s:%s%s" % (VR.NAMES[w], kd[w][0], "(%+.1e)" % v["dS"][k, w] if kd[w] == "disc" and w != 3 else "")
                       for w in range(7))
        print("   lane %5d: converged %s reason %d nrm %.1e it %d cert %.1e ordered %s | verifier weak %s admissible %s  %s" % (
            lane, r["converged"][k], r["reason"][k], r["nrm"][k], r["n_iter"][k], r["cert_resid"][k],
            r["ordered"][k], v["weak"][k], v["admissible"][k], ent))
