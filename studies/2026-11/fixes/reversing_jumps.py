#!/usr/bin/env python
"""Classify every slow jump that reverses B_t in the answers the signed-field
slow wave found (b_starts runs: bstarts3, bstarts5_both): is it exactly a
relativistic rotational discontinuity (an Alfven wave -- the closed-form
rotation from the state ahead at the jump's speed reproduces the state
behind), an intermediate shock with rising entropy, or a jump whose entropy
falls (not admissible as a discontinuity)?

    python reversing_jumps.py OUT.npz
"""
import glob
import sys
import numpy as np
import sol_profiles as SP
from rmhd.batched import fullcontact_b as FB, wave_speeds_b as WB

off = np.load("/Users/miler/Codes/claude_scratch/r2d_retrain/fixes/replay_final2/replay_off.npz")
rows = []
for d in ("bstarts3", "bstarts5_both"):
    for f in sorted(glob.glob(d + "/shard_*.npz")):
        z = np.load(f, allow_pickle=True)
        for i in np.flatnonzero(z["ok"] & ~z["ctrl"]):
            if 2 in z["kinds"][:, i]:
                rows.append((int(z["lane"][i]), tuple(int(k) for k in z["kinds"][:, i]), str(z["how"][i])))
seen, out = set(), []
g = SP.GAMMA


def gas(X, Bn):
    rho, Pt, vx, vy, vz, By, Bz = X
    v2 = vx * vx + vy * vy + vz * vz
    W2 = 1.0 / (1.0 - v2)
    vB = vx * Bn + vy * By + vz * Bz
    b2 = (Bn ** 2 + By ** 2 + Bz ** 2) / W2 + vB ** 2
    return Pt - 0.5 * b2, b2


for lane, kinds, how in rows:
    if lane in seen:
        continue
    seen.add(lane)
    p = float(abs(off["pstar"][lane]))
    try:
        sol = SP.solve(lane, kinds, p, p)
    except AssertionError:
        continue
    A, B, C, D = sol["zones"]
    Bn = sol["Bn"]
    for name, ahead, behind, V, sh in (("LS", A, B, sol["V"][1], sol["shocks"][1]),
                                      ("RS", D, C, sol["V"][3], sol["shocks"][2])):
        if not sh or np.sign(ahead[5][0]) == np.sign(behind[5][0]):
            continue
        pa, b2a = gas(ahead, Bn); pb, b2b = gas(behind, Bn)
        dS = (pb / behind[0] ** g) / (pa / ahead[0] ** g) - 1.0
        la_a, ra_a = WB.xi_all(*ahead, Bn, g)[1:3]
        la_b, ra_b = WB.xi_all(*behind, Bn, g)[1:3]
        alf_a = (la_a if name == "LS" else ra_a)[0]
        alf_b = (la_b if name == "LS" else ra_b)[0]
        psi_b = np.arctan2(behind[6], behind[5])
        rot, err = FB.exact_rotation(ahead, Bn, np.array([V]), psi_b, g)
        drot = np.inf if err[0] else max(abs(rot[j][0] - behind[j][0]) / max(abs(behind[j][0]), 1e-3)
                                         for j in range(7))
        out.append(dict(lane=lane, wave=name, V=V, alf_a=alf_a, alf_b=alf_b,
                        Bt_a=ahead[5][0], Bt_b=behind[5][0], drho=behind[0][0] / ahead[0][0] - 1,
                        dp=pb[0] / pa[0] - 1, db=np.sqrt(b2b[0] / b2a[0]) - 1, dS=dS[0], drot=drot))
keys = list(out[0].keys())
np.savez(sys.argv[1], **{k: np.array([o[k] for o in out]) for k in keys})
dS = np.array([o["dS"] for o in out]); drot = np.array([o["drot"] for o in out])
print("reversing slow jumps: %d in %d lanes" % (len(out), len(set(o["lane"] for o in out))))
rd = drot < 1e-6
print("  exactly a rotational discontinuity (closed form reproduces the state behind to 1e-6): %d" % rd.sum())
print("    of them |dS| < 1e-9: %d;  entropy change range %.1e .. %.1e" % ((np.abs(dS[rd]) < 1e-9).sum(), dS[rd].min() if rd.any() else 0, dS[rd].max() if rd.any() else 0))
nr = ~rd
print("  not a rotation: %d -- entropy rising (>1e-9) %d, flat (|dS|<=1e-9) %d, falling (<-1e-9) %d" % (
    nr.sum(), (dS[nr] > 1e-9).sum(), (np.abs(dS[nr]) <= 1e-9).sum(), (dS[nr] < -1e-9).sum()))
for o in out:
    if o["lane"] in (900, 1245, 214):
        print("  lane %d %s: V %.7f Alfven ahead %.7f behind %.7f  Bt %+.4f -> %+.4f  drho %.1e dp %.1e d|b| %.1e dS %.1e  rotation-mismatch %.1e" % (
            o["lane"], o["wave"], o["V"], o["alf_a"], o["alf_b"], o["Bt_a"], o["Bt_b"], o["drho"], o["dp"], o["db"], o["dS"], o["drot"]))
