"""Unflipped planar solve seeded from a flip branch's root: does the regular
(five-wave, no rotation) solution exist next to the crossed one?"""
import sys
import numpy as np
import os; sys.path.insert(0, os.environ.get("RMHD_DIR", "/Users/miler/Codes/rmhd_final"))
np.seterr(all="ignore")
from rmhd.batched import planar5_b as P5, wave_speeds_b as WB
POP = "/Users/miler/Codes/claude_scratch/r2d_retrain/anatomy/data/ledger/population.npz"
p = np.load(POP, allow_pickle=True); g = float(p["gamma"])
S = P5.make_solver(g); W = S["waves"]
for lane in [int(a) for a in sys.argv[1:]]:
    UL, UR = p["UL"][lane], p["UR"][lane]
    Bn = np.array([UL[7]])
    L = [np.array([UL[j]]) for j in range(7)]; R = [np.array([UR[j]]) for j in range(7)]
    print("==== lane", lane)
    roots = {}
    for fl in ((True, False), (False, True), (True, True)):
        r = S["solve"](L, R, Bn, flip=fl, verify_tol=None)
        if r["converged_raw"][0]:
            roots[fl] = r
            V = r["speeds"]; A, B, C, D = r["zones"]
            la = W["alfven_speed"](r["zones_mid"][0], Bn, "L")[0] if False else None
            print("  flip", fl, "root unk", np.round([u[0] for u in r["unk"]], 7), "full %.1e" % r["full_resid"][0],
                  "speeds", np.round([v[0] for v in V], 7))
    for fl, r in roots.items():
        for scale in (1.0, ):
            u0 = [np.array([c[0]]) for c in r["unk"]]
            r2 = S["solve"](L, R, Bn, unk3_init=u0, flip=(False, False), verify_tol=None, max_iter=80)
            print("  unflipped from the %s root: conv %s nrm %.2e it %d unk %s" % (
                fl, r2["converged_raw"][0], r2["nrm"][0], r2["n_iter"][0], np.round([u[0] for u in r2["unk"]], 7)))
            if r2["converged_raw"][0]:
                print("     speeds", np.round([v[0] for v in r2["speeds"]], 7), "full %.1e" % r2["full_resid"][0])
    # scan Bt around the root on the unflipped branch, fast pressures from the flip root
    if roots:
        r = list(roots.values())[0]
        u = [c[0] for c in r["unk"]]
        print("  unflipped residual along Bt_CD (fast pressures held at the root):")
        for d in np.linspace(-0.01, 0.01, 11):
            uu = [np.array([u[0]]), np.array([u[1] + d]), np.array([u[2]])]
            f, z, V, err, sl, _ = S["structure"](*S["to_planar"](L, R, Bn)[:2], uu, Bn)
            print("    dBt %+.4f err %d f %s" % (d, err[0], np.array2string(f[:, 0], precision=3)))
