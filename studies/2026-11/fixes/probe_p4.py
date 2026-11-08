"""The four-unknown planar solver on chosen lanes: reason, kinds, residual;
and planar5 unflipped from the default start."""
import os, sys
import numpy as np
sys.path.insert(0, os.environ.get("RMHD_DIR", "/Users/miler/Codes/rmhd_final"))
np.seterr(all="ignore")
from rmhd.batched import planar4_b as P4, planar5_b as P5
POP = "/Users/miler/Codes/claude_scratch/r2d_retrain/anatomy/data/ledger/population.npz"
p = np.load(POP, allow_pickle=True); g = float(p["gamma"])
S4 = P4.make_solver(g); S5 = P5.make_solver(g)
for lane in [int(a) for a in sys.argv[1:]]:
    UL, UR = p["UL"][lane], p["UR"][lane]
    Bn = np.array([UL[7]])
    L = [np.array([UL[j]]) for j in range(7)]; R = [np.array([UR[j]]) for j in range(7)]
    r = S4["solve"](L, R, Bn)
    print("lane %d planar4: reason %d nrm %.1e cert %.1e ordered %s kinds %s it %d unk %s" % (
        lane, r["reason"][0], r["nrm"][0], r["cert_resid"][0], r["ordered"][0], r["kinds"][:, 0],
        r["n_iter"][0], np.round([u[0] for u in r["unk"]], 6)))
    for kd in ((0, 0), (1, 1), (0, 1), (1, 0)):
        r = S4["solve"](L, R, Bn, kinds=np.array(kd, np.int8).reshape(2, 1), retry=False)
        print("   kinds %s: reason %d nrm %.1e it %d" % (kd, r["reason"][0], r["nrm"][0], r["n_iter"][0]))
    r5 = S5["solve"](L, R, Bn, verify_tol=None)
    print("   planar5: conv %s nrm %.1e it %d" % (r5["converged_raw"][0], r5["nrm"][0], r5["n_iter"][0]))
