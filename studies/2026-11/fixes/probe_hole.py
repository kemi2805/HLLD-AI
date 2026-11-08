"""Lane's slow waves from the post-fast states of its flip root, to contact
fields across the root: which side's slow-shock solve leaves the weak root?"""
import sys
import numpy as np
import os; sys.path.insert(0, os.environ.get("RMHD_DIR", "/Users/miler/Codes/rmhd_final"))
np.seterr(all="ignore")
from rmhd.batched import planar5_b as P5, wave_speeds_b as WB
POP = "/Users/miler/Codes/claude_scratch/r2d_retrain/anatomy/data/ledger/population.npz"
p = np.load(POP, allow_pickle=True); g = float(p["gamma"])
S = P5.make_solver(g); W = S["waves"]
lane = int(sys.argv[1]); span = float(sys.argv[2]) if len(sys.argv) > 2 else 0.006
UL, UR = p["UL"][lane], p["UR"][lane]
Bn = np.array([UL[7]])
L = [np.array([UL[j]]) for j in range(7)]; R = [np.array([UR[j]]) for j in range(7)]
L0, R0 = S["to_planar"](L, R, Bn)[:2]
r = S["solve"](L, R, Bn, flip=(True, False), verify_tol=None)
u = [c[0] for c in r["unk"]]
A, _, _ = W["fast_wave"](np.exp(np.array([u[0]])), L0, "LF", Bn)
D, _, _ = W["fast_wave"](np.exp(np.array([u[2]])), R0, "RF", Bn)
zero = np.zeros(1)
print("root Bt_CD %.6f  A.Bt %.6f  D.Bt %.6f" % (u[1], A[5][0], D[5][0]))
for X, sw in ((A, "LS"), (D, "RS")):
    e, la, ra, ok = WB.xi_all(*X, Bn, g)
    print("--", sw, "la %.7f LS %.7f RS %.7f ra %.7f" % (la[0], e[0, 1], e[0, 2], ra[0]))
    for bt in u[1] + np.linspace(-span, span, 13):
        B, Vs, slack, err = W["slow_wave6"](np.array([bt]), zero, X, sw, Bn)
        print("   Bt %.6f err %d Vs %.8f  Pt %.6f rho %.6f vx %.7f vt %.7f" % (bt, err[0], Vs[0], B[1][0], B[0][0], B[2][0], B[3][0]))
