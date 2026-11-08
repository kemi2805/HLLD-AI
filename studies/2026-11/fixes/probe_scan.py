"""Planar residual of one lane on one flip branch along Bt_CD, the two fast
pressures read from the tube's plateaus (xi_A left of the left cluster, xi_D
right of the right one).  Where is it unconstructible, where does it cross 0?

    python probe_scan.py LANE XI_A XI_D FLIPS BT_LO BT_HI [N]
"""
import os, sys
import numpy as np
sys.path.insert(0, os.environ.get("RMHD_DIR", "/Users/miler/Codes/rmhd_final"))
np.seterr(all="ignore")
from rmhd.batched import planar5_b as P5
POP = "/Users/miler/Codes/claude_scratch/r2d_retrain/anatomy/data/ledger/population.npz"
T = "/Users/miler/Codes/claude_scratch/r2d_retrain/anatomy/data/tubes/tube_%d_%d.npz"
p = np.load(POP, allow_pickle=True); g = float(p["gamma"])
S = P5.make_solver(g)
lane, xa, xd, fls = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
lo, hi = float(sys.argv[5]), float(sys.argv[6]); n = int(sys.argv[7]) if len(sys.argv) > 7 else 41
flip = (fls[0] == "F", fls[1] == "F")
UL, UR = p["UL"][lane], p["UR"][lane]
Bn = np.array([UL[7]])
L = [np.array([UL[j]]) for j in range(7)]; R = [np.array([UR[j]]) for j in range(7)]
L0, R0, alpha, _ = S["to_planar"](L, R, Bn)
z = np.load(T % (lane, 4096)); t = float(z["t"])
xi = ((np.arange(4096) + 0.5) / 4096 - 0.5) / t
def Pt_at(x0):
    i = np.argmin(np.abs(xi - x0))
    vx, vy, vz, Bx, By, Bz = (z[k][i] for k in ("vx", "vy", "vz", "Bx", "By", "Bz"))
    W2 = 1 / (1 - vx*vx - vy*vy - vz*vz); vB = vx*Bx + vy*By + vz*Bz
    return z["p"][i] + 0.5*((Bx*Bx + By*By + Bz*Bz)/W2 + vB*vB), By
PA, ByA = Pt_at(xa); PD, ByD = Pt_at(xd)
print("tube plateaus: Pt_A %.6f (By %.5f)  Pt_D %.6f (By %.5f); planar frame alpha %.3f (lab By -> planar sign %+d)" % (
    PA, ByA, PD, ByD, alpha[0], int(np.sign(np.cos(alpha[0])))))
bts = np.linspace(lo, hi, n)
u = [np.full(n, np.log(PA)), bts, np.full(n, np.log(PD))]
tile = lambda X: [np.repeat(c, n) for c in X]
f, zz, V, err, sl, _ = S["structure"](tile(L0), tile(R0), u, np.repeat(Bn, n), flip)
for k in range(n):
    print("  Bt %+.5f err %d  f %s  LS %.5f RS %.5f" % (bts[k], err[k], np.array2string(f[:, k], precision=3), V[1][k], V[3][k]))
