"""Which piece of the planar construction fails for a lane on a flip branch:
fast waves to the tube's plateau pressures, the pi-rotations, the slow waves."""
import os, sys
import numpy as np
sys.path.insert(0, os.environ.get("RMHD_DIR", "/Users/miler/Codes/rmhd_final"))
np.seterr(all="ignore")
from rmhd.batched import planar5_b as P5
POP = "/Users/miler/Codes/claude_scratch/r2d_retrain/anatomy/data/ledger/population.npz"
p = np.load(POP, allow_pickle=True); g = float(p["gamma"])
S = P5.make_solver(g); W = S["waves"]
lane, PA, PD = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
bts = [float(x) for x in sys.argv[4].split(",")]
UL, UR = p["UL"][lane], p["UR"][lane]
Bn = np.array([UL[7]])
L = [np.array([UL[j]]) for j in range(7)]; R = [np.array([UR[j]]) for j in range(7)]
L0, R0, alpha, _ = S["to_planar"](L, R, Bn)
A, VA, eA = W["fast_wave"](np.array([PA]), L0, "LF", Bn)
D, VD, eD = W["fast_wave"](np.array([PD]), R0, "RF", Bn)
print("fast: A err %d (Bt %.5f vt %.5f)  D err %d (Bt %.5f vt %.5f)" % (eA[0], A[5][0], A[3][0], eD[0], D[5][0], D[3][0]))
for X, side, nm in ((A, "L", "A"), (D, "R", "D")):
    Vs = W["alfven_speed"](X, Bn, side)
    out, err = W["alfven_checked"](Vs, X, W["seed_rot"](X, np.pi), Bn)
    print("  rotate %s by pi: err %d  Vs %.6f  -> Bt %.5f vt %.5f vx %.5f rho %.6f Pt %.6f" % (
        nm, err[0], Vs[0], out[5][0], out[3][0], out[2][0], out[0][0], out[1][0]))
    for bt in bts:
        for start, lab in ((X, "direct"), (out, "after pi")):
            B, V, sl, e = W["slow_wave6"](np.array([bt]), np.zeros(1), start, side + "S", Bn)
            print("    slow %s to Bt %+.4f (%s): err %d V %.6f slack %.1e Pt %.6f vx %.6f vt %.6f" % (
                side, bt, lab, e[0], V[0], sl[0], B[1][0], B[2][0], B[3][0]))
