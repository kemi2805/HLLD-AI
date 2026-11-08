"""The pressure-parametrised slow fan (planar4.slow_wave_p) from a lane's
post-fast states to falling total pressures: does it carry B_t through 0?"""
import os, sys
import numpy as np
sys.path.insert(0, os.environ.get("RMHD_DIR", "/Users/miler/Codes/rmhd_final"))
np.seterr(all="ignore")
from rmhd.batched import planar4_b as P4, planar5_b as P5, wave_speeds_b as WB
POP = "/Users/miler/Codes/claude_scratch/r2d_retrain/anatomy/data/ledger/population.npz"
p = np.load(POP, allow_pickle=True); g = float(p["gamma"])
S4 = P4.make_solver(g); S5 = P5.make_solver(g); W = S5["waves"]
lane, PA, PD = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
UL, UR = p["UL"][lane], p["UR"][lane]
Bn = np.array([UL[7]])
L = [np.array([UL[j]]) for j in range(7)]; R = [np.array([UR[j]]) for j in range(7)]
L0, R0, alpha, _ = S5["to_planar"](L, R, Bn)
A, _, _ = W["fast_wave"](np.array([PA]), L0, "LF", Bn)
D, _, _ = W["fast_wave"](np.array([PD]), R0, "RF", Bn)
def pg(X):
    rho, Pt, vx, vy, vz, By, Bz = X
    v2 = vx**2 + vy**2 + vz**2; W2 = 1/(1 - v2); vB = vx*Bn + vy*By + vz*Bz
    return Pt - 0.5*((Bn**2 + By**2 + Bz**2)/W2 + vB**2)
for X, sw, k in ((A, "LS", 1), (D, "RS", 2)):
    e, la, ra, ok = WB.xi_all(*X, Bn, g)
    print("-- %s ahead Bt %+.5f Pt %.6f slow %.6f steer %d" % (sw, X[5][0], X[1][0], e[0, k], S4["steer"](X, sw, Bn)[0]))
    for dP in (1e-5, 3e-5, 1e-4, 2e-4, 3e-4, 5e-4, 1e-3, 3e-3):
        Y, V, err, sh = S4["slow_wave_p"](X[1] - dP, X, sw, Bn)
        ey, _, _, _ = WB.xi_all(*Y, Bn, g)
        print("   Pt-%.0e err %d  head %.6f tail %.6f  Bt %+.5f vt %.6f vx %.6f rho %.6f dS %+.1e" % (
            dP, err[0], e[0, k], ey[0, k], Y[5][0], Y[3][0], Y[2][0], Y[0][0], (pg(Y)/Y[0]**g)[0]/(pg(X)/X[0]**g)[0] - 1))
