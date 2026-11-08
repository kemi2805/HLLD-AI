"""The reversing (intermediate) slow-shock branch from a post-fast state:
targets b of the opposite sign, the slow-shock solve with the window wide
open (numpy path, MARGIN patched), speed against the Alfven/slow speeds,
the slow speed behind (sonic?), entropy, and the RH residual."""
import os, sys
import numpy as np
sys.path.insert(0, os.environ.get("RMHD_DIR", "/Users/miler/Codes/rmhd_final"))
np.seterr(all="ignore")
from rmhd.batched import planar5_b as P5, slow_shock_b as SSB, wave_speeds_b as WB
from rmhd.batched.jump import rh_resid
POP = "/Users/miler/Codes/claude_scratch/r2d_retrain/anatomy/data/ledger/population.npz"
p = np.load(POP, allow_pickle=True); g = float(p["gamma"])
S = P5.make_solver(g); W = S["waves"]
lane, PA, PD = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
margin = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
UL, UR = p["UL"][lane], p["UR"][lane]
Bn = np.array([UL[7]])
L = [np.array([UL[j]]) for j in range(7)]; R = [np.array([UR[j]]) for j in range(7)]
L0, R0, alpha, _ = S["to_planar"](L, R, Bn)
A, _, _ = W["fast_wave"](np.array([PA]), L0, "LF", Bn)
D, _, _ = W["fast_wave"](np.array([PD]), R0, "RF", Bn)
SSB.MARGIN = margin
shock = SSB.make_solver(g)
def pg(X):
    rho, Pt, vx, vy, vz, By, Bz = X
    v2 = vx**2 + vy**2 + vz**2; W2 = 1/(1 - v2); vB = vx*Bn + vy*By + vz*Bz
    return Pt - 0.5*((Bn**2 + By**2 + Bz**2)/W2 + vB**2)
for X, sw, k in ((A, "LS", 1), (D, "RS", 2)):
    e, la, ra, ok = WB.xi_all(*X, Bn, g)
    print("-- %s ahead Bt %+.5f  la %.6f  slow %.6f  ra %.6f  fast %.6f  vx %.6f" % (
        sw, X[5][0], la[0], e[0, k], ra[0], e[0, 0 if sw == "LS" else 3], X[2][0]))
    K0 = pg(X)[0] / X[0][0]**g
    sgn = -np.sign(X[5][0])
    for f in (0.02, 0.1, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0):
        b = np.array([sgn * f * abs(X[5][0])])
        V, B, err = shock(b, np.zeros(1), X, sw, Bn, accuracy=1e-12)
        eb, lab, rab, okb = WB.xi_all(*B, Bn, g)
        r = rh_resid(X, B, V, Bn, g)[0]
        K1 = pg(B)[0] / B[0][0]**g
        print("   b %+.5f err %d V %.6f  RH %.1e  behind: slow %.6f alfv %.6f  rho %.6f Pt %.6f vx %.6f vt %.6f  dS %+.1e" % (
            b[0], err[0], V[0], r, eb[0, k], (lab if sw == "LS" else rab)[0], B[0][0], B[1][0], B[2][0], B[3][0], K1/K0 - 1))
