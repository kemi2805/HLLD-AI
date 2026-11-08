"""Along the planar slow family of a plateau state: shocks (|Bt| falls) and fans
(|Bt| rises) to a range of targets -- speed, the slow speed behind (Lax), the
entropy change.  Is the family convex here?

    python probe_family.py LANE XI_A XI_D
"""
import sys
import numpy as np
import os; sys.path.insert(0, os.environ.get("RMHD_DIR", "/Users/miler/Codes/rmhd_final"))
np.seterr(all="ignore")
from rmhd.batched import fullcontact_b as FB, wave_speeds_b as WB
from rmhd.batched.slow_shock_b import alfven_bounds

POP = "/Users/miler/Codes/claude_scratch/r2d_retrain/anatomy/data/ledger/population.npz"
T = "/Users/miler/Codes/claude_scratch/r2d_retrain/anatomy/data/tubes/tube_%d_%d.npz"
lane = int(sys.argv[1]); xa, xd = float(sys.argv[2]), float(sys.argv[3])
p = np.load(POP, allow_pickle=True); g = float(p["gamma"])
UL = p["UL"][lane]; Bn = np.array([UL[7]])
W = FB.make_waves(g)

def pg_of(X):
    rho, Pt, vx, vy, vz, By, Bz = X
    v2 = vx**2+vy**2+vz**2; W2 = 1/(1-v2); vB = vx*Bn+vy*By+vz*Bz
    return Pt - 0.5*((Bn**2+By**2+Bz**2)/W2 + vB**2)

def plateau(xi0):
    z = np.load(T % (lane, 4096)); t = float(z["t"])
    xi = ((np.arange(4096)+0.5)/4096-0.5)/t
    i = np.argmin(np.abs(xi-xi0))
    rho, pg, vx, vy, vz, By, Bz = (np.array([z[k][i]]) for k in ("rho","p","vx","vy","vz","By","Bz"))
    v2 = vx**2+vy**2+vz**2; W2 = 1/(1-v2); vB = vx*Bn+vy*By+vz*Bz
    return [rho, pg+0.5*((Bn**2+By**2+Bz**2)/W2+vB**2), vx, vy, vz, By, Bz]

zero = np.zeros(1)
for X, sw, col in ((plateau(xa), "LS", 1), (plateau(xd), "RS", 2)):
    e, la, ra, ok = WB.xi_all(*X, Bn, g)
    S0 = pg_of(X)/X[0]**g
    print("-- %s ahead Bt %.5f  la %.7f  LS %.7f  RS %.7f  ra %.7f  vx %.7f" % (sw, X[5][0], la[0], e[0,1], e[0,2], ra[0], X[2][0]))
    for f in np.linspace(1.25, 0.55, 15):
        bt = f*X[5][0]
        B, Vs, slack, err = W["slow_wave6"](np.array([bt]), zero, X, sw, Bn)
        eb, lab, rab, okb = WB.xi_all(*B, Bn, g)
        kind = "fan " if abs(bt) > abs(X[5][0]) else "shock"
        S1 = pg_of(B)/B[0]**g
        print("  %s |Bt| x%.3f err %d  Vs %.7f  lam_behind %.7f  (alfv %.7f)  Pt %.6f rho %.6f vx %.7f vt %.6f dS %+.1e" % (
            kind, f, err[0], Vs[0], eb[0, col], (lab if sw == "LS" else rab)[0], B[1][0], B[0][0], B[2][0], B[3][0], S1[0]/S0[0]-1))
