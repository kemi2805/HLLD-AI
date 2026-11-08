"""Probe one ledger lane with the planar machinery: production's planar solve,
then the left/right slow wave from the tube's plateaus to a range of contact
fields -- which targets are constructible, at what speed, with what entropy."""
import sys
import numpy as np
import os; sys.path.insert(0, os.environ.get("RMHD_DIR", "/Users/miler/Codes/rmhd_final"))
np.seterr(all="ignore")
from rmhd.batched import planar5_b as P5, fullcontact_b as FB, wave_speeds_b as WB
from rmhd.batched.slow_shock_b import alfven_bounds

POP = "/Users/miler/Codes/claude_scratch/r2d_retrain/anatomy/data/ledger/population.npz"
T = "/Users/miler/Codes/claude_scratch/r2d_retrain/anatomy/data/tubes/tube_%d_%d.npz"
lane = int(sys.argv[1])
p = np.load(POP, allow_pickle=True)
g = float(p["gamma"])
UL, UR = p["UL"][lane], p["UR"][lane]
Bn = np.array([UL[7]])
L = [np.array([UL[j]]) for j in range(7)]
R = [np.array([UR[j]]) for j in range(7)]
S = P5.make_solver(g)
W = S["waves"]
for fl in ((False, False), (True, False), (False, True), (True, True)):
    r = S["solve"](L, R, Bn, flip=fl)
    print("flip", fl, "conv", r["converged"][0], "raw", r["converged_raw"][0], "nrm %.2e" % r["nrm"][0],
          "it", r["n_iter"][0], "full %.2e" % r["full_resid"][0], "unk", np.round([u[0] for u in r["unk"]], 6))

def entropy(X):
    rho, Pt, vx, vy, vz, By, Bz = X
    v2 = vx**2 + vy**2 + vz**2; W2 = 1/(1-v2); vB = vx*Bn + vy*By + vz*Bz
    b2 = (Bn**2 + By**2 + Bz**2)/W2 + vB**2
    pg = Pt - 0.5*b2
    return pg / rho**g, pg

def plateau(nc, xi0):
    z = np.load(T % (lane, nc)); t = float(z["t"])
    xi = ((np.arange(nc) + 0.5)/nc - 0.5)/t
    i = np.argmin(np.abs(xi - xi0))
    rho, pg, vx, vy, vz, By, Bz = (np.array([z[k][i]]) for k in ("rho","p","vx","vy","vz","By","Bz"))
    v2 = vx**2+vy**2+vz**2; W2 = 1/(1-v2); vB = vx*Bn+vy*By+vz*Bz
    b2 = (Bn**2+By**2+Bz**2)/W2 + vB**2
    return [rho, pg + 0.5*b2, vx, vy, vz, By, Bz]

if len(sys.argv) > 3:
    xa, xd = float(sys.argv[2]), float(sys.argv[3])
    A = plateau(4096, xa); D = plateau(4096, xd)
    zero = np.zeros(1)
    for X, sw in ((A, "LS"), (D, "RS")):
        e, la, ra, ok = WB.xi_all(*X, Bn, g)
        K0, pg0 = entropy(X)
        print("--", sw, "ahead By %.5f  Pt %.5f  la %.6f LS %.6f | RS %.6f ra %.6f" % (X[5][0], X[1][0], la[0], e[0,1], e[0,2], ra[0]))
        for bt in np.linspace(X[5][0], 0.6*X[5][0], 9):
            B, Vs, slack, err = W["slow_wave6"](np.array([bt]), zero, X, sw, Bn)
            K1, pg1 = entropy(B)
            print("   Bt %.5f err %d Vs %.7f slack %.1e  Pt %.5f rho %.5f vx %.6f vt %.6f  dS %.2e" % (
                bt, err[0], Vs[0], slack[0], B[1][0], B[0][0], B[2][0], B[3][0], K1[0]/K0[0]-1))
