"""The closed-form pi-rotation (fullcontact_b.exact_rotation) on a lane's
post-fast states: its own RH residual, against the pinned solve's refusal."""
import os, sys
import numpy as np
sys.path.insert(0, os.environ.get("RMHD_DIR", "/Users/miler/Codes/rmhd_final"))
np.seterr(all="ignore")
from rmhd.batched import planar5_b as P5, fullcontact_b as FB
from rmhd.batched.jump import rh_resid
POP = "/Users/miler/Codes/claude_scratch/r2d_retrain/anatomy/data/ledger/population.npz"
p = np.load(POP, allow_pickle=True); g = float(p["gamma"])
S = P5.make_solver(g); W = S["waves"]
lane, PA, PD = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
UL, UR = p["UL"][lane], p["UR"][lane]
Bn = np.array([UL[7]])
L = [np.array([UL[j]]) for j in range(7)]; R = [np.array([UR[j]]) for j in range(7)]
L0, R0, alpha, _ = S["to_planar"](L, R, Bn)
A, _, _ = W["fast_wave"](np.array([PA]), L0, "LF", Bn)
D, _, _ = W["fast_wave"](np.array([PD]), R0, "RF", Bn)
for X, side in ((A, "L"), (D, "R")):
    Vs = W["alfven_speed"](X, Bn, side)
    psi = np.arctan2(X[6], X[5]) + np.pi
    seed, e = FB.exact_rotation(X, Bn, Vs, psi, g)
    r = rh_resid(X, seed, Vs, Bn, g)
    out, err = W["alfven_checked"](Vs, X, W["seed_rot"](X, np.pi), Bn)
    print("%s: closed form err %d RH %.1e  ahead Bt %.5f vt %.5f -> behind Bt %.5f vt %.5f vx %.6f rho %.6f | ladder err %d" % (
        side, e[0], r[0], X[5][0], X[3][0], seed[5][0], seed[3][0], seed[2][0], seed[0][0], err[0]))
