"""Print the loose-fallback fixture: lane 62's post-left-fast state of its
regular root and the contact field where the ladder returns its grid cell."""
import os, sys
import numpy as np
sys.path.insert(0, os.environ.get("RMHD_DIR", "/Users/miler/Codes/rmhd_final"))
np.seterr(all="ignore")
from rmhd.batched import planar5_b as P5, slow_shock_b as SSB
from rmhd.batched.jump import rh_resid
POP = "/Users/miler/Codes/claude_scratch/r2d_retrain/anatomy/data/ledger/population.npz"
p = np.load(POP, allow_pickle=True); g = float(p["gamma"])
S = P5.make_solver(g); W = S["waves"]
UL, UR = p["UL"][62], p["UR"][62]
Bn = np.array([UL[7]])
L = [np.array([UL[j]]) for j in range(7)]; R = [np.array([UR[j]]) for j in range(7)]
L0, R0, alpha, _ = S["to_planar"](L, R, Bn)
u = [np.array([0.3029687]), np.array([0.3188758]), np.array([0.3029711])]
A, _, _ = W["fast_wave"](np.exp(u[0]), L0, "LF", Bn)
print("GAMMA =", repr(g)); print("BN =", repr(float(Bn[0])))
print("AHEAD = [" + ", ".join(repr(float(c[0])) for c in A) + "]")
print("BT =", repr(float(u[1][0])))
solve = SSB.make_solver(g)
for on in (False, True):
    SSB.LOOSE_RETRY = on
    V, B, e = solve(u[1], np.zeros(1), A, "LS", Bn, accuracy=1e-10)
    print("LOOSE_RETRY", on, "err", e[0], "Vs", repr(float(V[0])), "Pt", B[1][0], "RH %.2e" % rh_resid(A, B, V, Bn, g)[0])
    if on:
        print("BEHIND = [" + ", ".join(repr(float(c[0])) for c in B) + "]")
        print("VS =", repr(float(V[0])))
