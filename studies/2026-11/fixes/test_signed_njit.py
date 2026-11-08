"""fan_signed numpy vs njit on the lane-900/215 states (from the probes)."""
import os, sys
import numpy as np
sys.path.insert(0, os.environ["RMHD_DIR"])
np.seterr(all="ignore")
from rmhd.batched import planar4_b as P4, planar5_b as P5
POP = "/Users/miler/Codes/claude_scratch/r2d_retrain/anatomy/data/ledger/population.npz"
p = np.load(POP, allow_pickle=True); g = float(p["gamma"])
S4 = P4.make_solver(g); W = S4["waves"]
res = {}
for lane, ps in ((900, 107.934), (215, 3.4613)):
    UL, UR = p["UL"][lane], p["UR"][lane]
    Bn = np.array([UL[7]])
    L = [np.array([UL[j]]) for j in range(7)]; R = [np.array([UR[j]]) for j in range(7)]
    L0, R0 = S4["to_planar"](L, R, Bn)[:2]
    A, _, _ = W["fast_wave"](np.array([ps]), L0, "LF", Bn)
    for f in (0.5, -3.0, -10.0):
        y, V, e = S4["fan_signed"](A[5] * f, A, "LS", Bn)
        print(lane, f, e[0], np.array2string(np.array([c[0] for c in y]), precision=10))
