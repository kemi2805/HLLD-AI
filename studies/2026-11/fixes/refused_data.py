"""Shared data for the rotation re-solve: the ledger population, the captured
answers (verify/ledger), the planar solvers.  Paths from FIX_DIR / POP."""
import glob
import os
import numpy as np
from rmhd.batched import planar5_b as P5

D = os.environ.get("FIX_DIR", "/Users/miler/Codes/claude_scratch/r2d_retrain/fixes")
POP = os.environ.get("SP_POP", "/Users/miler/Codes/claude_scratch/r2d_retrain/anatomy/data/ledger/population.npz")
pop = np.load(POP, allow_pickle=True)
g = float(pop["gamma"])
S5 = P5.make_solver(g)
key = lambda A, B: [x.tobytes() + y.tobytes() for x, y in zip(np.ascontiguousarray(A), np.ascontiguousarray(B))]
lane_of = {k: i for i, k in enumerate(key(pop["UL"], pop["UR"]))}
ans = {}
for f in sorted(glob.glob(D + "/verify/ledger/chunk_*.npz")):
    if f.endswith("_verify.npz"):
        continue
    a = np.load(f, allow_pickle=True)
    for i, k in enumerate(key(a["U_L"], a["U_R"])):
        ans[lane_of.get(k, -1)] = (a["zones"][i], a["speeds"][i])


def one(x):
    return [np.array([float(c)]) for c in x]
