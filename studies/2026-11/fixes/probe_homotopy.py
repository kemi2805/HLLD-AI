"""A homotopy in the jump for planar4: R(theta) = L + theta (R - L) in the
solver-frame primitives, theta stepped up to 1, each root the next start
(central differences, fixed steering).

    python probe_homotopy.py LANE KINDS NSTEPS [THETA0]
"""
import os, sys
import numpy as np
sys.path.insert(0, os.environ.get("RMHD_DIR", "/Users/miler/Codes/rmhd_final"))
np.seterr(all="ignore")
from rmhd.batched import planar4_b as P4
POP = "/Users/miler/Codes/claude_scratch/r2d_retrain/anatomy/data/ledger/population.npz"
p = np.load(POP, allow_pickle=True); g = float(p["gamma"])
S4 = P4.make_solver(g); W = S4["waves"]
lane = int(sys.argv[1]); kd = tuple(int(c) for c in sys.argv[2]); K = int(sys.argv[3])
th0 = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0 / K
UL, UR = p["UL"][lane], p["UR"][lane]
Bn = np.array([UL[7]])
L = [np.array([UL[j]]) for j in range(7)]
kinds = np.array(kd, np.int8).reshape(2, 1)
u = None
for th in np.linspace(th0, 1.0, K):
    R = [np.array([UL[j] + th * (UR[j] - UL[j])]) for j in range(7)]
    if u is None:
        L0, R0 = S4["to_planar"](L, R, Bn)[:2]
        p0 = 0.5 * (L0[1] + R0[1])
        A, _, _ = W["fast_wave"](p0, L0, "LF", Bn); D, _, _ = W["fast_wave"](p0, R0, "RF", Bn)
        st = lambda k, X: np.log(X[1] * (1 - 1e-4)) if k == 1 else X[5] * (1 + 1e-4)
        u = [np.log(p0), st(kd[0], A), st(kd[1], D), np.log(p0)]
    r = S4["solve"](L, R, Bn, kinds=kinds, retry=False, unk_init=u, central=True, max_iter=40)
    print("  theta %.3f: converged %s reason %d nrm %.1e it %d" % (th, r["converged"][0], r["reason"][0], r["nrm"][0], r["n_iter"][0]))
    if not r["converged_raw"][0]:
        break
    u = [np.array([c[0]]) for c in r["unk"]]
if r["converged"][0]:
    print("  ROOT at theta 1: unk %s cert %.1e ordered %s shocks %s" % (np.round([c[0] for c in u], 7), r["cert_resid"][0], r["ordered"][0], r["shocks"][:, 0].astype(int)))
