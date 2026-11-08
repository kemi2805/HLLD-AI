"""planar4 structure from the tube start with a CENTRAL-difference Jacobian
of step h: does the zig-zag of lane 1099 come from the Jacobian?"""
import os, sys
import numpy as np
sys.path.insert(0, os.environ.get("RMHD_DIR", "/Users/miler/Codes/rmhd_final"))
np.seterr(all="ignore")
from rmhd.batched import planar4_b as P4, planar5_b as P5
POP = "/Users/miler/Codes/claude_scratch/r2d_retrain/anatomy/data/ledger/population.npz"
p = np.load(POP, allow_pickle=True); g = float(p["gamma"])
S4 = P4.make_solver(g); S5 = P5.make_solver(g)
lane = int(sys.argv[1]); kinds = np.array([int(c) for c in sys.argv[2]], np.int8).reshape(2, 1)
h = float(sys.argv[3]); u0 = [float(x) for x in sys.argv[4].split(",")]
UL, UR = p["UL"][lane], p["UR"][lane]
Bn = np.array([UL[7]])
L = [np.array([UL[j]]) for j in range(7)]; R = [np.array([UR[j]]) for j in range(7)]
L0, R0, alpha, _ = S5["to_planar"](L, R, Bn)
u = np.array(u0)
F = lambda x: S4["structure"](L0, R0, [np.array([c]) for c in x], Bn, kinds)
for it in range(40):
    f, z, V, err, sh = F(u)
    nrm = np.max(np.abs(f[:, 0]))
    print("  it %2d err %d nrm %.2e u %s" % (it, err[0], nrm, np.round(u, 8)))
    if err[0] or nrm < 1e-10:
        break
    J = np.zeros((4, 4))
    for j in range(4):
        e = np.zeros(4); e[j] = h * max(1.0, abs(u[j]))
        fp, _, _, ep, _ = F(u + e); fm, _, _, em, _ = F(u - e)
        J[:, j] = (fp[:, 0] - fm[:, 0]) / (2 * e[j])
    print("      cond %.1e" % np.linalg.cond(J))
    step = np.linalg.lstsq(J, -f[:, 0], rcond=None)[0]
    lam = 1.0
    for _ in range(14):
        fc, _, _, ec, _ = F(u + lam * step)
        if not ec[0] and np.max(np.abs(fc[:, 0])) < nrm:
            u = u + lam * step
            break
        lam *= 0.5
    else:
        print("  line search failed"); break
