"""planar4 with given slow-wave kinds from a start read off the tube (fast
pressures at the plateaus, slow strengths = ln P_tot at the contact plateau):
does a root exist?  Plain damped Newton on planar4's own structure/jacobian."""
import os, sys
import numpy as np
sys.path.insert(0, os.environ.get("RMHD_DIR", "/Users/miler/Codes/rmhd_final"))
np.seterr(all="ignore")
from rmhd.batched import planar4_b as P4, planar5_b as P5
POP = "/Users/miler/Codes/claude_scratch/r2d_retrain/anatomy/data/ledger/population.npz"
T = os.environ.get("TUBES", "/Users/miler/Codes/claude_scratch/r2d_retrain/anatomy/data/tubes") + "/tube_%d_%d.npz"
p = np.load(POP, allow_pickle=True); g = float(p["gamma"])
S4 = P4.make_solver(g); S5 = P5.make_solver(g)
lane = int(sys.argv[1]); xa, xc, xd = (float(x) for x in sys.argv[2:5])
kinds = np.array([int(c) for c in sys.argv[5]], np.int8).reshape(2, 1)
UL, UR = p["UL"][lane], p["UR"][lane]
Bn = np.array([UL[7]])
L = [np.array([UL[j]]) for j in range(7)]; R = [np.array([UR[j]]) for j in range(7)]
L0, R0, alpha, _ = S5["to_planar"](L, R, Bn)
z = np.load(T % (lane, 4096)); t = float(z["t"]); xi = ((np.arange(4096) + 0.5)/4096 - 0.5)/t
def at(x0):
    i = np.argmin(np.abs(xi - x0))
    vx, vy, vz, Bx, By, Bz = (z[k][i] for k in ("vx", "vy", "vz", "Bx", "By", "Bz"))
    W2 = 1/(1 - vx*vx - vy*vy - vz*vz); vB = vx*Bx + vy*By + vz*Bz
    return z["p"][i] + 0.5*((Bx*Bx + By*By + Bz*Bz)/W2 + vB*vB), By * np.cos(alpha[0])
PA, _ = at(xa); PC, BtC = at(xc); PD, _ = at(xd)
if len(sys.argv) > 6:
    PA = PD = float(sys.argv[6])
    PC = float(sys.argv[7]) if len(sys.argv) > 7 else PA
    BtC = 0.5 * (L0[5][0] + R0[5][0])
u = [np.log(np.array([PA])),
     np.array([np.log(PC) if kinds[0, 0] == 1 else BtC]),
     np.array([np.log(PC) if kinds[1, 0] == 1 else BtC]),
     np.log(np.array([PD]))]
print("start: PA %.6f PC %.6f (Bt_C planar %+.5f) PD %.6f kinds %s" % (PA, PC, BtC, PD, kinds[:, 0]))
for it in range(30):
    f, J, zz, V, err, sh = S4["jacobian"](L0, R0, u, Bn, kinds)
    nrm = np.max(np.abs(f[:, 0]))
    print("  it %2d err %d nrm %.2e u %s" % (it, err[0], nrm, np.round([c[0] for c in u], 7)))
    if err[0] or nrm < 1e-10:
        break
    step = np.linalg.lstsq(J[0], -f[:, 0], rcond=None)[0]
    lam = 1.0
    for _ in range(12):
        cand = [u[c] + lam * step[c] for c in range(4)]
        fc, _, _, ec, _ = S4["structure"](L0, R0, cand, Bn, kinds)
        if not ec[0] and np.max(np.abs(fc[:, 0])) < nrm:
            u = cand
            break
        lam *= 0.5
    else:
        print("  line search failed"); break
if not err[0] and nrm < 1e-8:
    cert, ordered = S4["certify"](L0, R0, zz, V, sh, Bn)
    print("  ROOT: cert %.1e ordered %s speeds %s shocks %s" % (cert[0], ordered[0], np.round([v[0] for v in V], 6), sh[:, 0]))
    print("  zones Bt: A %+.5f B %+.5f C %+.5f D %+.5f" % tuple(zq[5][0] for zq in zz))
