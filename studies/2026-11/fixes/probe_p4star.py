"""planar4 from HLLD's p* on one lane, every steering, central differences:
the Newton history, and whether the final answer passes the certificate."""
import os, sys
import numpy as np
sys.path.insert(0, os.environ.get("RMHD_DIR", "/Users/miler/Codes/rmhd_final"))
np.seterr(all="ignore")
from rmhd.batched import planar4_b as P4, planar5_b as P5
POP = "/Users/miler/Codes/claude_scratch/r2d_retrain/anatomy/data/ledger/population.npz"
p = np.load(POP, allow_pickle=True); g = float(p["gamma"])
off = np.load("/Users/miler/Codes/claude_scratch/r2d_retrain/fixes/replay_fix3/replay_off.npz")
S4 = P4.make_solver(g); S5 = P5.make_solver(g); W = S4["waves"]
lane = int(sys.argv[1]); dl = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-4
UL, UR = p["UL"][lane], p["UR"][lane]
Bn = np.array([UL[7]])
L = [np.array([UL[j]]) for j in range(7)]; R = [np.array([UR[j]]) for j in range(7)]
L0, R0, alpha, _ = S5["to_planar"](L, R, Bn)
ps = abs(off["pstar"][lane])
pLF = pRF = ps
if os.environ.get("START") == "side":          # each fast wave just past its ahead pressure when p* is near it
    fr = float(os.environ.get("SIDE", "1e-3"))
    PL0, PR0 = float(L[1][0]), float(R[1][0])
    pLF = PL0 * (1 + fr) if abs(ps / PL0 - 1) < 0.02 else ps
    pRF = PR0 * (1 + fr) if abs(ps / PR0 - 1) < 0.02 else ps
    print("side start: P_L %.5f P_R %.5f p* %.5f -> p_LF %.5f p_RF %.5f" % (PL0, PR0, ps, pLF, pRF))
if os.environ.get("START") == "p5":            # the planar rescue's best iterate
    r5 = S5["solve"](L, R, Bn, verify_tol=None)
    pLF, pRF = float(np.exp(r5["unk"][0][0])), float(np.exp(r5["unk"][2][0]))
    print("planar5 best: nrm %.1e p_LF %.5f p_RF %.5f (p* %.5f)" % (r5["nrm"][0], pLF, pRF, ps))
print("lane", lane, "UL", np.round(UL, 5), "UR", np.round(UR, 5), "p*", ps)
A, _, eA = W["fast_wave"](np.array([pLF]), L0, "LF", Bn)
D, _, eD = W["fast_wave"](np.array([pRF]), R0, "RF", Bn)
KD = [tuple(int(c) for c in k) for k in (sys.argv[3].split(',') if len(sys.argv) > 3 else ['11', '10', '01', '00'])]
for kd in KD:
    kinds = np.array(kd, np.int8).reshape(2, 1)
    u = np.array([np.log(pLF), np.log(A[1][0]*(1-dl)) if kd[0] == 1 else A[5][0]*(1+dl),
                  np.log(D[1][0]*(1-dl)) if kd[1] == 1 else D[5][0]*(1+dl), np.log(pRF)])
    F = lambda x: S4["structure"](L0, R0, [np.array([c]) for c in x], Bn, kinds)
    hist = []
    for it in range(40):
        f, z, V, err, sh = F(u)
        nrm = np.max(np.abs(f[:, 0]))
        hist.append("%.0e%s" % (nrm, "E" if err[0] else ""))
        if err[0] or nrm < 1e-10:
            break
        J = np.zeros((4, 4)); bad = False
        for j in range(4):
            e = np.zeros(4); e[j] = 1e-6 * max(1.0, abs(u[j]))
            fp, _, _, ep, _ = F(u + e); fm, _, _, em, _ = F(u - e)
            if ep[0] and em[0]: bad = True; break
            J[:, j] = ((fp[:, 0] - fm[:, 0]) / (2*e[j]) if not (ep[0] or em[0])
                       else ((fp[:, 0] - f[:, 0]) / e[j] if not ep[0] else (f[:, 0] - fm[:, 0]) / e[j]))
        if bad:
            hist.append("Jfail"); break
        step = np.linalg.lstsq(J, -f[:, 0], rcond=None)[0]
        cap = min(1.0, 0.5 / max(np.abs(step).max(), 1e-300)); step *= cap
        lam = 1.0
        for _ in range(12):
            fc, _, _, ec, _ = F(u + lam*step)
            if not ec[0] and np.max(np.abs(fc[:, 0])) < nrm:
                u = u + lam*step; break
            lam *= 0.5
        else:
            hist.append("LSfail(cond %.0e)" % np.linalg.cond(J)); break
    f, z, V, err, sh = F(u)
    msg = "  kinds %s: %s" % (kd, " ".join(hist[:12] + (["..."] + hist[-3:] if len(hist) > 15 else hist[12:])))
    if not err[0] and np.max(np.abs(f[:, 0])) < 1e-8:
        cert, ordered = S4["certify"](L0, R0, z, V, sh, Bn)
        msg += "  | cert %.1e ordered %s shocks %s Bt %s" % (cert[0], ordered[0], sh[:, 0].astype(int),
                                                           np.round([zz[5][0] for zz in z], 4))
    print(msg)
