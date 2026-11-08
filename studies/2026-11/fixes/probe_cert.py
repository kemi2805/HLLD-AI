"""Planar5 answers that production refuses (full residual / self-crossing):
certify them independently of the Bt-method -- RH residual of every shock
(planar4.rh_residual), fan ordering, entropy across every wave -- and
compare their xi = 0 flux with the 4096-cell tube where one exists.

    python probe_cert.py LANE [LANE ...]
"""
import os, sys
import numpy as np
sys.path.insert(0, os.environ.get("RMHD_DIR", "/Users/miler/Codes/rmhd_final"))
np.seterr(all="ignore")
from rmhd.batched import planar4_b as P4, planar5_b as P5, wave_speeds_b as WB
POP = "/Users/miler/Codes/claude_scratch/r2d_retrain/anatomy/data/ledger/population.npz"
p = np.load(POP, allow_pickle=True); g = float(p["gamma"])
S4 = P4.make_solver(g); S5 = P5.make_solver(g); W = S5["waves"]
FLIPS = ((False, False), (True, False), (False, True), (True, True))


def pgas(X, Bn):
    rho, Pt, vx, vy, vz, By, Bz = X
    v2 = vx**2 + vy**2 + vz**2; W2 = 1 / (1 - v2); vB = vx*Bn + vy*By + vz*Bz
    return Pt - 0.5*((Bn**2 + By**2 + Bz**2)/W2 + vB**2)


def answers(lane):
    UL, UR = p["UL"][lane], p["UR"][lane]
    Bn = np.array([UL[7]])
    L = [np.array([UL[j]]) for j in range(7)]; R = [np.array([UR[j]]) for j in range(7)]
    L0, R0 = S5["to_planar"](L, R, Bn)[:2]
    d = P5.default_seed(L0, R0)
    seeds = [None]
    for bt in (abs(L0[5]), -abs(L0[5]), abs(R0[5]), -abs(R0[5])):
        seeds.append([d[0].copy(), np.array(bt, float), d[0].copy()])
    out = []
    for fl in FLIPS:
        for si, sd in enumerate(seeds):
            r = S5["solve"](L, R, Bn, unk3_init=sd, flip=fl, verify_tol=None)
            if not r["converged_raw"][0]:
                continue
            u = [c[0] for c in r["unk"]]
            if any(np.allclose(u, o["u"], atol=1e-7) and o["flip"] == fl for o in out):
                continue
            # rebuild in the planar frame for the certificate
            fv, z, V, err, sl, mid = S5["structure"](L0, R0, r["unk"], Bn, fl)
            A, B, C, D = z
            A2, D2 = mid
            shocks = np.array([[np.exp(u[0]) >= L0[1][0]], [abs(u[1]) < abs(A2[5][0])],
                               [abs(u[1]) < abs(D2[5][0])], [np.exp(u[2]) >= R0[1][0]]])
            rh = [S4["rh_residual"](a, b, v, Bn)[0] for a, b, v in ((L0, A, V[0]), (A2, B, V[1]), (D2, C, V[3]), (R0, D, V[4]))]
            rh = [x if s else 0.0 for x, s in zip(rh, shocks[:, 0])]
            K = [pgas(X, Bn)[0] / X[0][0]**g for X in (L0, A, A2, B, C, D2, D, R0)]
            dS = [K[1]/K[0]-1, K[3]/K[2]-1, K[4]/K[5]-1, K[6]/K[7]-1]   # LF, LS, RS, RF (behind/ahead)
            la = W["alfven_speed"](A2, Bn, "L")[0]; ra = W["alfven_speed"](D2, Bn, "R")[0]
            out.append(dict(u=u, flip=fl, seed=si, full=r["full_resid"][0], V=[v[0] for v in V], la=la, ra=ra,
                            rh=rh, dS=dS, shocks=shocks[:, 0], Bt=(A[5][0], A2[5][0], B[5][0], D2[5][0], D[5][0])))
    return out


for lane in [int(a) for a in sys.argv[1:]]:
    print("==== lane", lane)
    for o in answers(lane):
        fl = "".join("F" if f else "-" for f in o["flip"])
        V = o["V"]
        print(" flip %s seed %d full %.1e  unk %s" % (fl, o["seed"], o["full"], np.round(o["u"], 6)))
        print("    speeds LF %.5f [LA %.5f] LS %.5f CD %.5f RS %.5f [RA %.5f] RF %.5f" % (V[0], o["la"], V[1], V[2], V[3], o["ra"], V[4]))
        print("    shock? %s  RH %s  dS(behind/ahead-1) %s" % (o["shocks"].astype(int), np.array2string(np.array(o["rh"]), precision=1),
                                                         np.array2string(np.array(o["dS"]), precision=2)))
        print("    Bt  A %.4f A2 %.4f B %.4f | D2 %.4f D %.4f" % o["Bt"])
