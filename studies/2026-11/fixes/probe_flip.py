"""Lane's planar answer on every flip branch through production's acceptance
(_planar_and_ray): reason, zones, speeds, and the order of the waves."""
import sys
import numpy as np
sys.path.insert(0, "/Users/miler/Codes/HLLD")
np.seterr(all="ignore")
from rmhd.eos import set_eos
set_eos("ideal")
from src.physics import exact_flux as EF

POP = "/Users/miler/Codes/claude_scratch/r2d_retrain/anatomy/data/ledger/population.npz"
p = np.load(POP, allow_pickle=True)
g = float(p["gamma"])
P = EF._batched_parts(g)
names = ["rho", "Pt", "vx", "vt", "vz", "Bt", "Bz"]
for lane in [int(a) for a in sys.argv[1:]]:
    UL, UR = p["UL"][lane], p["UR"][lane]
    Bn = np.array([UL[7]])
    L = [np.array([UL[j]]) for j in range(7)]
    R = [np.array([UR[j]]) for j in range(7)]
    print("==== lane", lane, "Bn %.4f" % Bn[0])
    for fl in ((False, False), (True, False), (False, True), (True, True)):
        r = EF._planar_and_ray(P, g, L, R, Bn, accuracy=1e-8, max_iter=40, flip=fl)
        print("flip", fl, "reason", r["reason"][0], "resid %.1e full %.1e" % (r["resid"][0], r["full_resid"][0]))
        if r["VsLv"] is not None:
            Z = r["zones6"]
            print("   speeds LF %.6f LA %.6f LS %.6f | CD %.6f | RS %.6f RA %.6f RF %.6f" % (
                r["VsLv"][0][0], r["VsLv"][1][0], r["VsLv"][2][0], 0.5*(Z[2][2][0]+Z[3][2][0]),
                r["VsRv"][2][0], r["VsRv"][1][0], r["VsRv"][0][0]))
            for k, nm in enumerate(["A", "A2", "B", "C", "D2", "D"]):
                print("   %-2s" % nm, " ".join("%s %.6f" % (names[j], Z[k][j][0]) for j in (0, 1, 2, 3, 5)))
            print("   star", " ".join("%s %.6f" % (names[j], r["star"][j][0]) for j in (0, 1, 2, 3, 5)), "region", r["region"][0])
    print("   UL", np.round(UL, 6)); print("   UR", np.round(UR, 6))
