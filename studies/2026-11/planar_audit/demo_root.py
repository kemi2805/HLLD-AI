"""One flagged face (train3 row 486645): the slow-shock sub-solver on the
stored R3 and target B_t, in the planar frame (as planar5 calls it) and in the
raw frame (rotated by pi, as the audit's fullfuncv6 evaluation calls it)."""
import os
os.environ["RMHD_SLOWSHOCK"] = "numpy"
import numpy as np
from rmhd.batched import slow_shock_b as SSB
from rmhd.batched import jump as J
import wavecheck as WC
G = 5.0 / 3.0
z = np.load("rows_1103.npz", allow_pickle=True)
i = int(np.flatnonzero(z["meta_row"] == 486645)[0])
Z, Bn = z["bad_zones"][i], z["bad_U_L"][i, 7]
R3, R4 = Z[2], Z[3]
solve = SSB.make_solver(G)
def run(R3, target, label):
    ah = [np.array([c]) for c in R3]
    Vs, bh, err = solve(np.array([target[0]]), np.array([target[1]]), ah, "LS", np.array([Bn]), accuracy=1e-10)
    bh = [np.asarray(c).ravel() for c in bh]
    rh = J.rh_resid(ah, bh, np.asarray(Vs).ravel(), np.array([Bn]), G)
    K = lambda s: WC.pgas(s, np.array([Bn])) / s[0] ** G
    la = SSB.alfven_bounds(ah, np.array([Bn]), G)[0]
    print("%-12s err %s  Vs %+.7f (Alfven ahead %+.7f)  rho_b/rho_a %.4f  P_tot behind %.5f  K_b/K_a %.5f  RH %.1e" % (
        label, bool(np.asarray(err)[0]), float(np.asarray(Vs).ravel()[0]), la[0], bh[0][0] / R3[0], bh[1][0],
        (K(bh) / K(ah))[0], rh[0]))
flip = lambda s: np.array([s[0], s[1], s[2], -s[3], -s[4], -s[5], -s[6]])
print("stored R4: rho/rho3 %.4f  P_tot %.5f  LS speed %+.7f" % (R4[0] / R3[0], R4[1], z["bad_speeds"][i, 2]))
run(R3, (R4[5], R4[6]), "raw frame")
run(flip(R3), (-R4[5], -R4[6]), "planar frame")
