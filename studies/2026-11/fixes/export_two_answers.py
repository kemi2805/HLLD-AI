"""pgfplots coordinates of figs_two_answers.py's close-ups (lane 1013, left and
right slow/Alfven pairs): x = (xi - lambda_A) / 1e-6, B_t and rho, refused and regular."""
import numpy as np
import figs_two_answers as FT
import refused_data as CR
import verify_rmhd as VR
LANE = FT.LANE
Bn, g = FT.Bn, FT.g
Zr, Sr = CR.ans[LANE]
UL8 = np.concatenate([Zr[0], [Bn]])[None]; UR8 = np.concatenate([Zr[7], [Bn]])[None]
kr = VR.verify(UL8, UR8, Zr[None], Sr[None], g)["kind"][0]
reg = [r for r in np.load(CR.D + "/rd_solve_test.npy", allow_pickle=True) if r["lane"] == LANE and r["adm"]][0]
xr, sr = FT.answer_profile(Zr, Sr, kr)
xg, sg = FT.answer_profile(reg["Z"], reg["V"], reg["kind"])
for side, wrot, half in (("left", 1, 17e-6), ("right", 5, 19e-6)):
    Va = reg["V"][wrot]
    for name, x, s in (("refused", xr, sr), ("regular", xg, sg)):
        m = np.abs(x - Va) <= half
        xs, ys, rs = (x[m] - Va) / 1e-6, s[5][m], s[0][m]
        # thin the fan samples to ~30 points, keep every jump's endpoints
        keep = np.zeros(xs.size, bool); keep[0] = keep[-1] = True
        jumps = np.flatnonzero(np.abs(np.diff(xs)) < 1e-9)
        keep[jumps] = keep[np.minimum(jumps + 1, xs.size - 1)] = True
        step = max(1, xs.size // 30)
        keep[::step] = True
        xs, ys, rs = xs[keep], ys[keep], rs[keep]
        # extend to the window edges
        print("%% %s %s" % (side, name))
        print("Bt: " + " ".join("(%.3f,%.5f)" % (a, b) for a, b in zip(np.r_[-half / 1e-6, xs, half / 1e-6], np.r_[ys[0], ys, ys[-1]])))
        print("rho: " + " ".join("(%.3f,%.6f)" % (a, b) for a, b in zip(np.r_[-half / 1e-6, xs, half / 1e-6], np.r_[rs[0], rs, rs[-1]])))
