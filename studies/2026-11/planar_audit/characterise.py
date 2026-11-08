import numpy as np
import wavecheck as WC
G = 5.0 / 3.0
z = np.load("admiss_flagged_1103.npz")
UL, UR, Z, S = z["all_U_L"], z["all_U_R"], z["all_zones"], z["all_speeds"]
src = z["all_src"]
a = WC.admissibility(UL, UR, Z, S, G)
n = UL.shape[0]
def q(x):
    x = x[np.isfinite(x)]
    return "n %5d  p10 %.3g p50 %.3g p90 %.3g min %.3g" % (x.size, *np.quantile(x, [0.1, 0.5, 0.9]), x.min()) if x.size else "n 0"
for s, nm in ((0, "seven"), (1, "planar5")):
    m = src == s
    print("==", nm, "flagged rows", m.sum())
    for w, col, fcol in (("LS", 2, 0), ("RS", 4, 6)):
        sh = (a[w + "_solver_kind"] == 1) & m
        ent = sh & (a[w + "_K"] < 1 - 1e-6)
        print("  %s entropy-falling shocks %d: K ratio %s" % (w, ent.sum(), q(a[w + "_K"][ent])))
        print("     their density ratio %s" % q(a[w + "_drho"][ent]))
        print("     |Bt| ratio %s" % q(a[w + "_dBt"][ent]))
        # riding on the fast shock: same speed as the fast wave on its side
        fsh = a[("LF" if w == "LS" else "RF") + "_solver_kind"] == 1
        dv = np.abs(S[:, col] - S[:, fcol])
        ride = ent & fsh & (dv < 1e-4)
        print("     riding on a fast SHOCK (|s_slow - s_fast| < 1e-4): %d; beyond Alfven ahead: %d" % (
            ride.sum(), (ent & (a[w + "_lax_margins"][2 if w == "LS" else 3] < -1e-6)).sum()))
    allK = np.minimum(np.where(a["LS_solver_kind"] == 1, a["LS_K"], 1), np.where(a["RS_solver_kind"] == 1, a["RS_K"], 1))
    for thr in (1 - 1e-6, 0.9999, 0.999, 0.99, 0.9):
        print("  rows with an entropy-falling slow shock, K < %.6g: %d" % (thr, ((allK < thr) & m).sum()))
t = z["all_t"]
print("t of flagged rows: ", np.histogram(t, bins=[0, .05, .1, .15, .2, .25, .3, .35, .4, .45, .5])[0])
