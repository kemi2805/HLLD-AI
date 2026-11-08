import numpy as np
import wavecheck as WC
G = 5.0 / 3.0
z = np.load("admiss_flagged_1103.npz")
m = z["all_lax"] & ~z["all_entropy"] & ~z["all_cross"]
print("lax-only rows:", m.sum(), "of", m.size, " by source:", np.bincount(z["all_src"][m]))
a = WC.admissibility(z["all_U_L"][m], z["all_U_R"][m], z["all_zones"][m], z["all_speeds"][m], G)
for w in ("LF", "LS", "RS", "RF"):
    sh = a[w + "_solver_kind"] == 1
    M = a[w + "_lax_margins"]
    for j, t in enumerate(a[w + "_lax_tags"]):
        k = sh & (M[j] < -1e-4)
        if k.any():
            print("  %s %-18s fails on %4d  (median margin %.2e)  their K ratio p50 %.6f  |Bt| ratio p50 %.3f  |Bt behind|/|Bn| p50 %.3f" % (
                w, t, k.sum(), np.median(M[j][k]), np.median(a[w + "_K"][k]), np.median(a[w + "_dBt"][k]),
                np.median(a["Bt"][k, 3 if w == "LS" else 4] / np.abs(a["Bn"][k]))))
