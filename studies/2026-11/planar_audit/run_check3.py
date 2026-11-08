import numpy as np
import wavecheck as WC
G = 5.0 / 3.0
z = np.load("rows_1103.npz", allow_pickle=True)
np.set_printoptions(linewidth=220, precision=4, suppress=True)
d = WC.admissibility(z["bad_U_L"], z["bad_U_R"], z["bad_zones"], z["bad_speeds"], G)
order = np.argsort(-z["meta_res"])
K = ("-", "SHOCK", "fan")
for i in order:
    print("%-15s %8d src %2d res %.2e Bn %+.3f |Bt| %s" % (z["meta_file"][i], z["meta_row"][i], z["meta_src"][i],
          z["meta_res"][i], d["Bn"][i], np.array2string(d["Bt"][i], precision=3)))
    sp = z["bad_speeds"][i]
    for w, col in (("LF", 0), ("LS", 2), ("RS", 4), ("RF", 6)):
        k = d[w + "_solver_kind"][i]
        if k == 0:
            continue
        lm = d[w + "_lax_margins"][:, i] if k == 1 else None
        print("   %s %-5s s %+.7f str %.2e rho x%.4f K x%.6f |Bt| x%.3f RH %.1e %s" % (
            w, K[k], sp[col], d[w + "_strength"][i], d[w + "_drho"][i], d[w + "_K"][i], d[w + "_dBt"][i],
            d[w + "_rh"][i], ("lax " + np.array2string(lm, precision=4) + " " + str(d[w + "_lax_tags"])) if k == 1 else
            "head %+.5f tail %+.5f" % (d[w + "_fan_head"][i], d[w + "_fan_tail"][i])))
    print("   speeds %s  rotLA %.1e rotRA %.1e" % (np.array2string(sp, precision=6), d["LA_strength"][i], d["RA_strength"][i]))
