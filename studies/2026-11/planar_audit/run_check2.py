import numpy as np
import wavecheck as WC
G = 5.0 / 3.0
z = np.load("rows_1103.npz", allow_pickle=True)
np.set_printoptions(linewidth=220, precision=4, suppress=True)
d = WC.check(z["bad_U_L"], z["bad_U_R"], z["bad_zones"], z["bad_speeds"], G)
order = np.argsort(-z["meta_res"])
for i in order:
    print("%-15s row %8d src %2d res %.2e Bn %+.3f |Bt| %s" % (z["meta_file"][i], z["meta_row"][i], z["meta_src"][i],
          z["meta_res"][i], d["Bn"][i], np.array2string(d["Bt"][i], precision=3)))
    print("    ", WC.summary_line(d, i))
    for w in ("LF", "LS", "RS", "RF"):
        if d[w + "_kind"][i] == 1:
            print("     %s shock: drho %.3f dP %+.3f |Bt| x%.3f s %+.6f  margins %s %s" % (
                w, d[w + "_drho"][i], d[w + "_dP"][i], d[w + "_dBt"][i], z["bad_speeds"][i][("LF","LA","LS","CD","RS","RA","RF").index(w)],
                np.array2string(d[w + "_lax_margins"][:, i], precision=3), d[w + "_lax_tags"]))
    print("     speeds stored %s" % np.array2string(z["bad_speeds"][i], precision=5))
    print("     eigen (zones) %s" % np.array2string(d["lam"][i], precision=5))
    print("     CD jump %s  rot str LA %.1e RA %.1e" % (np.array2string(d["CD_jump"][:, i], precision=2), d["LA_strength"][i], d["RA_strength"][i]))
