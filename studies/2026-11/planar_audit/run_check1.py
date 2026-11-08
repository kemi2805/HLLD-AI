import numpy as np
import wavecheck as WC
G = 5.0 / 3.0
z = np.load("rows_1103.npz", allow_pickle=True)
np.set_printoptions(linewidth=200, precision=3)
for tag in ("all0", "all1"):
    d = WC.check(z[tag + "_U_L"], z[tag + "_U_R"], z[tag + "_zones"], z[tag + "_speeds"], G)
    print("==", tag, "n", d["Bn"].size)
    for w in ("LF", "LS", "RS", "RF"):
        k = d[w + "_kind"]
        for kind, nm in ((1, "shock"), (2, "fan")):
            m = k == kind
            if not m.any():
                continue
            rh = d[w + "_rh"][m]
            lx = d[w + "_lax_min"][m]
            print("  %s %-5s n %5d  rh p50 %.1e p99 %.1e max %.1e | lax min<-1e-6: %d (min %.2e) | fan-not-expanding %d"
                  % (w, nm, m.sum(), np.median(rh), np.quantile(rh, .99), rh.max(),
                     (lx < -1e-6).sum(), lx.min(), (~d[w + "_fan_expands"][m]).sum() if kind == 2 else -1))
    print("  CD jump max per comp", d["CD_jump"].max(axis=1))
    print("  rotation strength >1e-8: LA %d RA %d" % ((d["LA_strength"] > 1e-8).sum(), (d["RA_strength"] > 1e-8).sum()))
