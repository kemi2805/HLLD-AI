"""fullfuncv6 at the stored unknowns of train3 row 486645, stage by stage, in
the raw frame (as the audit evaluates it) and the planar frame (as
planar5_b.verify_full does)."""
import numpy as np
from rmhd.batched import fullcontact_b as FB
G = 5.0 / 3.0
W = FB.make_waves(G)
z = np.load("rows_1103.npz", allow_pickle=True)
i = int(np.flatnonzero(z["meta_row"] == 486645)[0])
Z, UL, UR = z["bad_zones"][i], z["bad_U_L"][i], z["bad_U_R"][i]
Bn = np.array([UL[7]])
wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
ang = lambda s: np.arctan2(s[6], s[5])
flip = lambda s: np.array([s[0], s[1], s[2], -s[3], -s[4], -s[5], -s[6]])
np.set_printoptions(precision=6, linewidth=200)
for label, f in (("raw", lambda s: np.asarray(s, float)), ("planar", flip)):
    Zf = np.array([f(Z[k]) for k in range(8)])
    L = [np.array([c]) for c in f(UL[:7])]
    R = [np.array([c]) for c in f(UR[:7])]
    u6 = [np.log(Zf[1, 1]), np.log(np.hypot(Zf[3, 5], Zf[3, 6])), ang(Zf[3]), np.log(Zf[5, 1]),
          wrap(ang(Zf[2]) - ang(Zf[1])), wrap(ang(Zf[6]) - ang(Zf[5]))]
    u6 = [np.array([x]) for x in u6]
    print("== %s frame: psi_CD %.17g  phi_L %.3g  phi_R %.3g" % (label, u6[2][0], u6[4][0], u6[5][0]))
    R2, V1, e1 = W["fast_wave"](np.exp(u6[0]), L, "LF", Bn)
    print("  R2 err %s Vs %+.7f  |R2 - stored| %.2e" % (e1[0], V1[0], np.max(np.abs(np.array(R2).ravel() - Zf[1]))))
    la = W["alfven_speed"](R2, Bn, "L")
    R3, e2 = W["alfven_checked"](la, R2, W["seed_rot"](R2, u6[4]), Bn)
    print("  R3 err %s  |R3 - stored| %.2e  |R3 - R2| %.2e" % (e2[0], np.max(np.abs(np.array(R3).ravel() - Zf[2])),
                                                           np.max(np.abs(np.array(R3).ravel() - np.array(R2).ravel()))))
    Bt = np.exp(u6[1])
    R4, V3, sl, e3 = W["slow_wave6"](Bt * np.cos(u6[2]), Bt * np.sin(u6[2]), R3, "LS", Bn)
    print("  R4 err %s Vs %+.7f slack %.1e |R4 - stored| %.2e  rho4 %.5f (stored %.5f)" % (
        e3[0], V3[0], sl[0], np.max(np.abs(np.array(R4).ravel() - Zf[3])), R4[0][0], Zf[3, 0]))
    f6 = W["fullfuncv6"](L, R, u6, Bn)[0][:, 0]
    print("  fullfuncv6 residual", f6)
