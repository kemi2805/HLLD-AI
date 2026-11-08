"""Re-solve the flagged rows with planar5_b (production's rescue, default seed
then the ladder's rungs) in the frame they were stored in, and evaluate the
seven-wave residual at the stored unknowns in the raw and the planar frame."""
import numpy as np
from rmhd.batched import planar5_b as P5
from rmhd.batched import fullcontact_b as FB
G = 5.0 / 3.0
z = np.load("rows_1103.npz", allow_pickle=True)
S = P5.make_solver(G)
W = FB.make_waves(G)
UL, UR, Z, Sp = z["bad_U_L"], z["bad_U_R"], z["bad_zones"], z["bad_speeds"]
res = z["meta_res"]
sel = np.flatnonzero((z["meta_src"] >= 0) & (res > 0.3))           # production rows, main group
L = [np.ascontiguousarray(UL[sel, j]) for j in range(7)]
R = [np.ascontiguousarray(UR[sel, j]) for j in range(7)]
Bn = np.ascontiguousarray(UL[sel, 7])
Lp, Rp, alpha, presid = P5.to_planar(L, R, Bn)
print("alpha (planar frame rotation)", alpha, "planar resid max", presid.max())
wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
ang = lambda Rz: np.arctan2(Rz[:, 6], Rz[:, 5])
Zs = Z[sel]
u6 = [np.log(Zs[:, 1, 1]), np.log(np.hypot(Zs[:, 3, 5], Zs[:, 3, 6])), ang(Zs[:, 3]), np.log(Zs[:, 5, 1]),
      wrap(ang(Zs[:, 2]) - ang(Zs[:, 1])), wrap(ang(Zs[:, 6]) - ang(Zs[:, 5]))]
f_raw, zr, VL, VR, er = W["fullfuncv6"](L, R, u6, Bn)
print("raw-frame residual at stored unk6:", np.max(np.abs(f_raw), axis=0))
print("  raw-frame rebuilt R4 |Bt|, P vs stored:", np.c_[np.hypot(zr[2][5], zr[2][6]), zr[2][1], np.hypot(Zs[:, 3, 5], Zs[:, 3, 6]), Zs[:, 3, 1]][:4])
print("  raw-frame LS speed rebuilt vs stored:", np.c_[VL[2], Sp[sel, 2]][:6])
# planar unknowns from the stored zones, planar frame
unk3 = [u6[0], np.hypot(Zs[:, 3, 5], Zs[:, 3, 6]) * np.cos(u6[2] + alpha), u6[3]]
print("verify_full (planar frame) at stored unk3:", S["verify_full"](Lp, Rp, unk3, Bn))
# the solve itself, default seed
r = S["solve"](L, R, Bn, accuracy=1e-8, max_iter=40, planar_tol=1e-6, verify_tol=1e-8)
print("planar5 default seed: converged", r["converged"], "raw", r["converged_raw"])
print("   full_resid", r["full_resid"])
d0 = np.max(np.abs(np.array(r["unk"]) - np.array(unk3)), axis=0)
print("   |unk - stored unk3|", d0)
for kind in ("+L", "+R", "-L"):
    from rmhd.batched import planar5_b as P5B
    d = P5B.default_seed(Lp, Rp)
    if kind == "s2p":
        bt = np.sqrt(2 * np.exp(d[0])) * np.sign(d[1])
    else:
        bt = np.abs((Lp if kind[1] == "L" else Rp)[5]) * (1 if kind[0] == "+" else -1)
    rr = S["solve"](L, R, Bn, unk3_init=[d[0].copy(), bt, d[0].copy()], accuracy=1e-8, max_iter=40,
                    planar_tol=1e-6, verify_tol=1e-8)
    dd = np.max(np.abs(np.array(rr["unk"]) - np.array(unk3)), axis=0)
    print("rung", kind, "converged", rr["converged"].astype(int), "same-as-stored", (dd < 1e-6).astype(int))
