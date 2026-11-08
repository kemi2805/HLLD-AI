"""Is the verify_full verdict at the stored answer batch-dependent, or
input-chaotic?  Same lane, same inputs (to_planar), alone vs in a batch."""
import numpy as np
from rmhd.batched import planar5_b as P5
G = 5.0 / 3.0
z = np.load("rows_1103.npz", allow_pickle=True)
S = P5.make_solver(G)
sel = np.flatnonzero((z["meta_src"] >= 0) & (z["meta_res"] > 0.3))
UL, UR, Zs = z["bad_U_L"][sel], z["bad_U_R"][sel], z["bad_zones"][sel]
L = [np.ascontiguousarray(UL[:, j]) for j in range(7)]
R = [np.ascontiguousarray(UR[:, j]) for j in range(7)]
Bn = np.ascontiguousarray(UL[:, 7])
Lp, Rp, alpha, _ = P5.to_planar(L, R, Bn)
ang4 = np.arctan2(Zs[:, 3, 6], Zs[:, 3, 5])
unk3 = [np.log(Zs[:, 1, 1]), np.hypot(Zs[:, 3, 5], Zs[:, 3, 6]) * np.cos(ang4 + alpha), np.log(Zs[:, 5, 1])]
batch = S["verify_full"](Lp, Rp, unk3, Bn)
alone = np.array([S["verify_full"]([c[k:k + 1] for c in Lp], [c[k:k + 1] for c in Rp],
                                   [u[k:k + 1] for u in unk3], Bn[k:k + 1])[0] for k in range(sel.size)])
np.set_printoptions(precision=2, linewidth=200)
print("batch:", batch)
print("alone:", alone)
print("rows differing:", np.flatnonzero(np.abs(np.log10(batch + 1e-30) - np.log10(alone + 1e-30)) > 1))
