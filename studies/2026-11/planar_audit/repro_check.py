"""Replay A's flux vs the flux of the STORED (run) answer, from its zones."""
import numpy as np
from alt_solution_lib import flux_at0
z = np.load("admiss_flagged_1103.npz")
A = np.load("replay/A_report/faces.npz")
keys = list(A["F_keys"])
print("flux keys", keys)
def k16(UL, UR):
    a = np.ascontiguousarray(np.concatenate([UL, UR], axis=1))
    return a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1]))).ravel()
ks = k16(z["all_U_L"], z["all_U_R"])
_, u = np.unique(ks, return_index=True)
ka = k16(A["U_L"], A["U_R"])
pos = {k.tobytes(): i for i, k in enumerate(ka)}
ia = np.array([pos[k.tobytes()] for k in ks[u]])
F0, st0, reg0, e0 = flux_at0(z["all_U_L"][u], z["all_U_R"][u], z["all_zones"][u], z["all_speeds"][u])
# cons_flux order: D, Sx, Sy, Sz, E, By, Bz ; tau flux = E - D
Fs = {"D": F0[:, 0], "Sx": F0[:, 1], "Sy": F0[:, 2], "Sz": F0[:, 3], "tau": F0[:, 4] - F0[:, 0], "By": F0[:, 5], "Bz": F0[:, 6]}
FA = A["F"][ia]
sc = np.max(np.abs(FA), axis=1)
d = np.zeros(len(ia))
for k, v in Fs.items():
    if k in keys:
        d = np.maximum(d, np.abs(FA[:, keys.index(k)] - v) / sc)
ex = A["exact"][ia]
print("stored-answer flux vs replay A, exact faces %d: p50 %.1e p99 %.1e max %.1e; >1e-8: %d" % (
    ex.sum(), np.median(d[ex]), np.quantile(d[ex], .99), d[ex].max(), (d[ex] > 1e-8).sum()))
