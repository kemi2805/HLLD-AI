"""Smoke: planar5 on 8 flagged production rows with the slow-shock admissibility
filter (RMHD_SLOWSHOCK_ADMISSIBLE=1, rmhd worktree on PYTHONPATH)."""
import numpy as np
import rmhd
from rmhd.batched import slow_shock_b as SSB
from rmhd.batched import planar5_b as P5
import wavecheck as WC
from alt_solution_lib import as_rows, flux_at0
print("rmhd from", rmhd.__file__, "SLOW_ADMISSIBLE", SSB.SLOW_ADMISSIBLE)
G = 5.0 / 3.0
z = np.load("rows_1103.npz", allow_pickle=True)
alt = np.load("alt_margin0.npz")
sel = alt["sel"][:8]
UL, UR, Zs, Ss = z["bad_U_L"][sel], z["bad_U_R"][sel], z["bad_zones"][sel], z["bad_speeds"][sel]
L = [np.ascontiguousarray(UL[:, j]) for j in range(7)]
R = [np.ascontiguousarray(UR[:, j]) for j in range(7)]
Bn = np.ascontiguousarray(UL[:, 7])
r = P5.make_solver(G)["solve"](L, R, Bn, accuracy=1e-8, max_iter=40, planar_tol=1e-6, verify_tol=1e-8)
Z1, S1 = as_rows(r, UL, UR)
F0 = alt["F0"][:8]
F1, st1, _, _ = flux_at0(UL, UR, Z1, S1)
a1 = WC.admissibility(UL, UR, Z1, S1, G)
Fm = alt["F1"][:8]
for i in range(sel.size):
    bad = [w for w in ("LS", "RS") if a1[w + "_solver_kind"][i] == 1 and (a1[w + "_K"][i] < 1 - 1e-6 or a1[w + "_lax_min"][i] < -1e-4)]
    sc = np.max(np.abs(F0[i]))
    print("row %8d conv %d inadm %s |F-Fstored|/|F| %.2e  |F-F(MARGIN0)|/|F| %.2e" % (
        z["meta_row"][sel][i], r["converged"][i], bad, np.max(np.abs(F1[i] - F0[i])) / sc,
        np.max(np.abs(F1[i] - Fm[i])) / sc))
