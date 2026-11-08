"""For the flagged production rows: the stored answer's xi = 0 flux vs the
answer planar5 gives when the slow-shock window stops at the Alfven speed
(slow_shock_b.MARGIN = 0, numpy kernel) -- and the admissibility of both."""
import os
import sys
import numpy as np
os.environ["RMHD_SLOWSHOCK"] = "numpy"
from rmhd.batched import slow_shock_b as SSB
from rmhd.batched import planar5_b as P5
from rmhd.batched import profile as PR
from rmhd.batched import ray_b as RAY
from rmhd.batched import jump as J
import wavecheck as WC
G = 5.0 / 3.0
margin = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
z = np.load("rows_1103.npz", allow_pickle=True)
xi_fn, fan_p, fan_n = PR.make_sampler(G)


def flux_at0(UL, UR, Z, S):
    n = UL.shape[0]
    sL = [np.ascontiguousarray(UL[:, j]) for j in range(7)]
    sR = [np.ascontiguousarray(UR[:, j]) for j in range(7)]
    Bn = np.ascontiguousarray(UL[:, 7])
    zones6 = [[np.ascontiguousarray(Z[:, k, j]) for j in range(7)] for k in range(1, 7)]
    VsLv = [S[:, 0].copy(), S[:, 1].copy(), S[:, 2].copy()]
    VsRv = [S[:, 6].copy(), S[:, 5].copy(), S[:, 4].copy()]
    st, reg, e = RAY.state_at_xi(sL, sR, zones6, VsLv, VsRv, Bn, G, xi_fn, fan_p, fan_n)
    U, F, bad = J.cons_flux(st, Bn, G)
    return np.stack(F, axis=1), np.stack(st, axis=1), np.asarray(reg), np.asarray(e)


def as_rows(r, UL, UR):
    """planar5 answer -> harvest row layout (zones R1..R8, speeds LF..RF)."""
    n = UL.shape[0]
    A, B, C, D = r["zones"]
    A2, D2 = r["zones_mid"]
    Z = np.empty((n, 8, 7))
    Z[:, 0], Z[:, 7] = UL[:, :7], UR[:, :7]
    for k, s in enumerate((A, A2, B, C, D2, D), start=1):
        Z[:, k] = np.stack(s, axis=1)
    Bn = UL[:, 7]
    W = P5.make_solver(G)["waves"]
    la = W["alfven_speed"]([Z[:, 1, j] for j in range(7)], Bn, "L")
    ra = W["alfven_speed"]([Z[:, 6, j] for j in range(7)], Bn, "R")
    V = r["speeds"]
    S = np.stack([V[0], la, V[1], V[2], V[3], ra, V[4]], axis=1)
    return Z, S


sel = np.flatnonzero(z["meta_src"] >= 0)
UL, UR, Zs, Ss = z["bad_U_L"][sel], z["bad_U_R"][sel], z["bad_zones"][sel], z["bad_speeds"][sel]
F0, st0, reg0, e0 = flux_at0(UL, UR, Zs, Ss)
SSB.MARGIN = margin
S5 = P5.make_solver(G)
L = [np.ascontiguousarray(UL[:, j]) for j in range(7)]
R = [np.ascontiguousarray(UR[:, j]) for j in range(7)]
Bn = np.ascontiguousarray(UL[:, 7])
best = None
for seed in (None, "+L", "+R"):
    if seed is None:
        u0 = None
    else:
        Lp, Rp = P5.to_planar(L, R, Bn)[:2]
        d = P5.default_seed(Lp, Rp)
        bt = np.abs((Lp if seed[1] == "L" else Rp)[5])
        u0 = [d[0].copy(), bt, d[0].copy()]
    r = S5["solve"](L, R, Bn, unk3_init=u0, accuracy=1e-8, max_iter=40, planar_tol=1e-6, verify_tol=1e-8)
    if best is None:
        best = r
        got = r["converged"].copy()
    else:
        new = r["converged"] & ~got
        for key in ("zones", "zones_mid"):
            for zk, zr in zip(best[key], r[key]):
                for j in range(7):
                    zk[j][new] = zr[j][new]
        for k in range(5):
            best["speeds"][k][new] = r["speeds"][k][new]
        got |= new
print("MARGIN", margin, "planar5 converged+verified on", got.sum(), "of", got.size)
Z1, S1 = as_rows(best, UL, UR)
F1, st1, reg1, e1 = flux_at0(UL, UR, Z1, S1)
a0 = WC.admissibility(UL, UR, Zs, Ss, G)
a1 = WC.admissibility(UL, UR, Z1, S1, G)
np.set_printoptions(linewidth=200, precision=4)
scale = np.max(np.abs(F0), axis=1)
for i in range(sel.size):
    def tag(a):
        bad = []
        for w in ("LF", "LS", "RS", "RF"):
            if a[w + "_solver_kind"][i] == 1:
                if a[w + "_K"][i] < 1 - 1e-6:
                    bad.append(w + ":entropy")
                if a[w + "_lax_min"][i] < -1e-5:
                    bad.append(w + ":lax")
        return ",".join(bad) or "admissible"
    df = np.max(np.abs(F1[i] - F0[i])) / scale[i] if got[i] else np.nan
    print("row %8d res %.1e | stored: %-28s | MARGIN=%g answer: %-5s %-28s | |dF|/|F| %.3e  P* %.4f -> %.4f" % (
        z["meta_row"][sel][i], z["meta_res"][sel][i], tag(a0), margin, bool(got[i]), tag(a1) if got[i] else "-",
        df, st0[i, 1], st1[i, 1] if got[i] else np.nan))
np.savez("alt_margin%g.npz" % margin, sel=sel, got=got, Z1=Z1, S1=S1, F0=F0, F1=F1, st0=st0, st1=st1)
