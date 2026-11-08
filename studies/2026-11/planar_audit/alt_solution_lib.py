import numpy as np
from rmhd.batched import planar5_b as P5
from rmhd.batched import profile as PR
from rmhd.batched import ray_b as RAY
from rmhd.batched import jump as J
G = 5.0 / 3.0
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


