"""One leaked lane of the 4Q filter run (flagged_adm64/r2d_64_exact_admiss.npz)
as a test fixture: a loose LS shock whose filter-off root's entropy falls."""
import sys
import numpy as np
from rmhd.batched import slow_shock_b as SB
np.seterr(all="ignore")
z = np.load(sys.argv[1])
g = float(z["gamma"])
hug, e416 = SB.make_hug(g), SB.make_eq416(g)
Zs, S, Bn_all = z["zones"], z["speeds"], z["U_L"][:, 7]
def ent(s, Bn):
    rho, P, vx, vy, vz, By, Bz = s
    v2 = vx * vx + vy * vy + vz * vz
    eta = Bn * vx + By * vy + Bz * vz
    b2 = (Bn * Bn + By * By + Bz * Bz) * (1.0 - v2) + eta * eta
    return (P - 0.5 * b2) / rho ** g
w, ka, kb, col = "LS", 2, 3, 2
i = np.flatnonzero(z[w + "_solver_kind"] == 1)
shown = 0
for n in i:
    ah = [np.array([Zs[n, ka, j]]) for j in range(7)]
    bh = [np.array([Zs[n, kb, j]]) for j in range(7)]
    Vst = np.array([S[n, col]]); Bn = np.array([Bn_all[n]])
    r = max(abs(hug(Vst, bh, ah, Bn)[0][0]), abs(e416(Vst, bh, ah, Bn)[0][0])) / (1 + abs(ah[1][0]))
    if not r > 1e-6:
        continue
    out = {}
    for flt in (False, True):
        SB.SLOW_ADMISSIBLE = flt
        for rt in (False, True):
            SB.SLOW_RETRY = rt
            V, b, e = SB.make_solver(g)(bh[5], bh[6], ah, w, Bn, accuracy=1e-10)
            out[(flt, rt)] = (bool(e[0]), float(V[0]), float((ent(b, Bn) / ent(ah, Bn))[0]))
    if out[(False, False)][0] or out[(False, False)][2] > 1 - 1e-5:
        continue
    print("row", n, "shard", int(z["shard"][n]), "src", int(z["source"][n]), "r_stored %.2e" % r)
    print("R_AHEAD =", [float(c[0]) for c in ah])
    print("TARGET =", (float(bh[5][0]), float(bh[6][0])), "BN =", float(Bn[0]), "stored Vs", float(Vst[0]))
    for k, v in out.items():
        print("  filter %s retry %s -> err %s Vs %.12f K %.8f" % (k[0], k[1], *v))
    shown += 1
    if shown >= 3:
        break
