"""Failing slow waves of production's answers, one by one: shock or fan by
production's own rule (|B_t| falls -> shock), the jump-condition residual per
conserved component, the fan re-integrations, and the wave's strength."""
import glob, numpy as np
import rmhd_ideal as RI
import sol_profiles as SP
from rmhd.batched.jump import rh_resid as rh_prod
import verify_rmhd as V
D = "/Users/miler/Codes/claude_scratch/r2d_retrain/fixes/verify"
ins = [dict(np.load(f.replace("_verify", ""), allow_pickle=True)) for f in sorted(glob.glob(D + "/run/chunk_00[0-3]_verify.npz"))]
outs = [dict(np.load(f, allow_pickle=True)) for f in sorted(glob.glob(D + "/run/chunk_00[0-3]_verify.npz"))]
cat = lambda L: {k: np.concatenate([d[k] for d in L]) for k in L[0] if np.ndim(L[0][k]) > 0}
a, r = cat(ins), cat(outs)
g = float(np.load(glob.glob(D + "/run/chunk_000.npz")[0])["gamma"])
k = r["kind"]
NAMES = ("rho", "Pt", "vx", "vy", "vz", "By", "Bz")
CN = ("D", "Sx", "Sy", "Sz", "tau", "By", "Bz")
rng = np.random.default_rng(1)
for w, sw in ((2, "LS"), (4, "RS")):
    m = np.flatnonzero(k[:, w] == "FAIL")
    pick = rng.choice(m, 4, replace=False)
    for i in pick:
        aZ, bZ = (w, w + 1) if w < 3 else (w + 1, w)
        A = [np.array([a["zones"][i, aZ, c]]) for c in range(7)]
        B = [np.array([a["zones"][i, bZ, c]]) for c in range(7)]
        Bn = np.array([a["U_L"][i, 7]])
        V_ = a["speeds"][i, w]
        shock = abs(B[5][0]) < abs(A[5][0])
        Ua, Fa = RI.cons_flux(A, Bn, g); Ub, Fb = RI.cons_flux(B, Bn, g)
        comp = np.abs((Fb - Fa) - V_ * (Ub - Ua))[:, 0]
        print("%s lane %d (%s by |B_t|): V %.7f  dPt/Pt %.1e  Bt %+.5f -> %+.5f  rho %.6f -> %.6f" % (
            sw, i, "shock" if shock else "fan", V_, B[1][0] / A[1][0] - 1, A[5][0], B[5][0], A[0][0], B[0][0]))
        print("    my rh %.1e  production's rh_resid %.1e   per component |dF - V dU|: %s" % (
            r["rh"][i, w], rh_prod(A, B, np.array([V_]), Bn, g)[0], " ".join("%s %.0e" % (c, x) for c, x in zip(CN, comp))))
        if not shock:
            yp, _, _ = SP.FAN_P(B[1], A, sw, Bn); ys, _, _ = SP.S4["fan_signed"](B[5], A, sw, Bn)
            yn, _, _ = SP.FAN_N(np.abs(B[5]), A, sw, Bn)
            for nm, y in (("pressure", yp), ("signed Bt", ys), ("|Bt| (production's)", yn)):
                print("    fan in %-20s end - stored: %s" % (nm, " ".join("%s %+.1e" % (c, y[j][0] - B[j][0]) for j, c in enumerate(NAMES))))
