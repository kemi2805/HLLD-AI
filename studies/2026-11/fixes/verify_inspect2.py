"""Slow fans that fail only the weak-solution identity: is the fan genuinely
not a simple wave, or is the identity ill-conditioned (a fan narrow in xi)?
For each: identity value, fan width in xi, field/pressure change, whether xi
is monotone along the samples, and the identity recomputed with 4x samples."""
import glob, numpy as np
import rmhd_ideal as RI
import sol_profiles as SP
D = "/Users/miler/Codes/claude_scratch/r2d_retrain/fixes/verify"
fs = sorted(glob.glob(D + "/run/chunk_00[0-7]_verify.npz"))
ins = [dict(np.load(f.replace("_verify", ""), allow_pickle=True)) for f in fs]
outs = [dict(np.load(f, allow_pickle=True)) for f in fs]
cat = lambda L: {k: np.concatenate([d[k] for d in L]) for k in L[0] if np.ndim(L[0][k]) > 0 and k != "failwhy"}
a, r = cat(ins), cat(outs)
g = 5.0 / 3.0
W, SW = {2: "LS", 4: "RS"}, None


def identity(A, B, sw, Bn, K, signed):
    Ak = [np.repeat(c, K) for c in A]
    tg = np.linspace(A[5 if signed else 1], B[5 if signed else 1], K).T.ravel()
    y, _, e = (SP.S4["fan_signed"] if signed else SP.FAN_P)(tg, Ak, sw, np.repeat(Bn, K))
    xi = SP.XI(y, sw, np.repeat(Bn, K)).reshape(-1, K)
    U, F = RI.cons_flux(y, np.repeat(Bn, K), g)
    U, F = U.reshape(7, -1, K), F.reshape(7, -1, K)
    x0, x1, x2 = xi[:, 0:-2:2], xi[:, 1:-1:2], xi[:, 2::2]
    h0, h1 = x1 - x0, x2 - x1
    with np.errstate(all="ignore"):
        w0 = (h0 + h1) / 6 * (2 - h1 / h0); w1 = (h0 + h1) / 6 * (h0 + h1) ** 2 / (h0 * h1); w2 = (h0 + h1) / 6 * (2 - h0 / h1)
    pan = np.nan_to_num(w0[None] * U[:, :, 0:-2:2] + w1[None] * U[:, :, 1:-1:2] + w2[None] * U[:, :, 2::2])
    intU = np.sum(pan, axis=2)
    Gend = (F[:, :, -1] - xi[None, :, -1] * U[:, :, -1] + intU) - (F[:, :, 0] - xi[None, :, 0] * U[:, :, 0])
    sc = np.abs(F).max(axis=2) + np.abs(xi).max(axis=1)[None] * np.abs(U).max(axis=2)
    sc = sc + 1e-3 * sc.max(axis=0)[None]
    mono = np.all(np.diff(xi, axis=1) * np.sign(xi[:, -1:] - xi[:, :1]) >= -1e-15, axis=1)
    return np.max(np.abs(Gend) / sc, axis=0), xi[:, -1] - xi[:, 0], mono


for w, sw in W.items():
    m = np.flatnonzero((r["kind"][:, w] == "FAIL") & (r["fan"][:, w] <= 1e-6))
    m = m[:12]
    aZ, bZ = (w, w + 1) if w < 3 else (w + 1, w)
    A = [a["zones"][m, aZ, c] for c in range(7)]; B = [a["zones"][m, bZ, c] for c in range(7)]
    Bn = a["U_L"][m, 7]
    for K in (33, 129, 513):
        for signed in (False, True):
            G, dxi, mono = identity(A, B, sw, Bn, K, signed)
            print("%s K=%3d %s: identity %s" % (sw, K, "signed" if signed else "press ", np.array2string(G, precision=0, max_line_width=200)))
    print("   width in xi:", np.array2string(dxi, precision=1, max_line_width=200))
    print("   monotone xi:", mono.astype(int), "  dPt/Pt:", np.array2string(B[1] / A[1] - 1, precision=1, max_line_width=200))
    print("   Bt ahead -> behind:", " ".join("%+.3f>%+.3f" % (x, y) for x, y in zip(A[5], B[5])))
