"""The 'order' fan failures and the out-of-order answers of the run sample:
by how much, and between which waves."""
import glob, numpy as np
import sol_profiles as SP
import rmhd_ideal as RI
D = "/Users/miler/Codes/claude_scratch/r2d_retrain/fixes/verify"
fs = sorted(glob.glob(D + "/run/chunk_*_verify.npz"))
ins = [dict(np.load(f.replace("_verify", ""), allow_pickle=True)) for f in fs]
outs = [dict(np.load(f, allow_pickle=True)) for f in fs]
cat = lambda L: {k: np.concatenate([d[k] for d in L]) for k in L[0] if np.ndim(L[0][k]) > 0 and k != "failwhy"}
a, r = cat(ins), cat(outs)
g = 5.0 / 3.0
k = r["kind"]
Z, S, Bn = a["zones"], a["speeds"], a["U_L"][:, 7]
N = len(Bn)
lo = np.full((N, 7), np.nan); hi = np.full((N, 7), np.nan)
for w in range(7):
    aZ, bZ = (w, w + 1) if w < 3 else (w + 1, w)
    A = [Z[:, aZ, c] for c in range(7)]; B = [Z[:, bZ, c] for c in range(7)]
    if w in (0, 2, 4, 6):
        sw = ("LF", None, "LS", None, "RS", None, "RF")[w]
        la, lb = SP.XI(A, sw, Bn), SP.XI(B, sw, Bn)
        fan = k[:, w] != "disc"
        lo[:, w] = np.where(fan, np.minimum(la, lb), S[:, w]); hi[:, w] = np.where(fan, np.maximum(la, lb), S[:, w])
        if w in (2, 4):
            m = (k[:, w] == "FAIL") & (r["fan"][:, w] <= 1e-6) & (r["G"][:, w] <= 1e-6)
            exp = (lb - la) if w < 3 else (la - lb)
            print("%s non-expanding fans: %d; lambda_behind - lambda_ahead (in the travel direction) p50 %.1e min %.1e; strength |dPt/Pt| p50 %.1e" % (
                sw, m.sum(), np.median(exp[m]) if m.any() else 0, exp[m].min() if m.any() else 0,
                np.median(np.abs(B[1][m] / A[1][m] - 1)) if m.any() else 0))
    else:
        V = S[:, w] if w != 3 else 0.5 * (Z[:, 3, 2] + Z[:, 4, 2])
        lo[:, w] = hi[:, w] = V
nz = k != "zero"
bad = ~r["order_ok"]
print("out of order: %d answers" % bad.sum())
pairs = {}
for i in np.flatnonzero(bad):
    last_w, last_hi = None, -np.inf
    for w in range(7):
        if not nz[i, w]:
            continue
        if lo[i, w] < last_hi - 1e-9:
            key = "%s>%s" % (("LF", "LA", "LS", "CD", "RS", "RA", "RF")[last_w], ("LF", "LA", "LS", "CD", "RS", "RA", "RF")[w])
            pairs.setdefault(key, []).append(last_hi - lo[i, w])
        last_w, last_hi = w, hi[i, w]
for p_, v in sorted(pairs.items(), key=lambda t: -len(t[1])):
    v = np.array(v)
    print("  %-7s overlaps %5d: by p50 %.1e p90 %.1e max %.1e" % (p_, len(v), np.median(v), np.percentile(v, 90), v.max()))
