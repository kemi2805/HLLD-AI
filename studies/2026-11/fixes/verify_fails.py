import glob, numpy as np
import verify_rmhd as V
D = "/Users/miler/Codes/claude_scratch/r2d_retrain/fixes/verify"
outs = [dict(np.load(f, allow_pickle=True)) for f in sorted(glob.glob(D + "/run/chunk_*_verify.npz"))]
r = {k: np.concatenate([d[k] for d in outs]) for k in outs[0] if np.ndim(outs[0][k]) > 0}
k = r["kind"]
for w, nm in enumerate(V.NAMES):
    m = k[:, w] == "FAIL"
    if not m.any():
        continue
    rh, fan, G = r["rh"][m, w], r["fan"][m, w], r["G"][m, w]
    q = lambda x: np.array2string(np.nanpercentile(x, [10, 50, 90, 99]), precision=1)
    print("%s FAIL %d: rh p10/50/90/99 %s | fan mismatch %s | identity %s | fan nan %d" % (
        nm, m.sum(), q(rh), q(fan) if np.isfinite(fan).any() else "-", q(G) if np.isfinite(G).any() else "-", np.isnan(fan).sum()))
    if np.isfinite(fan).any():
        print("     fan mismatch <= 1e-6: %d, <= 1e-4: %d, inf: %d;  identity <= 1e-6: %d, <= 1e-4: %d" % (
            (fan <= 1e-6).sum(), (fan <= 1e-4).sum(), np.isinf(fan).sum(), (G <= 1e-6).sum(), (G <= 1e-4).sum()))
    print("     rh <= 1e-6: %d, <= 1e-5: %d, <= 1e-4: %d, > 1e-3: %d" % ((rh <= 1e-6).sum(), (rh <= 1e-5).sum(), (rh <= 1e-4).sum(), (rh > 1e-3).sum()))
