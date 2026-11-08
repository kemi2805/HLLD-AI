"""For the 128^2 ledger's failing lanes: what did each planar attempt (method b,
4 flips x 6 seeds; method d, the wide window) end in, by loss type?"""
import glob
import numpy as np
D = "/Users/miler/Codes/claude_scratch/r2d_retrain/fixes/ledger128/"
lt = np.load("/Users/miler/Codes/claude_scratch/r2d_retrain/anatomy/data/ledger/lanes_typed.npz")
def gather(kind):
    out = {}
    for f in sorted(glob.glob(D + "shard_%s_*.npz" % kind)):
        z = np.load(f, allow_pickle=True)
        idx = z["idx"]
        for k in z.files:
            if k in ("idx", "methods", "wall"):
                continue
            out.setdefault(k, {})
            for j, i in enumerate(idx):
                out[k][int(i)] = z[k][j]
    return out
el, im = gather("el"), gather("im")
print("keys im:", sorted(im.keys())[:30])
fail = lt["fail"]; typ = lt["typ"]
print("types:", {t: int(((typ == t) & fail).sum()) for t in np.unique(typ)})
RN = {0: "ok", 1: "notconv", 2: "notplanar", 3: "fullres", 4: "ray", 5: "unphys", 6: "selfcross", -1: "-"}
for meth, src in (("b", el), ("d", im)):
    key = meth + "_t_why"
    if key not in src:
        print("no", key); continue
    print("== method", meth)
    for t in np.unique(typ[fail]):
        lanes = np.flatnonzero(fail & (typ == t))
        cnt = {}
        anycross = 0; anyconv_raw = 0
        for i in lanes:
            w = src[key].get(int(i))
            if w is None:
                continue
            for fi in range(4):
                for c in w[fi]:
                    cnt[(fi, int(c))] = cnt.get((fi, int(c)), 0) + 1
            anycross += int((w == 6).any())
            anyconv_raw += int(np.isin(w, (0, 3, 4, 5, 6)).any())
        print("  %-18s n=%d  any self-cross %d, any converged (raw) %d" % (t, len(lanes), anycross, anyconv_raw))
        for fi, fn in enumerate(("--", "L-", "-R", "LR")):
            row = {RN[c]: cnt.get((fi, c), 0) for c in (0, 1, 3, 4, 5, 6)}
            print("     flip %s" % fn, row)
