#!/usr/bin/env python
"""Final classification of every verified answer (one category each):

  valid       a weak solution of ideal RMHD (every jump satisfies the jump
              conditions, every fan is a genuine rarefaction, waves in order)
              AND entropy-admissible (no shock or rotation lowers the entropy)
  entropy     a weak solution, but a jump lowers the entropy
  compress    a "fan" whose characteristic speed does not spread (a
              compression, not a rarefaction)
  overlap     waves out of order (a wave runs past the next one)
  other       a jump that fails its jump conditions, a fan off its integral curve
"""
import glob
import numpy as np
import sol_profiles as SP

D = "/Users/miler/Codes/claude_scratch/r2d_retrain/fixes/verify"
pop = np.load("/Users/miler/Codes/claude_scratch/r2d_retrain/anatomy/data/ledger/population.npz", allow_pickle=True)
key = lambda A, B: [x.tobytes() + y.tobytes() for x, y in zip(np.ascontiguousarray(A), np.ascontiguousarray(B))]
grp = {k_: int(g_) for k_, g_ in zip(key(pop["UL"], pop["UR"]), pop["group"])}
NAMES = ("LF", "LA", "LS", "CD", "RS", "RA", "RF")
CATS = ("valid", "entropy", "compress", "overlap", "other")


def classify(a, r):
    n = len(r["weak"])
    cat = np.full(n, "valid", dtype="U8")
    cat[r["weak"] & ~r["admissible"]] = "entropy"
    fw = r["failwhy"] if "failwhy" in r and len(r["failwhy"]) else np.zeros((0, 3), object)
    comp = np.zeros(n, bool)
    other = np.zeros(n, bool)
    for l_, w, why in fw:
        if why == "order":
            comp[int(l_)] = True
        else:
            other[int(l_)] = True
    k = r["kind"]
    fail_disc = np.zeros(n, bool)
    for w in (1, 3, 5):
        fail_disc |= k[:, w] == "FAIL"
    other |= fail_disc
    cat[~r["order_ok"]] = "overlap"
    cat[comp] = "compress"
    cat[other & ~comp] = "other"
    return cat


def overlap_pairs(a, r, idx):
    Z, S, Bn = a["zones"], a["speeds"], a["U_L"][:, 7]
    k = r["kind"]
    out = {}
    for i in idx:
        lo, hi = [], []
        for w in range(7):
            if k[i, w] == "zero":
                lo.append(np.nan); hi.append(np.nan); continue
            aZ, bZ = (w, w + 1) if w < 3 else (w + 1, w)
            if w in (0, 2, 4, 6) and k[i, w] != "disc":
                sw = NAMES[w]
                A = [np.array([Z[i, aZ, c]]) for c in range(7)]; B = [np.array([Z[i, bZ, c]]) for c in range(7)]
                la, lb = SP.XI(A, sw, Bn[i:i + 1])[0], SP.XI(B, sw, Bn[i:i + 1])[0]
                lo.append(min(la, lb)); hi.append(max(la, lb))
            else:
                V = S[i, w] if w != 3 else 0.5 * (Z[i, 3, 2] + Z[i, 4, 2])
                lo.append(V); hi.append(V)
        last, lastw = -np.inf, None
        worst = 0.0
        for w in range(7):
            if np.isnan(lo[w]):
                continue
            if lo[w] < last - 1e-9:
                out.setdefault("%s>%s" % (NAMES[lastw], NAMES[w]), []).append(last - lo[w])
                worst = max(worst, last - lo[w])
            last, lastw = hi[w], w
        WORST[i] = worst
    return out


WORST = {}
for which in ("run", "ledger"):
    rows = []
    for f in sorted(glob.glob("%s/%s/chunk_*_verify.npz" % (D, which))):
        r = dict(np.load(f, allow_pickle=True))
        a = dict(np.load(f.replace("_verify.npz", ".npz"), allow_pickle=True))
        cat = classify(a, r)
        src = a["source"] if which == "ledger" else np.where(a["source"] == 1, 2, 1)
        g = (np.array([grp.get(k_, -1) for k_ in key(a["U_L"], a["U_R"])]) if which == "ledger"
             else np.zeros(len(cat), int))
        WORST = {}
        ov = overlap_pairs(a, r, np.flatnonzero(cat == "overlap"))
        # production's own wave-order tolerance is 1e-6 (exact_flux._WAVE_ORDER_TOL):
        # waves that coincide to that are one speed, not a crossing
        for i_, w_ in WORST.items():
            if w_ <= 1e-6 + 1e-12:
                cat[i_] = "valid" if (r["weak"][i_] or not r["order_ok"][i_]) and r["admissible"][i_] | (~r["order_ok"][i_] & ~((r["kind"][i_] == "disc") & (np.arange(7) != 3) & (r["dS"][i_] < -1e-9)).any()) else "entropy"
        rows.append((cat, src, g, ov))
    cat = np.concatenate([x[0] for x in rows]); src = np.concatenate([x[1] for x in rows]); g = np.concatenate([x[2] for x in rows])
    ov = {}
    for x in rows:
        for p_, v in x[3].items():
            ov.setdefault(p_, []).extend(v)
    print("\n######## %s: %d answers" % (which, len(cat)))
    groups = ((("128^2 run, production", np.ones(len(cat), bool)),) if which == "run" else
              (("ledger controls, production", g == 0), ("ledger, recovered by the fixes", g == 1),
               ("ledger, unmatched", g == -1)))
    for gname, gm in groups:
        for s, sn in ((1, "seven-wave"), (2, "planar 5-wave"), (3, "four-unknown")):
            m = gm & (src == s)
            if not m.any():
                continue
            c = {c_: int((cat[m] == c_).sum()) for c_ in CATS}
            print("  %-32s %-14s n=%6d  " % (gname, sn, m.sum()) +
                  "  ".join("%s %d (%.2f%%)" % (c_, c[c_], 100 * c[c_] / m.sum()) for c_ in CATS))
    for p_, v in sorted(ov.items(), key=lambda t: -len(t[1])):
        v = np.array(v)
        print("    overlap %-7s %5d: by p50 %.1e  max %.1e" % (p_, len(v), np.median(v), v.max()))
