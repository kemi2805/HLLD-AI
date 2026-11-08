#!/usr/bin/env python
"""Drop the inadmissible and the no-root answers from a training file
(2026-11-04; user: "yes, drop them from the training set only, if they are
still better than HLLD we keep them in the run").

    PYTHONPATH=<rmhd_final> python drop_inadmissible.py IN.npz OUT.npz OUT.json

A row goes if
  entropy  a shock in its answer lowers the entropy: K = p/rho^gamma falls by
           more than 1e-6 (any family; planar_audit/wavecheck.py's
           admissibility(), the same flags as admiss_all.py);
  lax      a slow shock breaks its family's Lax conditions, margin below -1e-4
           (with entropy: the side session's RMHD_SLOWSHOCK_ADMISSIBLE rule);
  noroot   a slow shock is no root of slow_shock_b's own equations: the
           ladder's loose fallback, max(|f_hug|, |f_416|) / (1 + |P_ahead|)
           > 1e-6 at the stored speed and states (roots are <= 1e-8 and
           nothing lies between; planar_audit/classify_tail.py).  The
           seven-wave residual cannot see it (eq. 4.17 alone is zero in the
           plane), so these pass the split's verification.
Every per-row key of IN is kept for the remaining rows, the rest (gamma)
copied.  The report counts rows and distinct problems (the split repeats the
complex problems in training), natural and complex (rotation > 0.5 rad).
Only training files go through this: VAL and TEST keep their faces, so the
networks stay comparable on them.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "planar_audit"))
sys.path.insert(0, HERE)
import wavecheck as WC                       # noqa: E402
from r2d_audit import keys16, rotation       # noqa: E402
from rmhd.batched import slow_shock_b as SSB  # noqa: E402

ENT_TOL, LAX_TOL, ROOT_TOL, ROT = 1e-6, 1e-4, 1e-6, 0.5


def flags(UL, UR, Z, S, gamma, chunk=200_000):
    n = UL.shape[0]
    f = {k: np.zeros(n, bool) for k in ("entropy", "lax", "noroot")}
    hug, e416 = SSB.make_hug(gamma), SSB.make_eq416(gamma)
    for i0 in range(0, n, chunk):
        s = slice(i0, i0 + chunk)
        a = WC.admissibility(UL[s], UR[s], Z[s], S[s], gamma)
        Bn = UL[s, 7]
        for w, ka, kb, col in (("LF", 0, 1, 0), ("LS", 2, 3, 2), ("RS", 5, 4, 4), ("RF", 7, 6, 6)):
            sh = a[w + "_solver_kind"] == 1
            f["entropy"][s] |= sh & (a[w + "_K"] < 1 - ENT_TOL)
            if w in ("LS", "RS"):
                f["lax"][s] |= sh & (a[w + "_lax_min"] < -LAX_TOL)
                i = np.flatnonzero(sh)
                if i.size:
                    Zc = Z[s]
                    ah = [np.ascontiguousarray(Zc[i, ka, j]) for j in range(7)]
                    bh = [np.ascontiguousarray(Zc[i, kb, j]) for j in range(7)]
                    Vs = np.ascontiguousarray(S[s][i, col])
                    with np.errstate(all="ignore"):
                        f1, _ = hug(Vs, bh, ah, Bn[i])
                        f2, _ = e416(Vs, bh, ah, Bn[i])
                    r = np.maximum(np.abs(f1), np.abs(f2)) / (1.0 + np.abs(ah[1]))
                    r = np.where(np.isfinite(r), r, np.inf)
                    f["noroot"][np.arange(n)[s][i[r > ROOT_TOL]]] = True
    f["drop"] = f["entropy"] | f["lax"] | f["noroot"]
    return f


def main():
    src, dst, rep = sys.argv[1:4]
    z = np.load(src)
    d = {k: z[k] for k in z.files}
    n = d["U_L"].shape[0]
    gamma = float(d["gamma"])
    f = flags(d["U_L"], d["U_R"], d["zones"], d["speeds"], gamma)
    keep = ~f["drop"]
    out = {k: (v[keep] if v.ndim and v.shape[0] == n else v) for k, v in d.items()}
    np.savez_compressed(dst, **out)
    _, first, inv = np.unique(keys16(d["U_L"], d["U_R"]), return_index=True, return_inverse=True)
    cx = rotation(d["zones"][first]) > ROT
    r = dict(src=src, dst=dst, gamma=gamma, rows=n, kept=int(keep.sum()),
             rows_dropped={k: int(v.sum()) for k, v in f.items()},
             problems=int(first.size), problems_complex=int(cx.sum()))
    for k, v in f.items():
        pv = np.zeros(first.size, bool)
        pv[inv[v]] = True
        r.setdefault("problems_dropped", {})[k] = dict(natural=int((pv & ~cx).sum()), complex=int((pv & cx).sum()))
    r["rows_dropped_pct"] = 100.0 * r["rows_dropped"]["drop"] / n
    p = r["problems_dropped"]["drop"]
    r["problems_dropped_pct"] = dict(natural=100.0 * p["natural"] / max(1, int((~cx).sum())),
                                     complex=100.0 * p["complex"] / max(1, int(cx.sum())))
    json.dump(r, open(rep, "w"), indent=1)
    print(json.dumps(r))


if __name__ == "__main__":
    main()
