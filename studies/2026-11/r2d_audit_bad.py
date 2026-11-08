#!/usr/bin/env python
"""The rows r2d_audit.py flagged (2026-11-03): stored "solutions" whose
seven-wave residual is above 1e-6 (30 problems in it3/train3.npz, 7 in
it1/eval.npz, worst 5.3) and rows the wave-order gate's rule rejects (2
problems).  Per row: where it came from (source / attempts / t, matched in
it1/all.npz), which residual component fails, the tangential-field angle and
magnitude of every zone R1..R8 (a reversal ACROSS A SLOW WAVE is outside the
seven-wave family: its angle may only change across the rotations R2->R3 and
R6->R7), and whether the seven-wave Newton started AT the stored solution
finds a root nearby.

    python r2d_audit_bad.py RETRAIN_DIR OUT.npz
"""
import sys

import numpy as np

from rmhd.batched import fullcontact_b as FB
from rmhd.batched import ml_b as MB

sys.path.insert(0, "/mnt/rafast/miler/scratch/r2d_retrain")
from r2d_audit import keys16, residual, gate_crossed, wrap      # noqa: E402

D, OUT = sys.argv[1], sys.argv[2]
G = 5.0 / 3.0
W, S = FB.make_waves(G), FB.make_solver(G)
za = np.load(D + "/it1/all.npz")
ka = keys16(np.asarray(za["U_L"], float), np.asarray(za["U_R"], float))
oa = np.argsort(ka)
ka_s = ka[oa]

rows = []
for f in ("it1/eval.npz", "it3/train3.npz"):
    z = np.load(D + "/" + f)
    UL, UR = np.asarray(z["U_L"], float), np.asarray(z["U_R"], float)
    Z, Sp = np.asarray(z["zones"], float), np.asarray(z["speeds"], float)
    k = keys16(UL, UR)
    _, u = np.unique(k, return_index=True)
    res, _ = residual(UL[u], UR[u], Z[u], W, 100_000)
    cx = gate_crossed(Z[u], Sp[u], 1e-6)
    flagged = np.flatnonzero((res > 1e-6) | cx)
    print(f, "unique", u.size, "flagged", flagged.size, flush=True)
    for pos in flagged:
        i = u[pos]
        j = np.searchsorted(ka_s, k[i])
        hit = j < ka_s.size and ka_s[j] == k[i]
        src = int(za["source"][oa[j]]) if hit else -1
        att = int(za["attempts"][oa[j]]) if hit else -1
        t = float(za["t"][oa[j]]) if hit else np.nan
        ang = np.arctan2(Z[i, :, 6], Z[i, :, 5])
        bt = np.hypot(Z[i, :, 5], Z[i, :, 6])
        d = wrap(np.diff(ang))                   # R1->R2 ... R7->R8
        sl = slice(i, i + 1)
        R2, R3, R4, R6, R7 = Z[sl, 1], Z[sl, 2], Z[sl, 3], Z[sl, 5], Z[sl, 6]
        a = lambda R: np.arctan2(R[:, 6], R[:, 5])
        u6 = [np.log(R2[:, 1]), np.log(np.maximum(np.hypot(R4[:, 5], R4[:, 6]), 1e-30)), a(R4),
              np.log(R6[:, 1]), wrap(a(R3) - a(R2)), wrap(a(R7) - a(R6))]
        left = [UL[sl, c].copy() for c in range(7)]
        right = [UR[sl, c].copy() for c in range(7)]
        with np.errstate(all="ignore"):
            fv = W["fullfuncv6"](left, right, u6, UL[sl, 7].copy())[0][:, 0]
            nr = S["fullcontact6"](left, right, u6, UL[sl, 7].copy(), accuracy=1e-8, max_iter=40)
        conv = bool(np.asarray(nr["converged"])[0])
        moved = float(np.max(np.abs(np.array([np.asarray(x)[0] for x in nr["unk"]])
                                    - np.array([x[0] for x in u6])))) if conv else np.nan
        rows.append(dict(file=f, row=int(i), src=src, att=att, t=t, res=float(res[pos]),
                         crossed=bool(gate_crossed(Z[sl], Sp[sl], 1e-6)[0]), fcomp=int(np.argmax(np.abs(fv))),
                         dang=d, bt=bt, speeds=Sp[i], newton_from_stored=conv, newton_moved=moved))
        r = rows[-1]
        print("%-15s row %8d src %2d att %2d t %.4f  res %.2e crossed %d  worst f[%d]  "
              "dangle R1..R8 %s  |Bt| %s  Newton from stored: %s moved %.2e"
              % (f, i, src, att, t, r["res"], r["crossed"], r["fcomp"],
                 np.array2string(d, precision=2, suppress_small=True),
                 np.array2string(bt, precision=3), conv, moved), flush=True)
np.savez(OUT, rows=np.array(rows, dtype=object))
print("wrote", OUT)
