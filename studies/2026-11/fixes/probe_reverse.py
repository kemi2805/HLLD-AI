"""planar5's reversal (_reverse: the Alfven discontinuity through pi) from our
fast states, against the rotation point the Hugoniot scan found."""
import sys
import numpy as np
import compound_refused as CR
from rmhd.batched import planar5_b as P5, fullcontact_b as FB
g, pop, ans = CR.g, CR.pop, CR.ans
S5 = CR.S5
W = FB.make_waves(g)
for lane in [int(a) for a in sys.argv[1:]]:
    UL, UR = pop["UL"][lane], pop["UR"][lane]
    Bn = np.array([UL[7]])
    L0, R0, alpha, _ = S5["to_planar"](CR.one(UL[:7]), CR.one(UR[:7]), Bn)
    Z, S = ans[lane]
    for side, P, X0, fsw in (("L", Z[1][1], L0, "LF"), ("R", Z[6][1], R0, "RF")):
        A, _, _ = W["fast_wave"](np.array([P]), X0, fsw, Bn)
        Vs = W["alfven_speed"](A, Bn, side)
        out, err = W["alfven_checked"](Vs, A, W["seed_rot"](A, np.pi), Bn)
        rr = CR.RI.rh_residual([c for c in A], [c for c in out], Vs, Bn, g)
        print("lane %d %s: ahead Bt %+.5f vt %+.5f -> reversal Bt %+.5f vt %+.5f Bz %+.2e err %s  rh %.1e  drho %+.1e  dS %+.1e  V %.8e" % (
            lane, side, A[5][0], A[3][0], out[5][0], out[3][0], out[6][0], err[0], rr[0],
            out[0][0] / A[0][0] - 1, CR.entropy_ratio(A, out, Bn), Vs[0]))
