import sys
import numpy as np
import compound_refused as CR
import rd_helpers as RH
g, pop, ans, S5 = CR.g, CR.pop, CR.ans, CR.S5
H = RH.make(g); W = H["W"]
for lane in [int(a) for a in sys.argv[1:]]:
    UL, UR = pop["UL"][lane], pop["UR"][lane]
    Bn = np.array([UL[7]])
    L0, R0, alpha, _ = S5["to_planar"](CR.one(UL[:7]), CR.one(UR[:7]), Bn)
    Z, S = ans[lane]
    for side, P, X0, fsw in (("L", Z[1][1], L0, "LF"), ("R", Z[6][1], R0, "RF")):
        A, _, _ = W["fast_wave"](np.array([P]), X0, fsw, Bn)
        f = H["rd_scan"](A, side, Bn)
        for b, V, Sx, dr, gap in f:
            rr = CR.RI.rh_residual(A, Sx, np.array([V]), Bn, g)[0]
            ref, Vs, err = H["rd_refine"](A, side, Bn, Sx)
            print("lane %d %s: ahead Bt %+.5f  rotation to Bt %+.6f (V %.9f, drho %.1e, V-alfven %.1e, rh %.1e)  pinned refine: Bt %+.6f err %s dBt %.1e" % (
                lane, side, A[5][0], b, V, dr, -gap, rr, ref[5][0], err[0], abs(ref[5][0] - Sx[5][0])))
        if not f:
            print("lane %d %s: ahead Bt %+.5f  NO rotation found" % (lane, side, A[5][0]))
