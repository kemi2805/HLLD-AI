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
        Vs = W["alfven_speed"](A, Bn, side)
        rots = RH.coplanar_rotations(A, Bn, Vs, g)
        for lab, (Sx, ok) in rots.items():
            rr = CR.RI.rh_residual(A, Sx, Vs, Bn, g)[0]
            print("lane %d %s %s: ahead Bt %+.5f -> Bt %+.6f ok %s rh %.1e drho %+.1e dS %+.1e" % (
                lane, side, lab, A[5][0], Sx[5][0], ok[0], rr, Sx[0][0] / A[0][0] - 1, CR.entropy_ratio(A, Sx, Bn)))
