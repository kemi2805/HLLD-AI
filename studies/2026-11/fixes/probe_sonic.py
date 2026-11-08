"""The reversed slow-family Hugoniot from our fast state, per side: shock speed,
the slow speed behind, sonic gap, entropy, error -- why compound_wave's sonic
search finds nothing on near-degenerate (slow ~ Alfven) states."""
import sys
import numpy as np
sys.path.insert(0, "/Users/miler/Codes/HLLD/scripts")
import compound_wave as CW
import compound_refused as CR          # loads the answers, the solvers
from rmhd.batched import slow_shock_b as SSB
g, E, S5, pop, ans = CR.g, CR.E, CR.S5, CR.pop, CR.ans
W = CW.FB.make_waves(g)
shock = SSB.make_solver(g)
for arg in sys.argv[1:]:
    lane, side = int(arg[:-1]), arg[-1]
    UL, UR = pop["UL"][lane], pop["UR"][lane]
    Bn = np.array([UL[7]])
    L0, R0, alpha, _ = S5["to_planar"](CR.one(UL[:7]), CR.one(UR[:7]), Bn)
    Z, S = ans[lane]
    ca, sa = np.cos(alpha[0]), np.sin(alpha[0])
    PA, PD = Z[1][1], Z[6][1]
    A, _, _ = W["fast_wave"](np.array([PA if side == "L" else PD]), L0 if side == "L" else R0,
                             "LF" if side == "L" else "RF", Bn)
    sw = "LS" if side == "L" else "RS"
    la = W["alfven_speed"](A, Bn, side)
    print("== lane %d %s: A Bt %.5f  xi_slow(A) %.8e  alfven(A) %.8e" % (lane, side, A[5][0], W["xi"](A, sw, Bn)[0], la[0]))
    import os
    sgn = -np.sign(A[5][0]) * (-1.0 if os.environ.get("SAME") else 1.0)
    bs = sgn * np.abs(A[5][0]) * np.exp(np.linspace(np.log(float(os.environ.get("BLO", "1e-4"))), np.log(float(os.environ.get("BHI", "12"))), 40))
    k = bs.size
    Ak = [np.repeat(c, k) for c in A]
    with CW.wide_window():
        V, Sx, e = shock(bs, np.zeros(k), Ak, sw, np.repeat(Bn, k), accuracy=1e-12)
    xs = W["xi"](Sx, sw, np.repeat(Bn, k))
    las = W["alfven_speed"](Sx, np.repeat(Bn, k), side)
    ent = CR.RI.entropy(Sx, np.repeat(Bn, k), g) / CR.RI.entropy(Ak, np.repeat(Bn, k), g) - 1
    for j in range(k):
        print("  b %+.5f  err %d  V %+.8e  xi_s(S)-V %+.2e  alfven(S)-V %+.2e  alfven(A)-V %+.2e  drho %+.2e  dS %+.2e" % (
            bs[j], e[j], V[j], xs[j] - V[j], las[j] - V[j], la[0] - V[j], Sx[0][j] / A[0][0] - 1, ent[j]))
