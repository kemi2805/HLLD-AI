"""Why does test_regular_answer_preferred fail: the two rotor lanes with
production's seed, REGULAR off and on, everything the solve says, and the
verifier's own intermediate test (wave_speeds_b.xi_all) next to the solver's."""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.getcwd(), "tests", "batched"))
np.seterr(all="ignore")
from rmhd.batched import coplanar7_b as CP7          # noqa: E402
from rmhd.batched import wave_speeds_b as WB         # noqa: E402
import test_planar_net as T                         # noqa: E402

L, R, Bn, s6 = T._rotor()
S = CP7.make_solver(T.GAMMA)
CP7.NET = ""
for reg in (False, True):
    CP7.REGULAR = reg
    st0 = dict(CP7.STATS)
    t0 = time.time()
    r = S["solve"](L, R, Bn, s6, accuracy=1e-8)
    print("== REGULAR %s (%.0f s): converged %s stage %s intermediate %s rot %s kinds %s n_iter %s" % (
        reg, time.time() - t0, r["converged"].tolist(), r["stage"].tolist(), r["intermediate"].tolist(),
        [x.tolist() for x in r["rot"]], [x.tolist() for x in r["kinds"]], r["n_iter"].tolist()))
    print("   STATS delta", {k: CP7.STATS[k] - st0[k] for k in CP7.STATS})
    R2, R3, R4, R5, R6, R7 = r["zones"]
    VL, VR = r["VsLv"], r["VsRv"]
    for i in range(2):
        z = lambda X: [X[c][i:i + 1] for c in range(7)]
        la3 = WB.xi_all(*z(R3), Bn[i:i + 1], T.GAMMA)[1][0]
        la4 = WB.xi_all(*z(R4), Bn[i:i + 1], T.GAMMA)[1][0]
        ra6 = WB.xi_all(*z(R6), Bn[i:i + 1], T.GAMMA)[2][0]
        ra5 = WB.xi_all(*z(R5), Bn[i:i + 1], T.GAMMA)[2][0]
        print("   lane %d: LS speed %.6f  la ahead %.6f behind %.6f  shock %s -> xi_all intermediate %s" % (
            i, VL[2][i], la3, la4, bool(r["shocks"][1][i]), bool(r["shocks"][1][i] and (VL[2][i] - la3) * (VL[2][i] - la4) < 0)))
        print("           RS speed %.6f  ra ahead %.6f behind %.6f  shock %s -> xi_all intermediate %s" % (
            VR[2][i], ra6, ra5, bool(r["shocks"][2][i]), bool(r["shocks"][2][i] and (VR[2][i] - ra6) * (VR[2][i] - ra5) < 0)))
