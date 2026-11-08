import sys
import numpy as np
sys.path.insert(0, "/Users/miler/Codes/rmhd_final/.claude/worktrees/stoic-swanson-701e4a/tests/batched")
from test_slow_shock_admissible import _problems, GAMMA
from rmhd.batched import slow_shock_b as SB
np.seterr(all="ignore")
hug, e416 = SB.make_hug(GAMMA), SB.make_eq416(GAMMA)
for seed, N in ((99, 300), (7, 60), (3, 1000)):
    ahead, Byt, Bzt, Bn = _problems(seed, N)
    for sw in ("LS", "RS"):
        for rt in (False, True):
            res = {}
            for flt in (False, True):
                SB.SLOW_ADMISSIBLE, SB.SLOW_RETRY = flt, rt
                V, b, e = SB.make_solver(GAMMA)(Byt, Bzt, ahead, sw, Bn, accuracy=1e-8)
                f1, _ = hug(V, b, ahead, Bn); f2, _ = e416(V, b, ahead, Bn)
                r = np.maximum(np.abs(f1), np.abs(f2)) / (1 + np.abs(ahead[1]))
                res[flt] = (V, b, e, ~e & ~(r <= 1e-6))
            (V0, b0, e0, l0), (V1, b1, e1, l1) = res[False], res[True]
            same = np.array_equal(l0, l1) and np.array_equal(V0[l0], V1[l1]) and all(np.array_equal(b0[j][l0], b1[j][l1]) for j in range(7))
            print("seed %d N %d %s retry %s: loose off %d on %d identical %s | err on & ~err off %d" % (
                seed, N, sw, rt, l0.sum(), l1.sum(), same, int((e1 & ~e0).sum())))
