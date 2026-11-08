"""For lanes whose signed-field answer has an EXPANSIVE reversing jump: from
the state ahead of that jump, where does the slow rarefaction (pressure
falling) take B_t -- toward the jump's target through zero, or away?"""
import sys
import numpy as np
import sol_profiles as SP
off = np.load("/Users/miler/Codes/claude_scratch/r2d_retrain/fixes/replay_final2/replay_off.npz")
for lane, wave in ((1245, "LS"), (788, "LS"), (214, "LS"), (1013, "LS")):
    p = float(abs(off["pstar"][lane]))
    sol = SP.solve(lane, (2, 2), p, p)
    A, B, C, D = sol["zones"]
    ahead, behind = (A, B) if wave == "LS" else (D, C)
    Bn = sol["Bn"]
    k = 9
    drop = np.linspace(0, 2.5 * (ahead[1][0] - behind[1][0]), k)
    y, _, e = SP.FAN_P(ahead[1][0] - drop, SP._rep(ahead, k), wave, np.repeat(Bn, k))
    print("lane %d %s: ahead Bt %+.4f -> jump target %+.4f (planar), P drop across the jump %.2e" % (
        lane, wave, ahead[5][0], behind[5][0], ahead[1][0] - behind[1][0]))
    print("   slow fan from the same state, B_t at P drops %s:" % np.array2string(drop, precision=1),
          np.array2string(y[5], precision=4), " err", e.astype(int))
