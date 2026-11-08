"""Build the four example solutions and print their waves (sanity check)."""
import numpy as np
import sol_profiles as SP
off = np.load("/Users/miler/Codes/claude_scratch/r2d_retrain/fixes/replay_final2/replay_off.npz")
ps = lambda l: float(abs(off["pstar"][l]))
cases = {
    62: dict(kinds=(0, 0), unk=[np.array([0.3029687]), np.array([0.3188758]), np.array([0.3188758]), np.array([0.3029711])]),
    215: dict(kinds=(1, 1), pl=ps(215), pr=ps(215)),
    900: dict(kinds=(2, 2), pl=ps(900), pr=ps(900)),
    858: dict(kinds=(1, 1), pl=ps(858), pr=float(SP.problem(858)[1][1][0]) * 1.001),
}
for lane, c in cases.items():
    sol = SP.solve(lane, c["kinds"], c.get("pl"), c.get("pr"), unk=c.get("unk"))
    xi, st, waves = SP.profile(sol)
    print("lane %d:" % lane, ", ".join("%s %s [%.6f, %.6f]" % w for w in waves))
    print("   Bt lab: A %.4f B %.4f C %.4f D %.4f  (planar alpha %.3f)" % tuple(
        [SP.lab(z, sol["alpha"])[5][0] for z in sol["zones"]] + [sol["alpha"][0]]))
