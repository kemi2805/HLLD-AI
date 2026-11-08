"""Lane 215-type faces: from the tube's post-fast states, the sonic
intermediate shock on each side (compound_wave.sonic_shock) and the planar
residual with the compound element on both sides along Bt_CD.  numpy slow
shock (RMHD_SLOWSHOCK unset), wide window inside the element."""
import os, sys
import numpy as np
sys.path.insert(0, "/Users/miler/Codes/HLLD/scripts")
sys.path.insert(0, os.environ.get("RMHD_DIR", "/Users/miler/Codes/rmhd_final"))
np.seterr(all="ignore")
import compound_wave as CW
from rmhd.batched import planar5_b as P5
POP = "/Users/miler/Codes/claude_scratch/r2d_retrain/anatomy/data/ledger/population.npz"
p = np.load(POP, allow_pickle=True); g = float(p["gamma"])
S = P5.make_solver(g); E = CW.make(g); W = S["waves"]
lane, PA, PD = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
UL, UR = p["UL"][lane], p["UR"][lane]
Bn = np.array([UL[7]])
L = [np.array([UL[j]]) for j in range(7)]; R = [np.array([UR[j]]) for j in range(7)]
L0, R0, alpha, _ = S["to_planar"](L, R, Bn)
A, _, _ = W["fast_wave"](np.array([PA]), L0, "LF", Bn)
D, _, _ = W["fast_wave"](np.array([PD]), R0, "RF", Bn)
for X, sw in ((A, "LS"), (D, "RS")):
    Sx, V, b, ok = E["sonic_shock"](X, sw, Bn)
    print("%s ahead Bt %.5f: sonic IS ok %d  V %.6f  b %.5f  behind rho %.6f Pt %.6f vx %.6f vt %.6f" % (
        sw, X[5][0], ok[0], V[0], b[0], Sx[0][0], Sx[1][0], Sx[2][0], Sx[3][0]))
for mode in ((True, True), (True, False), (False, True)):
    best = None
    for bt in np.linspace(-0.2, 0.2, 41):
        f, err, st = E["structure"](L0, R0, [np.log(np.array([PA])), np.array([bt]), np.log(np.array([PD]))], Bn, mode)
        if not err[0]:
            nr = np.max(np.abs(f[:, 0]))
            if best is None or nr < best[0]:
                best = (nr, bt, f[:, 0])
    print("mode", mode, "best along Bt (fast pressures from the tube):", best)
    u0 = [np.log(np.array([PA])), np.array([best[1] if best else -0.088]), np.log(np.array([PD]))]
    u, nrm = E["solve"](L0, R0, Bn, u0, mode)
    ok, rep, st = E["verify"](L0, R0, Bn, u, mode)
    print("   solve: nrm %.1e unk %s  verified %s" % (nrm[0], np.round([c[0] for c in u], 6), ok[0]))
    print("   ", {k: (float(v[0]) if np.ndim(v) else v) for k, v in rep.items()})
