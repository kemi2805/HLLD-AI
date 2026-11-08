#!/usr/bin/env python3
"""Population for the per-sweep cost replay: every face the seven-wave Newton
did not answer in a sample of sweeps of a 64^2 run, with its sweep and the
rung that answered it in the run (from the coverage masks).

    python3 pop_sweeps.py RUN STEP_FIRST STEP_LAST EVERY OUT.npz

The unsolved shard rows of a sweep are the Newton's refused lanes in face
order; they are matched to the coverage's attempted & ~seven faces of the
same sweep (the count is checked per sweep)."""
import glob, sys
import numpy as np
R = "/mnt/rafast/miler/codes/rotor2d/HLLD/results"
run, s0, s1, every, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
PER = 6
H = R + "/" + run + "/harvest"
K = ("attempted", "seven", "planar5", "coplanar7", "planar4", "exact", "three_wave", "degenerate", "continuation")
UL, UR, SW, LAB, CLS = [], [], [], [], []
bad = 0
for fc in sorted(glob.glob(H + "/coverage_r000_*.npz")):
    fu = fc.replace("coverage_", "unsolved_")
    zc, zu = np.load(fc), np.load(fu)
    row = 0
    for j in range(len(zc["sweep"])):
        n = int(zc["n"][j])
        u = {k: np.unpackbits(zc[k][j])[:n].astype(bool) for k in K if k in zc.files}
        miss = u["attempted"] & ~u["seven"]
        m = int(miss.sum())
        sw = int(zc["sweep"][j])
        step = sw // PER + 1
        keep = s0 <= step <= s1 and (sw // PER - (s0 - 1)) % every == 0
        if keep:
            f = np.flatnonzero(miss)
            lab = np.full(m, 9, np.int8)                 # 9 = other
            lab[u["planar5"][f]] = 2
            lab[u["coplanar7"][f]] = 4
            lab[u["planar4"][f]] = 3
            lab[~u["exact"][f]] = 0                       # lost
            UL.append(zu["U_L"][row:row + m]); UR.append(zu["U_R"][row:row + m])
            SW.append(np.full(m, sw)); LAB.append(lab); CLS.append(zu["cls"][row:row + m])
        row += m
    if row != len(zu["U_L"]):
        bad += 1
        print("shard %s: %d unsolved rows, coverage says %d" % (fc.split("_")[-1], len(zu["U_L"]), row))
UL, UR, SW, LAB = (np.concatenate(x) for x in (UL, UR, SW, LAB))
CLS = np.concatenate(CLS)
print("shards with a count mismatch:", bad)
print("lanes %d in %d sweeps (steps %d-%d every %d); labels: lost %d p5 %d c7 %d p4 %d other %d" % (
    len(SW), len(np.unique(SW)), s0, s1, every, (LAB == 0).sum(), (LAB == 2).sum(), (LAB == 4).sum(),
    (LAB == 3).sum(), (LAB == 9).sum()))
g = float(np.load(sorted(glob.glob(H + "/unsolved_r000_*.npz"))[0])["gamma"])
np.savez_compressed(out, UL=UL, UR=UR, sweep=SW, label=LAB, cls=CLS, gamma=g,
                    group=(LAB != 2) & (LAB != 9))
