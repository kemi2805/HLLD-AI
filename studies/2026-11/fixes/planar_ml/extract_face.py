#!/usr/bin/env python3
"""The states of replay faces (replay_net.py population order), by face index:
    python3 extract_face.py ARM.npz OUT.npz FACE [FACE ...]
Rebuilds each face's sweep from the harvest exactly as replay_net.sweeps_of
does (numpy only)."""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import planar_data as PD                                         # noqa: E402

arm = np.load(sys.argv[1], allow_pickle=True)
runs = list(arm["runs"])
out = {}
for fi in (int(x) for x in sys.argv[3:]):
    run = runs[int(arm["face_run"][fi])]
    sw = int(arm["face_sweep"][fi])
    first = int(np.flatnonzero((arm["face_run"] == arm["face_run"][fi]) & (arm["face_sweep"] == sw))[0])
    pos = fi - first
    H = os.path.join(PD.R, run, "harvest")
    for f in sorted(glob.glob(os.path.join(H, "coverage_r*_*.npz"))):
        z = np.load(f)
        if sw not in set(z["sweep"].tolist()):
            continue
        j = z["sweep"].tolist().index(sw)
        tag = os.path.basename(f)[len("coverage_"):-len(".npz")]
        cnt = lambda key: np.array([int(np.unpackbits(z[key][q])[:int(z["n"][q])].sum()) for q in range(len(z["sweep"]))])
        n7, n4 = cnt("coplanar7"), cnt("planar4")
        s = np.load(os.path.join(H, "solved_" + tag + ".npz"))
        i2, i3 = np.flatnonzero(s["source"] == 2), np.flatnonzero(s["source"] == 3)
        o2, o3 = np.concatenate([[0], np.cumsum(n7)]), np.concatenate([[0], np.cumsum(n4)])
        rows = np.concatenate([i2[o2[j]:o2[j + 1]], i3[o3[j]:o3[j + 1]]])
        r = rows[pos]
        assert s["source"][r] == arm["face_src"][fi]
        out["f%d_UL" % fi], out["f%d_UR" % fi] = s["U_L"][r], s["U_R"][r]
        out["f%d_gamma" % fi] = float(np.asarray(s["gamma"]).reshape(-1)[0])
        out["f%d_zones_harvest" % fi] = s["zones"][r]
        out["f%d_info" % fi] = np.array([run, tag, str(sw), str(pos), str(int(s["source"][r]))])
        print(fi, run, tag, "sweep", sw, "pos", pos, "source", s["source"][r])
        break
np.savez(sys.argv[2], **out)
