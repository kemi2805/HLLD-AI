#!/usr/bin/env python
"""Re-solve replay faces with and without the planar network, harvest each
answer (RMHD_HARVEST_RESCUES) and judge it with verify_rmhd.py, which writes
the conservation laws from the equations, independent of the solvers.

    python face_detail.py FACES.npz OUTDIR      (FACES: extract_face.py)
"""
import glob
import os
import sys

import numpy as np
import torch

FIX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ["HLLD_DIR"])
from rmhd.eos import set_eos                                     # noqa: E402
set_eos("ideal")
from rmhd.batched import coplanar7_b as CP7                      # noqa: E402
from src.physics import exact_flux as EF                         # noqa: E402
from src.physics.eos import hybrid_eos                           # noqa: E402
from src.physics.harvest import Harvester                        # noqa: E402
from src.physics.hlld import LAST_DIAG                           # noqa: E402
sys.path.insert(0, FIX)
import verify_rmhd as V                                          # noqa: E402

NET = "data/ml_planar_v2a.pt"


def lab(U, eos):
    rho, Pt, vx, vy, vz, By, Bz, Bx = (U[:, j] for j in range(8))
    W2 = 1.0 / (1.0 - (vx * vx + vy * vy + vz * vz))
    eta = Bx * vx + By * vy + Bz * vz
    b2 = (Bx * Bx + By * By + Bz * Bz) / W2 + eta * eta
    t = lambda v: torch.tensor(np.ascontiguousarray(v), dtype=torch.float64)
    d = {"rho": t(rho), "vx": t(vx), "vy": t(vy), "vz": t(vz),
         "p": t(Pt - 0.5 * b2), "Bx": t(Bx), "By": t(By), "Bz": t(Bz)}
    d["eps"] = eos.eps__press_rho(d["p"], d["rho"])
    return d


def main():
    d = np.load(sys.argv[1], allow_pickle=True)
    out = sys.argv[2]
    faces = sorted({int(k.split("_")[0][1:]) for k in d.files})
    assert EF._HARVEST_RESCUES, "set RMHD_HARVEST_RESCUES=1"
    for fi in faces:
        UL, UR, g = d["f%d_UL" % fi][None], d["f%d_UR" % fi][None], float(d["f%d_gamma" % fi])
        print("######## face %d %s" % (fi, list(d["f%d_info" % fi])))
        eos = hybrid_eos(K=0.0, gamma=g, gamma_th=g)
        for arm in ("off", "net"):
            CP7.NET = "" if arm == "off" else NET
            hd = os.path.join(out, "f%d_%s" % (fi, arm))
            os.makedirs(hd, exist_ok=True)
            for f in glob.glob(os.path.join(hd, "*.npz")):
                os.remove(f)
            H = Harvester(hd, g, only_retried=False, max_solved=10 ** 9, max_unsolved=10 ** 9,
                          flush_every=10 ** 9)
            F, _, _ = EF.exact_flux_batched(lab(UL, eos), lab(UR, eos), eos, idir=0, tau_weak=1e-2,
                                            tau_bt=1e-9, n_retries=2, max_iter=40, harvester=H)
            dd = dict(LAST_DIAG)
            H._shard = 0
            H.flush()
            fl = np.array([float(F[k].reshape(-1)[0]) for k in sorted(F)])
            print("== %s: exact %s, late rung %s stage %s rot %s, four-unknown %s kinds %s" % (
                arm, bool(np.asarray(dd["exact_mask"])[0]),
                np.asarray(dd["coplanar7_late_mask"]).tolist(), np.asarray(dd["coplanar7_late_stage"]).tolist(),
                np.asarray(dd["coplanar7_late_rot"]).reshape(-1).tolist(), np.asarray(dd["planar4_mask"]).tolist(),
                np.asarray(dd["planar4_kinds"]).reshape(-1).tolist()))
            print("   flux %s = %s" % (sorted(F), np.array2string(fl, precision=10)))
            s = sorted(glob.glob(os.path.join(hd, "solved_*.npz")))
            if not s:
                print("   no solved row")
                continue
            z = np.load(s[-1])
            k = np.flatnonzero(z["source"] >= 1)
            if k.size == 0:
                print("   no rescue row; sources", z["source"].tolist())
                continue
            k = k[-1:]
            print("   harvested source %d, steer %s, rot %s, stage %s" % (
                z["source"][k[0]], z["steer"][k[0]].tolist(), z["rot"][k[0]].tolist(), z["stage"][k[0]]))
            print("   speeds (LF LA LS CD RS RA RF) %s" % np.array2string(z["speeds"][k[0]], precision=6))
            Z = z["zones"][k[0]]
            for j, nm in enumerate(("L", "R2", "R3", "R4", "R5", "R6", "R7", "R")):
                print("   %-3s rho %.8f Pt %.8f v (%.6f %.6f %.6f) B_t (%.6f %.6f)" % (nm, *Z[j]))
            r = V.verify(z["U_L"][k], z["U_R"][k], z["zones"][k], z["speeds"][k], g)
            print("   verify: weak solution %s, admissible %s, order %s" % (
                bool(r["weak"][0]), bool(r["admissible"][0]), bool(r["order_ok"][0])))
            for w, nm in enumerate(V.NAMES):
                print("     %s %-5s rh %.1e fan %.1e dS %.2e intermediate %s" % (
                    nm, r["kind"][0, w], r["rh"][0, w], r["fan"][0, w], r["dS"][0, w], bool(r["inter"][0, w])))
    CP7.NET = ""


if __name__ == "__main__":
    main()
