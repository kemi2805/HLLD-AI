#!/usr/bin/env python
"""The planar dataset: re-solve a population of refused faces
(pop_refused.py) with the full exact-flux chain and keep every answer as a
harvest row -- the seven-wave Newton's (source 0), the five-wave solver's (1),
the planar seven-wave rung's (2) and the four-unknown solver's (3), each with
its slow-wave steering, rotations and stage (RMHD_HARVEST_RESCUES=1, which the
caller sets, with the rest of the chain's switches).

    python build_dataset.py POP.npz OUTDIR [--chunk 4096]

The faces go through in chunks of consecutive rows (shards are time order);
chunk c is written as OUTDIR/{solved,unsolved,coverage}_r000_<c>.npz and
recorded in OUTDIR/manifest.json with its source shards, faces, what
answered them and its wall.  A chunk already in the manifest is skipped, so
the job resumes where it stopped.  The call is the run's: tau_weak 1e-2,
tau_bt 1e-9, 2 retries, 40 iterations."""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HLLD = os.environ["HLLD_DIR"]
sys.path.insert(0, HLLD)
from rmhd.eos import set_eos                                     # noqa: E402
set_eos("ideal")
from src.physics import exact_flux as EF                         # noqa: E402
from src.physics.eos import hybrid_eos                           # noqa: E402
from src.physics.harvest import Harvester                        # noqa: E402
from src.physics.hlld import LAST_DIAG                           # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pop")
    ap.add_argument("out")
    ap.add_argument("--chunk", type=int, default=4096)
    a = ap.parse_args()
    assert EF._HARVEST_RESCUES, "set RMHD_HARVEST_RESCUES=1"
    os.makedirs(a.out, exist_ok=True)
    print("env:", {k: v for k, v in sorted(os.environ.items()) if k.startswith("RMHD_")}, flush=True)
    p = np.load(a.pop, allow_pickle=True)
    UL, UR, SH, KD = p["UL"], p["UR"], p["shard"], p["kind"]
    gam = float(p["gamma"])
    eos = hybrid_eos(K=0.0, gamma=gam, gamma_th=gam)
    mpath = os.path.join(a.out, "manifest.json")
    man = json.load(open(mpath)) if os.path.exists(mpath) else dict(
        pop=os.path.abspath(a.pop), run=str(p["run"]), gamma=gam, chunk=a.chunk, chunks={})

    def lab(U):
        rho, Pt, vx, vy, vz, By, Bz, Bx = (U[:, j] for j in range(8))
        W2 = 1.0 / (1.0 - (vx * vx + vy * vy + vz * vz))
        eta = Bx * vx + By * vy + Bz * vz
        b2 = (Bx * Bx + By * By + Bz * Bz) / W2 + eta * eta
        t = lambda v: torch.tensor(np.ascontiguousarray(v), dtype=torch.float64)
        d = {"rho": t(rho), "vx": t(vx), "vy": t(vy), "vz": t(vz),
             "p": t(Pt - 0.5 * b2), "Bx": t(Bx), "By": t(By), "Bz": t(Bz)}
        d["eps"] = eos.eps__press_rho(d["p"], d["rho"])
        return d

    H = Harvester(a.out, gam, only_retried=False, max_solved=10 ** 9,
                  max_unsolved=10 ** 9, flush_every=10 ** 9)
    n = len(UL)
    nch = (n + a.chunk - 1) // a.chunk
    t_all = time.time()
    for ci in range(nch):
        if str(ci) in man["chunks"]:
            continue
        j = np.arange(ci * a.chunk, min(n, (ci + 1) * a.chunk))
        t0 = time.time()
        EF.exact_flux_batched(lab(UL[j]), lab(UR[j]), eos, idir=0, tau_weak=1e-2,
                              tau_bt=1e-9, n_retries=2, max_iter=40, harvester=H)
        d = dict(LAST_DIAG)
        H._shard = ci
        H.flush()
        cnt = lambda k: int(np.asarray(d[k], bool).sum()) if k in d else None
        ex = d.get("exact_mask")
        ex = np.asarray(ex.numpy() if hasattr(ex, "numpy") else ex) if ex is not None else None
        man["chunks"][str(ci)] = dict(
            rows=[int(j[0]), int(j[-1]) + 1], shards=[int(SH[j].min()), int(SH[j].max())],
            kind1=int((KD[j] == 1).sum()), attempted=int(np.asarray(d["sel"]).size),
            exact=None if ex is None else int(ex.sum()),
            five_wave=cnt("planar5_mask"), planar7=cnt("coplanar7_late_mask"),
            four_unknown=cnt("planar4_mask"), wall=round(time.time() - t0, 1))
        with open(mpath + ".tmp", "w") as f:
            json.dump(man, f, indent=1)
        os.replace(mpath + ".tmp", mpath)
        c = man["chunks"][str(ci)]
        print("chunk %d/%d: %d faces (shards %d-%d), attempted %d exact %s: five-wave %s planar-seven %s "
              "four-unknown %s; %.0f s (total %.0f s)" % (
                  ci + 1, nch, j.size, c["shards"][0], c["shards"][1], c["attempted"], c["exact"],
                  c["five_wave"], c["planar7"], c["four_unknown"], c["wall"], time.time() - t_all), flush=True)
    print("done: %d chunks in %.0f s" % (nch, time.time() - t_all))


if __name__ == "__main__":
    main()
