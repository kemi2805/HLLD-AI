"""Is a p* network invariant under the rotor's pi-rotation, face by face?

    python scripts/pstar_mirror_check.py ROWS.npz --ckpt v3_s1.pt \
        --stats v3_s1_norm_stats.npz [--max-faces 20000]

The rotation maps a face (L, R) onto its partner (rot R, rot L): L and R
swap, v -> -v, B -> -B (or B -> B with the field flip folded in, as
run_2d.symmetry_error's parities do).  HLLD's p* is invariant under it to
round-off.  A one-shot network sees the partner's features permuted and
need not be; this measures by how much, and how often the two partners'
HLLE-fallback decisions differ -- one-shot and warm-started -- which is
what the rotor's symmetry error picks up.  ROWS is any file in the shard
format (pstar_label_states.py output), only its HLLD-valid faces are used.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.data.labelling import PRIM_KEYS, make_eos
from src.models.network import PressureNet
from src.physics.hlld import hlld_ai_flux, hlld_flux


def side(P, r, vsign=1.0, bsign=1.0):
    s = {k: torch.from_numpy(np.ascontiguousarray(P[r, i]))
         for i, k in enumerate(PRIM_KEYS)}
    for k in ("vx", "vy", "vz"):
        s[k] = vsign * s[k]
    for k in ("Bx", "By", "Bz"):
        s[k] = bsign * s[k]
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--hidden", default="256,256,256,256,128,64")
    ap.add_argument("--max-faces", type=int, default=20000)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "2")))
    z = np.load(a.rows)
    stats = dict(np.load(a.stats))
    sd = torch.load(a.ckpt, map_location="cpu")
    net = PressureNet(sd["net.0.weight"].shape[1],
                      [int(h) for h in a.hidden.split(",")], "silu")
    net.load_state_dict(sd)
    net.eval()
    n = min(a.max_faces, z["valid"].size)
    print(f"{a.rows}: first {n} faces")
    for g in np.unique(z["gamma"][:n]):
        eos = make_eos(float(g))
        for d in np.unique(z["idir"][:n]):
            r = np.flatnonzero((z["idir"][:n] == d) & z["valid"][:n]
                               & np.isclose(z["gamma"][:n], g))
            sL, sR = side(z["prim_L"], r), side(z["prim_R"], r)
            mL, mR = side(z["prim_R"], r, -1.0, 1.0), side(z["prim_L"], r, -1.0, 1.0)
            _, _, p0 = hlld_flux(sL, sR, eos, idir=int(d))
            _, _, p1 = hlld_flux(mL, mR, eos, idir=int(d))
            ok = ((p0 > 0) & (p1 > 0)).numpy()
            hl = (np.abs(p1.numpy() - p0.numpy()) / np.abs(p0.numpy()))[ok]
            print(f" gamma {g:.4f} idir {int(d)}: {r.size} HLLD-valid faces; "
                  f"HLLD p* partner/face - 1: max {hl.max():.2e}")
            for mode in ("oneshot", "warmstart"):
                rd0, rd1 = {}, {}
                hlld_ai_flux(sL, sR, eos, net, stats, idir=int(d), mode=mode, row_diag=rd0)
                hlld_ai_flux(mL, mR, eos, net, stats, idir=int(d), mode=mode, row_diag=rd1)
                both = (~rd0["failed"] & ~rd1["failed"]).numpy()
                q = (np.abs(rd1["p_used"].numpy() - rd0["p_used"].numpy())
                     / rd0["p_used"].numpy())[both]
                dis = np.mean(rd0["failed"].numpy() != rd1["failed"].numpy())
                print(f"   {mode:9s}: p* partner/face - 1: median {np.median(q):.2e} "
                      f"p99 {np.quantile(q, 0.99):.2e} max {q.max():.2e}; "
                      f"HLLE-fallback decision differs on {dis:.4%} of the pairs")


if __name__ == "__main__":
    main()
