"""Label RECORDED interface states (run_2d.py --envelope-states) with today's
HLLD, in the shard format of scripts/pstar_make_dataset.py.

    python scripts/pstar_label_states.py results/pstar_rec/rotor_256_states.npz \
        --out .../rotor_256_rows.npz [--workers 32]

Each face is solved in the orientation the run solved it (idir = its sweep
axis) with the recorded eps, i.e. exactly the Riemann problem the run's
HLLD was handed; gamma comes from the recording.  Valid and invalid faces
are all kept, flagged (src/data/labelling.py).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import numpy as np


def _work(args):
    path, lo, hi, gamma, batch = args
    import torch
    torch.set_num_threads(1)
    from src.data.labelling import PRIM_KEYS, label_batched, make_eos
    z = np.load(path)
    keys = [str(k) for k in z["state_keys"]]
    assert tuple(keys) == tuple(PRIM_KEYS), keys
    sl, sr = z["states_L"][lo:hi], z["states_R"][lo:hi]
    sL = {k: sl[:, i] for i, k in enumerate(keys)}
    sR = {k: sr[:, i] for i, k in enumerate(keys)}
    return label_batched(sL, sR, make_eos(gamma), z["axis"][lo:hi], batch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("states")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--batch", type=int, default=50_000)
    a = ap.parse_args()
    t0 = time.time()
    z = np.load(a.states)
    if "states_L" not in z.files:
        sys.exit(f"{a.states} has no full states (record with --envelope-states)")
    gamma = float(z["gamma"])
    n = z["states_L"].shape[0]
    step = -(-n // max(a.workers, 1))
    jobs = [(a.states, lo, min(lo + step, n), gamma, a.batch)
            for lo in range(0, n, step)]
    if a.workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(a.workers) as pool:
            parts = pool.map(_work, jobs)
    else:
        parts = [_work(j) for j in jobs]
    lab = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    from src.data.labelling import invalid_reasons
    reasons = invalid_reasons(lab)
    meta = dict(source=os.path.abspath(a.states), gamma=gamma,
                problem=str(z["problem"]), n=int(z["n"]),
                tend=float(z["tend"]), every=int(z["every"]) if "every" in z.files else None,
                rows=int(n), reasons=reasons, wall_s=round(time.time() - t0, 1))
    np.savez(a.out, prim_L=z["states_L"], prim_R=z["states_R"],
             gamma=np.full(n, gamma), idir=z["axis"].astype(np.int8),
             call=z["call"], comp=np.full(n, -1, np.int8),
             meta=np.array(json.dumps(meta)), **lab)
    v = lab["valid"]
    print(f"{a.states}: {n} faces, {v.sum()} valid ({v.mean():.4f}), "
          f"axis 0/1 = {(z['axis'] == 0).sum()}/{(z['axis'] == 1).sum()}, "
          f"reasons {reasons}, {time.time()-t0:.0f}s -> {a.out}", flush=True)
    envelope_report(z["states_L"], z["states_R"], z["axis"], gamma, v)


def envelope_report(SL, SR, axis, gamma, valid, cfg_path=None):
    """The measured envelope per sweep axis, and how much of it the training
    box + caps (configs/pstar-union.yaml) contain."""
    import yaml
    from src.data.sampling import merged_cfg
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = merged_cfg(yaml.safe_load(open(cfg_path or os.path.join(
        root, "configs", "pstar-union.yaml")))["sampling"])
    box, caps = cfg["box"], cfg["caps"]
    n = SL.shape[0]
    rows = np.arange(n)
    inside = np.ones(n, bool)
    cols = {}
    for tag, S in (("L", SL), ("R", SR)):
        rho, vx, vy, vz, p = S[:, 0], S[:, 1], S[:, 2], S[:, 3], S[:, 4]
        B = S[:, 6:9]
        v = np.stack([vx, vy, vz], 1)
        v2 = (v ** 2).sum(1)
        W = 1.0 / np.sqrt(np.clip(1.0 - v2, 1e-300, None))
        bn = np.abs(B[rows, axis])
        bt = np.sqrt(np.clip((B ** 2).sum(1) - bn ** 2, 0.0, None))
        b2 = (B ** 2).sum(1) * (1 - v2) + (v * B).sum(1) ** 2
        sig = b2 / (rho + gamma / (gamma - 1) * p)
        lT = np.log10(p / rho)
        cols.update({f"lrho_{tag}": np.log10(rho), f"lp_{tag}": np.log10(p),
                     f"W_{tag}": W, f"Bt_{tag}": bt, f"sigma_{tag}": sig,
                     f"lT_{tag}": lT})
        cols["Bn"] = bn
        inside &= ((np.log10(rho) >= box["lrho"][0]) & (np.log10(rho) <= box["lrho"][1])
                   & (np.log10(p) >= box["lpress"][0]) & (np.log10(p) <= box["lpress"][1])
                   & (W <= 1 + 10 ** box["wm1_log"][1]) & (bn <= box["Bn"][1])
                   & (bt <= box["Bt"][1]) & (sig <= caps["sigma_max"])
                   & (lT >= caps["log_T"][0]) & (lT <= caps["log_T"][1]))
    print(f"  inside the training box and caps (both states): "
          f"{inside.mean():.4f} of all faces, {inside[valid].mean():.4f} of "
          f"the HLLD-valid ones")
    for ax in np.unique(axis):
        m = axis == ax
        print(f"  axis {int(ax)} ({m.sum()} faces):  min / 0.1% / median / 99.9% / max")
        for k, c in cols.items():
            q = np.quantile(c[m], [0, 0.001, 0.5, 0.999, 1])
            print(f"    {k:9s} " + " ".join(f"{x:10.4g}" for x in q))


if __name__ == "__main__":
    main()
