"""Generate HLLD p* training shards: sample (src/data/sampling.py), label with
today's HLLD (src/data/labelling.py), write one .npz per chunk.

    python scripts/pstar_make_dataset.py --config configs/pstar-union.yaml \
        --out /mnt/rafast/.../pstar_union --chunks 0-59 --workers 60

A chunk is reproducible from (seed, chunk index): its generator is
np.random.default_rng([seed, chunk]) and its EOS gamma is
gammas[chunk % len(gammas)].  An existing shard is skipped, so a killed job
is resumed by submitting it again.  Each shard holds every row, valid or
not, with its flags (labelling.py) and raw features; the split builder
(scripts/pstar_build_splits.py) keeps the valid ones for training.
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import numpy as np
import yaml


def _git_rev():
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root,
                              capture_output=True, text=True,
                              timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def make_chunk(args):
    k, cfg, out, rows, seed, batch, rev = args
    import torch
    torch.set_num_threads(1)
    from src.data.labelling import (PRIM_KEYS, invalid_reasons, label_batched,
                                    make_eos, to_prims)
    from src.data.sampling import COMPONENTS, merged_cfg, sample_pairs
    path = os.path.join(out, f"shard_{k:05d}.npz")
    if os.path.exists(path):
        return k, "exists", None
    t0 = time.time()
    gammas = cfg["generation"]["gammas"]
    gamma = float(gammas[k % len(gammas)])
    eos = make_eos(gamma)
    rng = np.random.default_rng([seed, k])
    scfg = merged_cfg(cfg.get("sampling"))
    S = sample_pairs(rows, scfg, rng, gamma)
    L, R = to_prims(S["L"], eos), to_prims(S["R"], eos)
    t1 = time.time()
    lab = label_batched(L, R, eos, 0, batch)
    t2 = time.time()
    prim = lambda s: np.stack([s[key].numpy() for key in PRIM_KEYS], axis=1)
    reasons = invalid_reasons(lab)
    meta = dict(chunk=k, seed=seed, gamma=gamma, rows=rows,
                accept=S["accept"], t_sample=t1 - t0, t_label=t2 - t1,
                git=rev, sampling=scfg, components=list(COMPONENTS),
                prim_keys=list(PRIM_KEYS), reasons=reasons)
    tmp = path + ".tmp.npz"
    np.savez(tmp, prim_L=prim(L), prim_R=prim(R), comp=S["comp"],
             gamma=np.full(rows, gamma), idir=np.zeros(rows, np.int8),
             meta=np.array(json.dumps(meta)), **lab)
    os.replace(tmp, path)
    return k, "done", dict(valid=float(lab["valid"].mean()),
                           t_sample=round(t1 - t0, 1),
                           t_label=round(t2 - t1, 1), gamma=gamma,
                           accept=round(S["accept"], 3), **reasons)


def parse_range(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunks", required=True, help="e.g. 0-59 or 0,3,7-9")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--rows", type=int, default=None,
                    help="rows per chunk (default: generation.rows_per_chunk)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--batch", type=int, default=50_000)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    gen = cfg["generation"]
    rows = a.rows or int(gen["rows_per_chunk"])
    seed = a.seed if a.seed is not None else int(gen["seed"])
    os.makedirs(a.out, exist_ok=True)
    rev = _git_rev()
    chunks = parse_range(a.chunks)
    print(f"== {len(chunks)} chunks x {rows} rows -> {a.out}  seed={seed} "
          f"workers={a.workers}  git={rev}", flush=True)
    jobs = [(k, cfg, a.out, rows, seed, a.batch, rev) for k in chunks]
    t0 = time.time()
    if a.workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(a.workers, maxtasksperchild=4) as pool:
            for k, st, info in pool.imap_unordered(make_chunk, jobs):
                print(f"  chunk {k:5d} {st} {info or ''}  [{time.time()-t0:.0f}s]",
                      flush=True)
    else:
        for j in jobs:
            k, st, info = make_chunk(j)
            print(f"  chunk {k:5d} {st} {info or ''}  [{time.time()-t0:.0f}s]",
                  flush=True)
    print(f"== done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
