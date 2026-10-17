"""Labelled shards -> train/val/test files for the p* network.

    python scripts/pstar_build_splits.py --shards DIR --out SPLITS \
        [--transform asinh] [--ref ptot_avg] [--frac 0.9 0.05 0.05] [--seed 0]

Keeps the VALID rows only (labelling.py: converged, wave order holds,
B_n != 0, finite, p* > 0) and reports the invalid ones by reason.  One
random split of the valid rows is shared by every feature version, so the
versions' test sets are the same Riemann problems.  Per version
(SPLITS/<version>/):

    X_{train,val,test}.npy   float32, normalised exactly as at inference:
                             ai_features.normalise with the stats below
    y_{train,val,test}.npy   float32, standardised log10 p*
    norm_stats.npz           mu, sigma, y_mu, y_sigma, feature_version,
                             x_transform, x_scale (+ provenance)

and once (SPLITS/):
    test_rows.npz            the test split's Riemann problems in full
                             (prims, gamma, idir, labels, flags) for
                             scripts/pstar_evaluate.py
    invalid_rows.npz         up to --keep-invalid HLLD-invalid rows, flagged
    index.npz                (shard, row) of every train/val/test row
    summary.json             counts, reasons, ranges, provenance

Statistics (x_scale, mu, sigma, y_mu, y_sigma) come from the TRAIN split.
x_scale is the median |x| of each feature over the train split (1 where
that median is 0): asinh(x / x_scale) is linear below the typical size and
logarithmic above it.
"""
import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "8")

import numpy as np
import torch

from src.physics import ai_features as ai

LABEL_KEYS = ("p_star", "p_root", "resid", "p_hll", "n_iter", "converged",
              "wave_order_ok", "bn_zero", "finite", "valid")
ROW_KEYS = ("prim_L", "prim_R", "gamma", "idir", "comp") + LABEL_KEYS


def p_ref_rows(rows, chunk=2_000_000):
    """ai_features.p_ref_from_prims for every row (the call hlld_ai_flux
    makes), grouped by the row's EOS."""
    from src.data.labelling import PRIM_KEYS, make_eos
    out = np.empty(rows["gamma"].size)
    for g in np.unique(rows["gamma"]):
        idx = np.flatnonzero(rows["gamma"] == g)
        eos = make_eos(float(g))
        for i0 in range(0, idx.size, chunk):
            m = idx[i0:i0 + chunk]
            mk = lambda P: {k: torch.from_numpy(np.ascontiguousarray(P[m, i]))
                            for i, k in enumerate(PRIM_KEYS)}
            out[m] = ai.p_ref_from_prims(mk(rows["prim_L"]), mk(rows["prim_R"]),
                                         eos).numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--versions", default=",".join(ai.FEATURE_SETS))
    ap.add_argument("--transform", default="asinh", choices=("asinh", "none"))
    ap.add_argument("--ref", default="none", choices=("none", "ptot_avg"),
                    help="ptot_avg: scale-free features and target "
                         "log10(p*/p_ref) (ai_features.X_REF_POWER)")
    ap.add_argument("--frac", type=float, nargs=3, default=(0.9, 0.05, 0.05))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-invalid", type=int, default=1_000_000)
    ap.add_argument("--max-shards", type=int, default=None)
    a = ap.parse_args()
    t0 = time.time()
    versions = [v for v in a.versions.split(",") if v]
    for v in versions:
        ai.n_features(v)
    files = sorted(glob.glob(os.path.join(a.shards, "shard_*.npz")))
    if a.max_shards:
        files = files[: a.max_shards]
    if not files:
        sys.exit(f"no shards in {a.shards}")
    os.makedirs(a.out, exist_ok=True)

    X = {v: [] for v in versions}
    rows = {k: [] for k in ROW_KEYS}
    inv = {k: [] for k in ROW_KEYS}
    src_id = []
    reasons_tot: dict = {}
    n_all = 0
    metas = []
    rng_inv = np.random.default_rng(a.seed + 1)
    for si, f in enumerate(files):
        z = np.load(f)
        meta = json.loads(str(z["meta"]))
        metas.append(dict(file=os.path.basename(f), chunk=meta.get("chunk"),
                          gamma=meta.get("gamma"), git=meta.get("git")))
        for k, c in meta.get("reasons", {}).items():
            reasons_tot[k] = reasons_tot.get(k, 0) + int(c)
        valid = z["valid"]
        n_all += valid.size
        vi = np.flatnonzero(valid)
        for v in versions:
            X[v].append(z[f"X_{v}"][vi])
        for k in ROW_KEYS:
            rows[k].append(z[k][vi])
        src_id.append(np.stack([np.full(vi.size, si), vi], axis=1))
        ii = np.flatnonzero(~valid)
        cap = a.keep_invalid // len(files)
        if ii.size > cap:
            ii = np.sort(rng_inv.choice(ii, cap, replace=False))
        for k in ROW_KEYS:
            inv[k].append(z[k][ii])
        print(f"  {os.path.basename(f)}: {vi.size}/{valid.size} valid "
              f"[{time.time()-t0:.0f}s]", flush=True)

    rows = {k: np.concatenate(v) for k, v in rows.items()}
    src_id = np.concatenate(src_id)
    n = rows["valid"].size
    perm = np.random.default_rng(a.seed).permutation(n)
    n_tr = int(a.frac[0] * n)
    n_va = int(a.frac[1] * n)
    split = {"train": perm[:n_tr], "val": perm[n_tr:n_tr + n_va],
             "test": perm[n_tr + n_va:]}
    np.savez(os.path.join(a.out, "index.npz"), files=np.array(
        [os.path.basename(f) for f in files]),
        **{f"{s}": src_id[ix] for s, ix in split.items()})
    np.savez(os.path.join(a.out, "test_rows.npz"),
             **{k: rows[k][split["test"]] for k in ROW_KEYS})
    np.savez(os.path.join(a.out, "invalid_rows.npz"),
             **{k: np.concatenate(v) for k, v in inv.items()})

    p_ref = p_ref_rows(rows) if a.ref == "ptot_avg" else None
    y_log = np.log10(rows["p_star"] if p_ref is None else rows["p_star"] / p_ref)
    tr = split["train"]
    y_mu, y_sigma = float(y_log[tr].mean()), float(y_log[tr].std() + 1e-12)
    summary = dict(
        shards=len(files), rows_total=int(n_all), rows_valid=int(n),
        rows_invalid=int(n_all - n), invalid_reasons=reasons_tot,
        split={s: int(ix.size) for s, ix in split.items()},
        valid_by_gamma={f"{g:.4f}": int((np.isclose(rows['gamma'], g)).sum())
                        for g in np.unique(rows["gamma"])},
        log10_pstar=dict(min=float(y_log.min()), max=float(y_log.max()),
                         mean=y_mu, std=y_sigma),
        transform=a.transform, ref=a.ref, seed=a.seed, versions=versions,
        shard_meta=metas[:3] + (["..."] if len(metas) > 3 else []),
    )

    for v in versions:
        Xv = np.concatenate(X[v])
        X[v] = None
        sub = os.path.join(a.out, v)
        os.makedirs(sub, exist_ok=True)
        Xt = torch.from_numpy(Xv[tr])
        if p_ref is not None:
            Xt = ai.scale_by_ref(Xt, v, torch.from_numpy(p_ref[tr]))
        stats = dict(feature_version=np.array(v),
                     x_transform=np.array(a.transform), ref=np.array(a.ref),
                     y_mu=np.array(y_mu), y_sigma=np.array(y_sigma))
        if a.transform == "asinh":
            scale = np.median(np.abs(Xt.numpy()), axis=0)
            scale = np.where(scale > 0, scale, 1.0)
            stats["x_scale"] = scale
            Xt = torch.asinh(Xt / torch.as_tensor(scale))
        mu = Xt.mean(dim=0).numpy()
        sigma = (Xt.std(dim=0) + 1e-12).numpy()
        stats.update(mu=mu, sigma=sigma)
        del Xt
        stats["provenance"] = np.array(json.dumps(dict(
            shards=os.path.abspath(a.shards), n_train=int(tr.size),
            seed=a.seed, frac=list(a.frac), built=time.strftime("%Y-%m-%dT%H:%M:%S"))))
        np.savez(os.path.join(sub, "norm_stats.npz"), **stats)
        for s, ix in split.items():
            # the same function hlld_ai_flux calls on the raw features
            Xs = ai.normalise(torch.from_numpy(Xv[ix]), stats,
                              None if p_ref is None else
                              torch.from_numpy(p_ref[ix])).numpy()
            ys = ((y_log[ix] - y_mu) / y_sigma).astype(np.float32)
            np.save(os.path.join(sub, f"X_{s}.npy"), Xs)
            np.save(os.path.join(sub, f"y_{s}.npy"), ys)
        summary[f"features_{v}"] = dict(
            n=int(Xv.shape[1]), names=list(ai.FEATURE_SETS[v]),
            raw_min=Xv.min(axis=0).tolist(), raw_max=Xv.max(axis=0).tolist())
        print(f"  {v}: written [{time.time()-t0:.0f}s]", flush=True)
        del Xv

    summary["wall_s"] = round(time.time() - t0, 1)
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print(json.dumps({k: summary[k] for k in ("rows_total", "rows_valid",
                                              "rows_invalid", "invalid_reasons",
                                              "split", "valid_by_gamma",
                                              "log10_pstar")}, indent=1))


if __name__ == "__main__":
    main()
