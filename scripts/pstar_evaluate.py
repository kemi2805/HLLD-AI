"""Score p* networks against today's HLLD on labelled Riemann problems.

    python scripts/pstar_evaluate.py \
        --model new_v2=ckpt.pt,ckpt_norm_stats.npz \
        --model archived=default-mignone.pt,norm_stats.npz \
        --testset synthetic=SPLITS/test_rows.npz \
        --testset rotor=rotor_256_rows.npz --out eval.json --workers 48

A model is NAME=CKPT,STATS[,hidden=256-256-...][,act=silu]; the hidden
layers default to the archived [256,256,256,256,128,64].  A test set is any
file in the shard format (pstar_make_dataset.py / pstar_label_states.py /
pstar_build_splits.py's test_rows.npz).  Every face goes through
hlld.hlld_ai_flux -- the solver the network would actually serve -- in the
orientation (idir) and EOS (gamma) it was labelled with:

  p* error      |p_nn - p*_HLLD| / p*_HLLD on the faces HLLD solves
  valid         the one-shot flux passes HLLD's own checks (wave order,
                B_n != 0; p_nn > 0 by construction), as a fraction of the
                faces HLLD solves, and of all faces
  same root     the warm-started root-find (mode="warmstart": the secant
                from [0.99, 1.01] p_nn, HLLD's tolerance) converges to
                HLLD's p* (rel. 1e-8): the prediction sat in the basin of
                HLLD's root.  "other root": converged somewhere else
  flux error    max_k |F_ai,k - F_hlld,k| / max_k |F_hlld,k| per face, on the
                faces HLLD solves (a face the one-shot path hands to HLLE
                counts with its HLLE flux)

Breakdowns: by sampling component and gamma (synthetic sets), by sweep axis
(recorded sets), and on the faces inside the archived checkpoint's own
sampling box (W <= 2, 0.1 <= p <= 1, 0.1 <= rho <= 10, 0.01 <= |B_n| <= 1.01,
|B_t| <= 1; configs/default-mignone.yaml).
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

ARCHIVED_HIDDEN = (256, 256, 256, 256, 128, 64)
COMPONENTS = ("indep", "indep_planar", "jump", "jump_planar")
RES_KEYS = ("p_nn", "failed", "wave_bad", "resid_nn", "p_ws", "ws_conv",
            "ws_iter", "flux_err", "nonfinite")


def parse_model(s):
    name, _, rest = s.partition("=")
    parts = rest.split(",")
    m = dict(name=name, ckpt=parts[0], stats=parts[1],
             hidden=list(ARCHIVED_HIDDEN), act="silu")
    for p in parts[2:]:
        k, _, v = p.partition("=")
        if k == "hidden":
            m["hidden"] = [int(x) for x in v.split("-")]
        elif k == "act":
            m["act"] = v
    return m


def load_model(m):
    import torch
    from src.models.network import PressureNet
    stats = dict(np.load(m["stats"]))
    sd = torch.load(m["ckpt"], map_location="cpu")
    net = PressureNet(n_input=sd["net.0.weight"].shape[1], hidden=m["hidden"],
                      activation=m["act"])
    net.load_state_dict(sd)
    net.eval()
    return net, stats


# the current test set's arrays, loaded ONCE in the parent: the fork-started
# workers inherit them (an .npz cannot be memory-mapped, and re-reading a
# million-row file per task would move ~100 GB through CephFS)
_DATA: dict = {}


def _task(args):
    m, path, lo, hi, batch = args
    import torch
    torch.set_num_threads(1)
    from src.data.labelling import PRIM_KEYS, make_eos
    from src.physics.hlld import hlld_ai_flux, hlld_flux
    net, stats = load_model(m)
    z = _DATA if _DATA.get("_path") == path else np.load(path)
    sl, sr = z["prim_L"][lo:hi], z["prim_R"][lo:hi]
    gam, idir = z["gamma"][lo:hi], z["idir"][lo:hi].astype(int)
    n = hi - lo
    out = dict(p_nn=np.zeros(n), failed=np.zeros(n, bool),
               wave_bad=np.zeros(n, bool), resid_nn=np.zeros(n),
               p_ws=np.zeros(n), ws_conv=np.zeros(n, bool),
               ws_iter=np.zeros(n, np.int16), flux_err=np.zeros(n),
               nonfinite=np.zeros(n, bool))
    for g in np.unique(gam):
        eos = make_eos(float(g))
        for d in np.unique(idir):
            rows = np.flatnonzero((gam == g) & (idir == d))
            for i0 in range(0, rows.size, batch):
                r = rows[i0:i0 + batch]
                mk = lambda S: {k: torch.from_numpy(np.ascontiguousarray(S[r, i]))
                                for i, k in enumerate(PRIM_KEYS)}
                sL, sR = mk(sl), mk(sr)
                fH, _, _ = hlld_flux(sL, sR, eos, idir=int(d))
                rd, rw = {}, {}
                fA, _, _ = hlld_ai_flux(sL, sR, eos, net, stats, idir=int(d),
                                        row_diag=rd)
                hlld_ai_flux(sL, sR, eos, net, stats, idir=int(d),
                             mode="warmstart", row_diag=rw)
                ks = sorted(fH)
                FH = torch.stack([fH[k] for k in ks], 1)
                FA = torch.stack([fA[k] for k in ks], 1)
                scale = FH.abs().max(1).values.clamp(min=1e-300)
                out["p_nn"][r] = rd["p_nn"].numpy()
                out["failed"][r] = rd["failed"].numpy()
                out["wave_bad"][r] = rd["wave_order_bad"].numpy()
                out["resid_nn"][r] = rd["resid_nn"].numpy()
                out["p_ws"][r] = rw["p_used"].numpy()
                out["ws_conv"][r] = (rw["err"] == 0).numpy()
                out["ws_iter"][r] = rw["n_iter"].numpy()
                out["flux_err"][r] = ((FA - FH).abs().max(1).values / scale).numpy()
                out["nonfinite"][r] = (~torch.isfinite(FA).all(1)).numpy()
    return lo, out


def in_archived_box(z, n):
    """Faces whose BOTH states lie in default-mignone.yaml's sampling box."""
    ok = np.ones(n, bool)
    idir = z["idir"][:n].astype(int)
    for side in ("prim_L", "prim_R"):
        P = z[side][:n]
        rho, vx, vy, vz, p = P[:, 0], P[:, 1], P[:, 2], P[:, 3], P[:, 4]
        B = P[:, 6:9]
        W = 1.0 / np.sqrt(np.clip(1.0 - vx ** 2 - vy ** 2 - vz ** 2, 1e-300, None))
        bn = np.abs(B[np.arange(n), idir])
        bt = np.sqrt(np.clip((B ** 2).sum(1) - bn ** 2, 0, None))
        ok &= ((W <= 2.0) & (p >= 0.1) & (p <= 1.0) & (rho >= 0.1)
               & (rho <= 10.0) & (bn >= 0.01) & (bn <= 1.01) & (bt <= 1.0))
    return ok


def summarise(lab, res, sel):
    """Metrics on the faces `sel` (bool mask)."""
    v = lab["valid"] & sel
    out = dict(n_rows=int(sel.sum()), n_hlld_valid=int(v.sum()),
               hlld_valid_frac=float(lab["valid"][sel].mean()) if sel.any() else float("nan"))
    if sel.any():
        out["valid_frac_all_rows"] = float((~res["failed"][sel]).mean())
    if not v.any():
        return out
    p = lab["p_star"][v]
    rel = np.abs(res["p_nn"][v] - p) / p
    same = res["ws_conv"][v] & (np.abs(res["p_ws"][v] - p) <= 1e-8 * p)
    q = lambda x, a: float(np.quantile(x, a))
    out.update(
        rel_median=float(np.median(rel)), rel_p90=q(rel, 0.9),
        rel_p99=q(rel, 0.99), rel_max=float(rel.max()),
        rel_mean=float(rel.mean()),
        valid_frac_on_hlld_valid=float((~res["failed"][v]).mean()),
        wave_order_bad_on_hlld_valid=float(res["wave_bad"][v].mean()),
        ws_same_root=float(same.mean()),
        ws_other_root=float((res["ws_conv"][v] & ~same).mean()),
        ws_not_converged=float((~res["ws_conv"][v]).mean()),
        ws_mean_iter=(float(res["ws_iter"][v][res["ws_conv"][v]].mean())
                      if res["ws_conv"][v].any() else float("nan")),
        hlld_mean_iter=float(lab["n_iter"][v].mean()),
        flux_err_median=float(np.median(res["flux_err"][v])),
        flux_err_p99=q(res["flux_err"][v], 0.99),
        flux_err_max=float(res["flux_err"][v].max()),
        resid_nn_median=float(np.median(res["resid_nn"][v])),
    )
    out["ai_flux_nonfinite_all_rows"] = int(res["nonfinite"][sel].sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--testset", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=20_000)
    ap.add_argument("--batch", type=int, default=20_000)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--save-rows", default=None,
                    help="directory for the per-face results (npz per pair)")
    a = ap.parse_args()
    t0 = time.time()
    models = [parse_model(s) for s in a.model]
    tsets = [dict(zip(("name", "path"), s.split("=", 1))) for s in a.testset]
    report = dict(models=models, testsets=tsets, results={})
    import multiprocessing as mp
    for ts in tsets:
        z = np.load(ts["path"])
        _DATA.clear()
        _DATA.update({k: z[k] for k in ("prim_L", "prim_R", "gamma", "idir")})
        _DATA["_path"] = ts["path"]
        # forked AFTER the load, so every worker shares _DATA
        pool = mp.get_context("fork").Pool(a.workers) if a.workers > 1 else None
        n = z["valid"].size if a.max_rows is None else min(a.max_rows, z["valid"].size)
        lab = {k: z[k][:n] for k in ("p_star", "valid", "bn_zero", "n_iter",
                                      "gamma", "idir", "comp") if k in z.files}
        box = in_archived_box(z, n)
        subsets = {"all": np.ones(n, bool), "in_archived_box": box}
        if "comp" in lab and (lab["comp"] >= 0).any():
            for c, name in enumerate(COMPONENTS):
                subsets[f"comp={name}"] = lab["comp"] == c
        for g in np.unique(lab["gamma"]):
            subsets[f"gamma={g:.4f}"] = np.isclose(lab["gamma"], g)
        for d in np.unique(lab["idir"]):
            subsets[f"idir={int(d)}"] = lab["idir"] == d
        for m in models:
            jobs = [(m, ts["path"], lo, min(lo + a.chunk, n), a.batch)
                    for lo in range(0, n, a.chunk)]
            parts = sorted(pool.map(_task, jobs) if pool else
                           [_task(j) for j in jobs], key=lambda t: t[0])
            res = {k: np.concatenate([p[1][k] for p in parts]) for k in RES_KEYS}
            key = f"{m['name']}|{ts['name']}"
            report["results"][key] = {s: summarise(lab, res, sel)
                                       for s, sel in subsets.items()}
            if a.save_rows:
                os.makedirs(a.save_rows, exist_ok=True)
                np.savez(os.path.join(a.save_rows, f"{m['name']}__{ts['name']}.npz"),
                         **res)
            r = report["results"][key]["all"]
            print(f"  {key:40s} n={n}  rel med {r.get('rel_median', np.nan):.2e} "
                  f"p99 {r.get('rel_p99', np.nan):.2e} max {r.get('rel_max', np.nan):.2e}  "
                  f"valid {r.get('valid_frac_on_hlld_valid', np.nan):.4f}  "
                  f"same-root {r.get('ws_same_root', np.nan):.4f}  "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        if pool:
            pool.close()
            pool.join()
    report["wall_s"] = round(time.time() - t0, 1)
    json.dump(report, open(a.out, "w"), indent=1)
    print(f"== {a.out}  ({report['wall_s']}s)")


if __name__ == "__main__":
    main()
