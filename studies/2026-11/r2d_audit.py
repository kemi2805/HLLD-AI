#!/usr/bin/env python
"""Audit of the four-quadrant retrain data (2026-11-03; user: "Are we only
using proper training and validation data for the Q4 ml?  Did we check the
eos and all the states?  How many datapoints do we have").

    python r2d_audit.py RETRAIN_DIR [--chunk 100000] [--json OUT.json]

RETRAIN_DIR = results/r2d_retrain (it1/, it2/, it3/).  Per data file:
  rows, unique problems (the (U_L, U_R) pair bit for bit), gamma;
  every state finite and physical -- U_L, U_R and the eight zones R1..R8:
  rho > 0, p > 0, v^2 < 1; R1 == U_L and R8 == U_R; one B_n on both sides;
  the seven speeds finite, inside the light cone and in order;
  the stored solution re-verified: its six unknowns (raw frame, the frame the
  row was produced in) put into the batched seven-wave residual (fullfuncv6,
  what the trainer's proxy and production's verification use), max|f| per row.
Leakage: problems shared bit for bit between the training set and the
held-out set, and every held-out face's nearest training problem in the
network's own normalised features, by distance from the held-out windows'
centres (a time split leaks at its edges); for scale, the nearest-neighbour
spacing inside the training set.  Checkpoints: the gamma each was trained for.
"""
import argparse
import json
import os

import numpy as np

from rmhd import ml_guess as mg
from rmhd.batched import fullcontact_b as FB
from rmhd.batched.ml_b import canonical_features

WINDOWS, DT = (0.175, 0.325), 0.0125


def wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def keys16(UL, UR):
    a = np.ascontiguousarray(np.concatenate([UL, UR], axis=1), dtype=np.float64)
    return a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1]))).ravel()


def rotation(Z):
    ang = lambda R: np.arctan2(R[:, 6], R[:, 5])
    return np.abs(wrap(ang(Z[:, 2]) - ang(Z[:, 1]))) + np.abs(wrap(ang(Z[:, 6]) - ang(Z[:, 5])))


def physical(S):
    """(N, >=5) states [rho, p, vx, vy, vz, ...] -> bool per row."""
    v2 = S[:, 2] ** 2 + S[:, 3] ** 2 + S[:, 4] ** 2
    return np.isfinite(S).all(axis=1) & (S[:, 0] > 0) & (S[:, 1] > 0) & (v2 < 1.0)


def gate_crossed(Z, S, tol):
    """HLLD exact_flux._self_crossing on stored rows: a rotation (LA, RA) may
    sit out of order with its slow wave only if it carries no jump in rho or
    v; every other adjacent pair must be ordered."""
    o = [S[:, j] for j in range(7)]

    def strength(a, b):                      # zones index: R2 = 1 ... R7 = 6
        rho = np.maximum(np.abs(Z[:, a, 0]), 1e-30)
        return np.maximum(np.abs(Z[:, a, 0] - Z[:, b, 0]) / rho,
                          np.sqrt(sum((Z[:, a, k] - Z[:, b, k]) ** 2 for k in (2, 3, 4))))

    c = (o[2] - o[1] < -tol) & (strength(1, 2) > 1e-8)
    c |= (o[5] - o[4] < -tol) & (strength(5, 6) > 1e-8)
    for k in (0, 2, 3, 5):
        c |= o[k + 1] - o[k] < -tol
    return c


def residual(UL, UR, Z, W, chunk):
    """max|f| of the seven-wave residual at the stored solution, per row."""
    out = np.empty(UL.shape[0])
    bad = np.zeros(UL.shape[0], bool)
    for i in range(0, UL.shape[0], chunk):
        s = slice(i, i + chunk)
        R2, R3, R4, R6, R7 = Z[s, 1], Z[s, 2], Z[s, 3], Z[s, 5], Z[s, 6]
        ang = lambda R: np.arctan2(R[:, 6], R[:, 5])
        u6 = [np.log(R2[:, 1]), np.log(np.maximum(np.hypot(R4[:, 5], R4[:, 6]), 1e-30)), ang(R4),
              np.log(R6[:, 1]), wrap(ang(R3) - ang(R2)), wrap(ang(R7) - ang(R6))]
        left = [np.ascontiguousarray(UL[s, j]) for j in range(7)]
        right = [np.ascontiguousarray(UR[s, j]) for j in range(7)]
        with np.errstate(all="ignore"):
            f, _, _, _, err = W["fullfuncv6"](left, right, u6, np.ascontiguousarray(UL[s, 7]))
        out[s] = np.where(err, np.inf, np.max(np.abs(f), axis=0))
        bad[s] = np.asarray(err, bool)
    return out, bad


def audit_file(path, W, chunk, do_residual=True):
    z = np.load(path)
    r = dict(file=os.path.relpath(path, os.path.dirname(os.path.dirname(path))), keys=sorted(z.files))
    UL, UR = np.asarray(z["U_L"], float), np.asarray(z["U_R"], float)
    n = UL.shape[0]
    k = keys16(UL, UR)
    r.update(rows=n, unique=int(np.unique(k).size),
             gamma=float(z["gamma"]) if "gamma" in z.files else None)
    r["finite_UL_UR"] = bool(np.isfinite(UL).all() and np.isfinite(UR).all())
    r["unphysical_UL"] = int((~physical(UL)).sum())
    r["unphysical_UR"] = int((~physical(UR)).sum())
    r["Bn_differs"] = int((UL[:, 7] != UR[:, 7]).sum())
    r["trivial_UL_eq_UR"] = int((UL == UR).all(axis=1).sum())
    if "zones" in z.files:
        Z = np.asarray(z["zones"], float)
        S = np.asarray(z["speeds"], float)
        r["unphysical_zone_rows"] = int((~np.all([physical(Z[:, j]) for j in range(8)], axis=0)).sum())
        r["R1_ne_UL"] = int((Z[:, 0] != UL[:, :7]).any(axis=1).sum())
        r["R8_ne_UR"] = int((Z[:, 7] != UR[:, :7]).any(axis=1).sum())
        r["speeds_nonfinite"] = int((~np.isfinite(S)).any(axis=1).sum())
        r["speeds_outside_light_cone"] = int((np.abs(S) >= 1.0).any(axis=1).sum())
        d = np.diff(S, axis=1)
        pairs = ("LF>LA", "LA>LS", "LS>CD", "CD>RS", "RS>RA", "RA>RF")
        r["speeds_inverted_by_pair"] = {pairs[j]: int((d[:, j] < -1e-6).sum()) for j in range(6)}
        r["speeds_worst_inversion"] = float(-min(0.0, np.nanmin(d)))
        r["gate_rule_crossed"] = int(gate_crossed(Z, S, 1e-6).sum())
        rot = rotation(Z)
        r["complex_rot_gt_0p5"] = int((rot > 0.5).sum())
        r["Bt_CD_zero"] = int((np.hypot(Z[:, 3, 5], Z[:, 3, 6]) == 0).sum())
        if not do_residual:
            r["residual"] = "skipped: every row is in it3/train3.npz or it1/eval.npz"
            return r, k
        _, u = np.unique(k, return_index=True)        # one evaluation per problem
        res, err = residual(UL[u], UR[u], Z[u], W, chunk)
        fin = res[np.isfinite(res)]
        r["residual"] = dict(problems=int(u.size),
            error_flag=int(err.sum()), le_1e_10=int((res <= 1e-10).sum()), le_1e_8=int((res <= 1e-8).sum()),
            le_1e_6=int((res <= 1e-6).sum()), gt_1e_6=int((res > 1e-6).sum()),
            median=float(np.median(fin)) if fin.size else None,
            p999=float(np.quantile(fin, 0.999)) if fin.size else None,
            max=float(fin.max()) if fin.size else None)
    if "source" in z.files:
        src = np.asarray(z["source"])
        r["source_counts"] = {int(s): int((src == s).sum()) for s in np.unique(src)}
    if "t" in z.files:
        t = np.asarray(z["t"], float)
        r["t_range"] = [float(t.min()), float(t.max())]
    return r, k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--chunk", type=int, default=100_000)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    D = a.dir
    files = ["it1/all.npz", "it1/train.npz", "it1/eval.npz", "it1/proxy.npz", "it2/perturb.npz",
             "it2/lost_net.npz", "it2/train2.npz", "it3/lost_it2.npz", "it3/newly_won.npz", "it3/train3.npz"]
    W = FB.make_waves(5.0 / 3.0)
    out, K = {}, {}
    for f in files:
        p = os.path.join(D, f)
        if not os.path.exists(p):
            print("missing", f)
            continue
        r, k = audit_file(p, W, a.chunk, do_residual=f not in ("it1/all.npz", "it1/train.npz",
                                                                    "it2/train2.npz", "it2/perturb.npz"))
        out[f], K[f] = r, k
        print(json.dumps(r), flush=True)

    # --- leakage between the training set and the held-out set
    need = ("it3/train3.npz", "it1/eval.npz", "it1/proxy.npz", "it1/all.npz")
    if not all(f in K for f in need):
        print("leakage step skipped: needs", ", ".join(need))
        main_ck(D, a, out, {})
        return
    ktr, kev = np.unique(K["it3/train3.npz"]), np.unique(K["it1/eval.npz"])
    kpx = np.unique(K["it1/proxy.npz"])
    lk = dict(train3_unique=int(ktr.size), eval_unique=int(kev.size),
              shared_train3_eval=int(np.intersect1d(ktr, kev).size),
              shared_train3_proxy=int(np.intersect1d(ktr, kpx).size),
              proxy_in_eval=int(np.intersect1d(kpx, kev).size))
    ka = K["it1/all.npz"]
    lk["all_unique"] = int(np.unique(ka).size)
    print(json.dumps(lk), flush=True)

    # nearest training problem of every held-out face, in the network's features
    try:
        from scipy.spatial import cKDTree
        _, scaler = mg.load(os.path.join(D, "it3/ml_guess_r2d_it3.pt"))
        xm = np.asarray(scaler.x_mean.cpu(), float)
        xs = np.asarray(scaler.x_std.cpu(), float)

        def feats(path):
            z = np.load(path)
            UL, UR = np.asarray(z["U_L"], float), np.asarray(z["U_R"], float)
            _, idx = np.unique(keys16(UL, UR), return_index=True)
            F = canonical_features(UL[idx, :7], UR[idx, :7], UL[idx, 7])[0]
            return (np.asarray(F, float) - xm) / xs, idx, z

        Ftr, _, _ = feats(os.path.join(D, "it3/train3.npz"))
        Fev, iev, zev = feats(os.path.join(D, "it1/eval.npz"))
        tree = cKDTree(Ftr)
        dev = tree.query(Fev, k=1, workers=-1)[0]
        rng = np.random.default_rng(0)
        smp = rng.choice(Ftr.shape[0], size=min(20000, Ftr.shape[0]), replace=False)
        dtr = tree.query(Ftr[smp], k=2, workers=-1)[0][:, 1]
        t = np.asarray(zev["t"], float)[iev]
        dc = np.min([np.abs(t - w) for w in WINDOWS], axis=0)
        nn = dict(features=int(Ftr.shape[1]),
                  train_spacing=dict(median=float(np.median(dtr)), p10=float(np.quantile(dtr, 0.1))))
        for lo, hi in ((0.0, DT / 3), (DT / 3, 2 * DT / 3), (2 * DT / 3, DT)):
            m = (dc >= lo) & (dc < hi)
            dd = dev[m]
            nn["eval |t-centre| %.4f-%.4f" % (lo, hi)] = dict(
                n=int(m.sum()), exact_dup=int((dd == 0).sum()),
                below_1e_3=int((dd < 1e-3).sum()), below_1e_2=int((dd < 1e-2).sum()),
                median=float(np.median(dd)), p10=float(np.quantile(dd, 0.1)))
        lk["nearest"] = nn
        print(json.dumps(nn), flush=True)
    except ImportError as e:
        print("nearest-neighbour step skipped:", e)

    main_ck(D, a, out, lk)


def main_ck(D, a, out, lk):
    # --- the gamma each network was trained for
    ck = {}
    for c in ("it1/ml_guess_r2d_ft1.pt", "it2/ml_guess_r2d_it2.pt", "it2/ml_guess_r2d_it2_proxy.pt",
              "it3/ml_guess_r2d_it3.pt"):
        p = os.path.join(D, c)
        if os.path.exists(p):
            ck[c] = float(mg.load(p)[1].gamma)
    print(json.dumps(dict(checkpoint_gamma=ck)), flush=True)
    if a.json:
        json.dump(dict(files=out, leakage=lk, checkpoint_gamma=ck), open(a.json, "w"), indent=1)
        print("wrote", a.json)


if __name__ == "__main__":
    main()
