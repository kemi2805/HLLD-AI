"""What predicts a compound interface, and where does one sit?  (science plan
F, step 1c)

Reads the snapshot census (``snapshot_census.py``: every attempted interface
of the rotor's snapshots, with two labels -- COMPOUND from the tube test, and
whether today's solver answers it) and asks which physical quantities of the
interface predict the labels.  Runs on the laptop: the fits are seconds on
~10^4 rows, and scikit-learn lives only in the local venv.

Features, all computable from (L, R, B_n) before any solve, reused rather
than rewritten:

* ``rmhd_final/scripts/failure_map.py::_features`` -- |B_n|, Lorentz factors,
  magnetisation sigma, plasma beta and |B_t| on each side, the log pressure
  and density ratios, dv_x, the angle between the tangential fields;
* the speed gaps from ``rmhd.batched.api.classify_batch``, each relative to
  the fast-wave fan width: ``d_slow`` (Alfven pair to slow pair -- the merger
  compound waves live on), ``d_fast``, ``d_contact``, and ``sin_rot``;
* the field reversal across the interface (``planar5_b.to_planar``: with the
  left tangential field along +y, the right one points along -y);
* the Rankine-Hugoniot defect for one best-fit speed
  (``coplanar_limit.rh_defect``);
* log10 |B_n| / (|B_t,L| + |B_t,R|): the normal field against the tangential
  one, the quantity the failure map's x/y asymmetry points at.

What is fitted, per target (compound; unsolvable; seven-wave fails):
per-feature AUC (``failure_map._auc``); an L1 logistic regression on
standardised features and a depth-3 tree, both cross-validated by snapshot
TIME (GroupKFold), so no time is scored on a model that saw it.  The tree's
rules are printed: a rule of a few physical quantities is the result worth
having; the regression shows which features survive the L1 penalty.

    python scripts/compound_classifier.py results/snapshot_census_64 \
        --run results/rotor_64_exact --maps figs/census
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import warnings

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "scripts"))
from rmhd import paths                                             # noqa: E402
sys.path.insert(0, os.path.join(str(paths.repo_root()), "scripts"))
from rmhd.eos import set_eos                                       # noqa: E402
set_eos("ideal")
import failure_map as FM                                           # noqa: E402
from rmhd.batched import api as API                                # noqa: E402
from rmhd.batched import planar5_b as P5                           # noqa: E402

warnings.filterwarnings("ignore")
np.seterr(all="ignore")
GAMMA = 5.0 / 3.0
LOG = ("sigma_L", "sigma_R", "beta_L", "beta_R", "max_sigma", "d_slow", "d_fast",
       "d_contact", "rh_defect")


def load(d):
    Z = [np.load(f) for f in sorted(glob.glob(os.path.join(d, "shard_*.npz")))]
    if not Z:
        raise SystemExit("no shards in %s" % d)
    keys = ("t", "idir", "fa", "fb", "UL", "UR", "Bn", "prod_ok", "solvable",
            "compound", "btzero")
    return {k: np.concatenate([z[k] for z in Z]) for k in keys}


def features(C):
    UL, UR, Bn = C["UL"], C["UR"], C["Bn"]
    n = Bn.size
    rows = [FM._features(UL[i, :7], UR[i, :7], Bn[i], GAMMA) for i in range(n)]
    names = list(rows[0])
    cols = {k: np.array([r[k] for r in rows]) for k in names}
    left = [UL[:, j].copy() for j in range(7)]
    right = [UR[:, j].copy() for j in range(7)]
    _, info = API.classify_batch(left, right, Bn, GAMMA)
    for k in ("d_contact", "d_fast", "d_slow", "sin_rot"):
        cols[k] = np.asarray(info[k], float)
    _, R0, _, _ = P5.to_planar(left, right, Bn)
    cols["reversal"] = (R0[5] < 0.0).astype(float)
    import coplanar_limit as CLM                    # torch + HLLD src; only here
    _, cols["rh_defect"] = CLM.rh_defect(UL, UR)
    cols["log_Bn_over_Bt"] = np.log10(np.abs(Bn) / (cols["Bt_L"] + cols["Bt_R"] + 1e-30)
                                      + 1e-30)
    for k in LOG:
        cols[k] = np.log10(np.maximum(cols[k], 1e-30))
    names = list(cols)
    X = np.column_stack([cols[k] for k in names])
    X = np.where(np.isfinite(X), X, np.nan)
    med = np.nanmedian(X, axis=0)
    X = np.where(np.isnan(X), med[None, :], X)      # the rare non-finite entry
    return X, names


def auc_table(X, names, y, top=10):
    a = np.array([FM._auc(X[:, j], ~y, y) for j in range(len(names))])
    order = np.argsort(-np.abs(a - 0.5))
    return [(names[j], a[j]) for j in order[:top]]


def fit(X, names, y, groups):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier, export_text
    g = np.unique(groups)
    k = min(5, g.size)
    lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(
        penalty="l1", solver="liblinear", C=0.1, class_weight="balanced"))
    tr = lambda: DecisionTreeClassifier(max_depth=3, class_weight="balanced",
                                        min_samples_leaf=max(30, y.size // 200))
    res = {}
    for name, make in (("L1 logistic", lr), ("depth-3 tree", tr)):
        scores = []
        for tri, tei in GroupKFold(n_splits=k).split(X, y, groups):
            if y[tei].all() or (~y[tei]).all():
                continue
            m = make().fit(X[tri], y[tri])
            scores.append(roc_auc_score(y[tei], m.predict_proba(X[tei])[:, 1]))
        res[name] = (np.mean(scores) if scores else np.nan,
                     np.std(scores) if scores else np.nan, len(scores))
    full_lr = lr().fit(X, y)
    coef = full_lr[-1].coef_[0]
    nz = [(names[j], coef[j]) for j in np.argsort(-np.abs(coef)) if coef[j] != 0]
    full_tr = tr().fit(X, y)
    rules = export_text(full_tr, feature_names=list(names), decimals=3)
    return res, nz, rules


def maps(C, rundir, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out, exist_ok=True)
    snaps = {}
    for f in glob.glob(os.path.join(rundir, "snap_*.npz")):
        if not f.endswith("snap_fin.npz"):
            z = np.load(f)
            snaps[round(float(z["t"]), 3)] = {k: z[k] for k in z.files}
    for t in np.unique(C["t"]):
        sn = snaps.get(round(float(t), 3))
        if sn is None:
            continue
        x, y = sn["x"], sn["y"]
        dx, dy = x[1] - x[0], y[1] - y[0]
        fig, ax = plt.subplots(1, 2, figsize=(12.5, 6.0))
        for d, lab in ((0, "x-faces"), (1, "y-faces")):
            m = (C["t"] == t) & (C["idir"] == d)
            fa, fb = C["fa"][m], C["fb"][m]
            px = x[0] - 0.5 * dx + fa * dx if d == 0 else x[fa]
            py = y[fb] if d == 0 else y[0] - 0.5 * dy + fb * dy
            cp, sv = C["compound"][m], C["solvable"][m]
            a = ax[d]
            a.pcolormesh(x, y, np.log10(sn["rho"]).T, cmap="Greys", shading="auto")
            a.scatter(px[sv & ~cp], py[sv & ~cp], s=4, c="#9ca3af", label="solved")
            a.scatter(px[~sv & ~cp], py[~sv & ~cp], s=10, c="#2563eb",
                      label="unsolved, elementary")
            a.scatter(px[cp], py[cp], s=12, c="#dc2626", label="compound")
            a.set_aspect("equal")
            a.set_title("%s, t = %.2f: %d attempted, %d compound, %d unsolved"
                        % (lab, t, m.sum(), cp.sum(), (~sv).sum()), fontsize=10)
            if d == 0:
                a.legend(loc="upper right", fontsize=8, markerscale=1.5)
        fig.tight_layout()
        f = os.path.join(out, "census_t%.2f.png" % t)
        fig.savefig(f, dpi=120)
        plt.close(fig)
        print("wrote", f)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("census")
    ap.add_argument("--run", help="the rotor run whose snapshots the census used "
                                  "(for --maps)")
    ap.add_argument("--maps", help="write the per-time maps here")
    a = ap.parse_args()
    C = load(a.census)
    n = C["t"].size
    print("census %s: %d attempted interfaces, %d times" % (a.census, n,
                                                            np.unique(C["t"]).size))
    X, names = features(C)
    targets = (("compound (tube)", C["compound"]),
               ("unsolvable (neither seven-wave nor planar)", ~C["solvable"]),
               ("seven-wave fails", ~C["prod_ok"]))
    u, k = ~C["solvable"], C["compound"]
    print("base rates: compound %.1f%%, unsolvable %.1f%%, seven-wave fails %.1f%%;"
          " P(compound | unsolvable) %.1f%%, P(unsolvable | compound) %.1f%%"
          % (100 * k.mean(), 100 * u.mean(), 100 * (~C["prod_ok"]).mean(),
             100 * k[u].mean() if u.any() else np.nan,
             100 * u[k].mean() if k.any() else np.nan))
    for lab, y in targets:
        y = np.asarray(y, bool)
        print("\n== target: %s  (%d of %d)" % (lab, y.sum(), n))
        if y.sum() < 20 or (~y).sum() < 20:
            print("   too few of one class to fit")
            continue
        print("   single-feature AUC (0.5 = no signal):")
        for nm, v in auc_table(X, names, y):
            print("     %-16s %.3f" % (nm, v))
        res, nz, rules = fit(X, names, y, C["t"])
        for m, (mu, sd, kk) in res.items():
            print("   %-13s cross-validated by time: AUC %.3f +- %.3f (%d folds)"
                  % (m, mu, sd, kk))
        print("   L1 logistic, non-zero standardised coefficients:")
        for nm, c in nz[:10]:
            print("     %-16s %+.3f" % (nm, c))
        print("   depth-3 tree:")
        print("\n".join("     " + ln for ln in rules.splitlines()))
    if a.maps:
        if not a.run:
            raise SystemExit("--maps needs --run (the snapshots)")
        maps(C, a.run, a.maps)


if __name__ == "__main__":
    main()
