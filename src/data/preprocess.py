"""
Data preprocessing and splitting.

Loads raw data, applies normalization, and creates train/val/test splits.

Usage:
    python -m src.data.preprocess --config configs/default.yaml
"""

import argparse
import numpy as np
import os
import yaml


def load_raw(path: str):
    data = np.load(path)
    return data["inputs"], data["targets"]


def normalize(X: np.ndarray, method: str = "standard"):
    """
    Normalize features. Returns normalized data + stats for inverse transform.
    """
    if method == "standard":
        mu = X.mean(axis=0)
        sigma = X.std(axis=0) + 1e-12
        return (X - mu) / sigma, {"mu": mu, "sigma": sigma}
    elif method == "minmax":
        lo = X.min(axis=0)
        hi = X.max(axis=0) + 1e-12
        return (X - lo) / (hi - lo), {"min": lo, "max": hi}
    elif method == "log":
        return np.log1p(np.abs(X)) * np.sign(X), {}
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def split(X, y, train_frac=0.8, val_frac=0.1, seed=42):
    """Shuffle and split into train / val / test."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_train = int(len(X) * train_frac)
    n_val = int(len(X) * val_frac)

    return {
        "train": (X[idx[:n_train]], y[idx[:n_train]]),
        "val": (X[idx[n_train:n_train + n_val]], y[idx[n_train:n_train + n_val]]),
        "test": (X[idx[n_train + n_val:]], y[idx[n_train + n_val:]]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    X, y = load_raw(cfg["data"]["raw_path"])

    # Targets that HLLD could not produce.  hlld_flux writes
    # p_star = -|p_hll| - 1e-10 exactly where it FAILED (root-find did not
    # converge, wave ordering unphysical, or B_n = 0), so a non-positive
    # target is not noise -- it is a label saying "the classical solver could
    # not solve this configuration".
    #
    # These are still excluded from the regression (log10 needs y > 0), but
    # they are now COUNTED and REPORTED rather than silently discarded.  The
    # distinction matters: dropping them quietly biases the training set away
    # from precisely the hardest configurations, so the network ends up
    # weakest exactly where HLLD already struggles -- and nothing in a loss
    # curve reveals that.  The reported fraction is also the honest answer to
    # "on what fraction of interfaces is the network extrapolating?".
    bad = ~np.isfinite(y)
    nonpos = y <= 0
    mask = bad | nonpos
    n_tot = len(y)
    if mask.any():
        print(f"HLLD-unsolvable targets: {int(mask.sum())}/{n_tot} "
              f"({100*mask.mean():.2f}%)  "
              f"[{int(bad.sum())} non-finite, {int(nonpos.sum())} non-positive]")
        print("  -> excluded from the regression loss, retained as a "
              "held-out 'HLLD failure' set")
        np.savez_compressed(
            os.path.splitext(cfg["data"]["raw_path"])[0] + "_failed.npz",
            X=X[mask], y=y[mask], frac=float(mask.mean()))
        X, y = X[~mask], y[~mask]
    else:
        print(f"HLLD-unsolvable targets: 0/{n_tot}")

    X_norm, stats = normalize(X, method=cfg["data"].get("norm_method", "standard"))

    # Normalize targets: log10 is natural for pressure (positive, wide range)
    y_log = np.log10(y)
    y_mu, y_sigma = y_log.mean(), y_log.std() + 1e-12
    y_norm = (y_log - y_mu) / y_sigma

    splits = split(X_norm, y_norm, seed=cfg["data"].get("seed", 42))

    #config_stem = os.path.splitext(os.path.basename(args.config))[0]
    #out_dir = os.path.join("data/splits", config_stem)
    out_dir = cfg["data"]["split_dir"]
    os.makedirs(out_dir, exist_ok=True)
    for name, (Xi, yi) in splits.items():
        np.savez(os.path.join(out_dir, f"{name}.npz"), inputs=Xi, targets=yi)

    #processed_dir = os.path.join("data/processed", config_stem)
    processed_dir = os.path.join(os.path.dirname(out_dir), "processed", os.path.basename(out_dir))
    os.makedirs(processed_dir, exist_ok=True)
    # Stamp the feature definition into the stats file.  driver._load_ai_solver
    # refuses a checkpoint whose stamp does not match the running code -- a
    # feature-set change is otherwise undetectable at run time, since the
    # network will consume any 15 numbers and return a plausible p*.
    from src.physics.ai_features import FEATURE_VERSION
    np.savez(os.path.join(processed_dir, "norm_stats.npz"),
             **stats,
             y_mu=np.array(y_mu), y_sigma=np.array(y_sigma),
             feature_version=np.array(FEATURE_VERSION))
    print(f"Target log10(p_tot*): mean={y_mu:.3f}, std={y_sigma:.3f}")
    print(f"Splits saved: {[f'{k}: {len(v[0])}' for k, v in splits.items()]}")


if __name__ == "__main__":
    main()
