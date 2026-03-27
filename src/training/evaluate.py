"""
Evaluation and error analysis on the test set.

Usage:
    python -m src.training.evaluate --config configs/default.yaml
"""

import argparse
import os
import numpy as np
import torch
import yaml

from src.data.dataset import HLLDDataset
from src.models.network import PressureNet


def evaluate(cfg: dict):
    device = torch.device(cfg["training"].get("device", "cpu"))

    # Load model
    model = PressureNet(
        n_input=cfg["model"].get("n_input", 15),
        hidden=cfg["model"].get("hidden", [128, 128, 64]),
        activation=cfg["model"].get("activation", "silu"),
    ).to(device)
    checkpoint_path = cfg["_checkpoint_path"]
    print("The checkpoint path is",checkpoint_path)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Load test data
    ds = HLLDDataset(os.path.join(cfg["data"]["split_dir"], "test.npz"))
    X = ds.X.to(device)
    y_norm_true = ds.y.numpy().flatten()

    with torch.no_grad():
        y_norm_pred = model(X).cpu().numpy().flatten()

    # Inverse-transform to physical units using saved norm stats
    stats_path = os.path.join("data/processed", cfg["_config_stem"], "norm_stats.npz")
    norm_stats = np.load(stats_path)
    y_mu    = float(norm_stats["y_mu"])
    y_sigma = float(norm_stats["y_sigma"])
    y_true = 10.0 ** (y_norm_true * y_sigma + y_mu)
    y_pred = 10.0 ** (y_norm_pred * y_sigma + y_mu)

    # Metrics in physical units
    mse = np.mean((y_pred - y_true) ** 2)
    rmse = np.sqrt(mse)
    rel_err = np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-12)
    max_rel = rel_err.max()
    mean_rel = rel_err.mean()
    median_rel = np.median(rel_err)

    print(f"Test MSE       : {mse:.6e}")
    print(f"Test RMSE      : {rmse:.6e}")
    print(f"Mean  rel error: {mean_rel:.4%}")
    print(f"Median rel error: {median_rel:.4%}")
    print(f"Max   rel error: {max_rel:.4%}")

    # Save predictions for plotting
    results_dir = cfg.get("_results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "test_predictions.npz")
    np.savez(out_path, y_true=y_true, y_pred=y_pred, rel_err=rel_err)
    print(f"Predictions saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    config_stem = os.path.splitext(os.path.basename(args.config))[0]
    cfg["data"]["split_dir"] = os.path.join("data/splits", config_stem)
    cfg["_checkpoint_path"] = os.path.join("checkpoints", f"{config_stem}.pt")
    cfg["_results_dir"] = os.path.join("results", config_stem)
    cfg["_config_stem"] = config_stem

    evaluate(cfg)


if __name__ == "__main__":
    main()
