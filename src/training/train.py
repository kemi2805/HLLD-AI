"""
Training loop.

Usage:
    python -m src.training.train --config configs/default.yaml
"""

import argparse
import os
import torch
import yaml

from src.data.dataset import get_dataloaders
from src.models.network import PressureNet
from src.models.losses import PhysicsInformedLoss, PressureLoss, RelativeMSELoss


def train(cfg: dict):
    device = torch.device(cfg["training"].get("device", "cpu"))

    # Data
    loaders = get_dataloaders(
        cfg["data"]["split_dir"],
        batch_size=cfg["training"]["batch_size"],
        device=cfg["training"].get("device", "cpu"),
    )

    # Model
    model = PressureNet(
        n_input=cfg["model"].get("n_input", 15),
        hidden=cfg["model"].get("hidden", [128, 128, 64]),
        activation=cfg["model"].get("activation", "silu"),
        dropout=cfg["model"].get("dropout", 0.0),
    ).to(device)

    # Loss + optimizer
    #criterion = PhysicsInformedLoss(
    #    lambda_phys=cfg["training"].get("lambda_phys", 0.1)
    #)
    criterion = PressureLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"].get("weight_decay", 0.0),
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=cfg["training"].get("lr_patience", 10),
        factor=cfg["training"].get("lr_factor", 0.5),
        min_lr=cfg["training"].get("lr_floor", 1e-5)
    )
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #    optimizer,
    #    T_max=cfg["training"].get("T_max", 300),      # Cycle length in epochs — try 200-500 for 1000 epoch runs
    #    eta_min=cfg["training"].get("eta_min", 1.e-6),   # Floor LR — don't let it die completely
    #)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #    optimizer,
    #    T_0=cfg["training"].get("T_max", 200),       # restart every 200 epochs
    #    T_mult=cfg["training"].get("T_mult", 1),      # keep same cycle length
    #    eta_min=cfg["training"].get("eta_min", 1.e-6),
    #)
    #scheduler = torch.optim.lr_scheduler.CyclicLR(
    #    optimizer,
    #    base_lr=cfg["training"].get("lr_min", 1e-7),
    #    max_lr=cfg["training"].get("lr_max", 5e-3),
    #    step_size_up=cfg["training"].get("lr_step_size_up", 5),
    #    step_size_down=cfg["training"].get("lr_step_size_down", 10),
    #    mode="triangular2",  # decays peak each cycle — good for convergence
    #    cycle_momentum=False,  # set True only if using SGD
    #)



    # Training loop
    best_val_loss = float("inf")
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for X, y in loaders["train"]:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(X)
        train_loss /= len(loaders["train"].dataset)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in loaders["val"]:
                X, y = X.to(device), y.to(device)
                val_loss += criterion(model(X), y).item() * len(X)
        val_loss /= len(loaders["val"].dataset)

        scheduler.step(val_loss)
        #scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}  train={train_loss:.6f}  val={val_loss:.6f}")

        # Checkpoint best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), cfg["_checkpoint_path"])

    print(f"Training complete. Best val loss: {best_val_loss:.6f}")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    config_stem = os.path.splitext(os.path.basename(args.config))[0]
    #cfg["data"]["split_dir"] = os.path.join("data/splits", config_stem)
    cfg["_checkpoint_path"] = os.path.join("checkpoints", f"{config_stem}.pt")

    train(cfg)


if __name__ == "__main__":
    main()
