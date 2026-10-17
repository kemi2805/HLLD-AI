"""Train PressureNet on a split built by scripts/pstar_build_splits.py.

    python -u scripts/pstar_train.py --split SPLITS/v2-idir-abs-bn \
        --out checkpoints/pstar_union_v2.pt --device cuda --seed 0

The optimiser is the archived run's (configs/default-mignone.yaml: Adam,
weight decay 1e-4, ReduceLROnPlateau patience 10 / factor 0.5 / floor 1e-5,
MSE on standardised log10 p*, gradient clipping at 1, best-validation
checkpoint) -- only the data loading differs: the whole split lives on the
device and each epoch is a device-side permutation.  On Goethe's MI210 nodes
the host cores run at ~400 MHz under load, so a CPU DataLoader, or a small
batch's per-step launch overhead, would set the pace; hence the default
batch of 32768 (the archived run used 4096).

Writes <out> (a plain state_dict, loadable by driver._load_ai_solver like
the archived checkpoints), <out>.json (arguments, per-epoch losses, final
validation metrics in physical units) and copies norm_stats.npz next to it
as <out stem>_norm_stats.npz, so a checkpoint always travels with the
statistics it was trained on.
"""
import argparse
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.models.network import PressureNet


def _save_atomic(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def evaluate(model, X, y, batch=262144):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch):
            out.append(model(X[i:i + batch]).squeeze(-1))
    pred = torch.cat(out)
    return float(torch.mean((pred - y) ** 2)), pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, help="SPLITS/<feature version>")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", default="256,256,256,256,128,64")
    ap.add_argument("--activation", default="silu")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=32768)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--factor", type=float, default=0.5)
    ap.add_argument("--lr-floor", type=float, default=1e-5)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--max-train", type=int, default=None,
                    help="use only the first N training rows (pilots)")
    ap.add_argument("--time-limit", type=float, default=None,
                    help="stop after this many hours, keeping the best")
    ap.add_argument("--threads", type=int, default=None,
                    help="torch CPU threads (default: torch's own choice)")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device(a.device)
    t0 = time.time()
    load = lambda s, mm=True: np.load(os.path.join(a.split, s),
                                      mmap_mode="r" if mm else None)
    Xtr = load("X_train.npy")
    ytr = load("y_train.npy")
    if a.max_train:
        Xtr, ytr = Xtr[: a.max_train], ytr[: a.max_train]
    Xtr = torch.from_numpy(np.array(Xtr)).to(dev)
    ytr = torch.from_numpy(np.array(ytr)).to(dev)
    Xva = torch.from_numpy(load("X_val.npy", False)).to(dev)
    yva = torch.from_numpy(load("y_val.npy", False)).to(dev)
    stats = dict(np.load(os.path.join(a.split, "norm_stats.npz")))
    version = str(stats["feature_version"])
    n, d = Xtr.shape
    hidden = [int(h) for h in a.hidden.split(",") if h]
    print(f"== {a.split}  version={version}  train={n} val={Xva.shape[0]} "
          f"inputs={d}  hidden={hidden}  device={dev}  "
          f"load {time.time()-t0:.0f}s", flush=True)

    model = PressureNet(n_input=d, hidden=hidden, activation=a.activation).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr,
                           weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=a.patience, factor=a.factor, min_lr=a.lr_floor)
    lossf = torch.nn.MSELoss()

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    stem = os.path.splitext(a.out)[0]
    shutil.copyfile(os.path.join(a.split, "norm_stats.npz"),
                    stem + "_norm_stats.npz")
    log = dict(args=vars(a), version=version, n_train=n,
               n_val=int(Xva.shape[0]), n_params=sum(
                   p.numel() for p in model.parameters()), epochs=[])
    best = float("inf")
    best_epoch = 0
    for ep in range(1, a.epochs + 1):
        te = time.time()
        model.train()
        perm = torch.randperm(n, device=dev)
        tot = torch.zeros((), device=dev)
        for i in range(0, n, a.batch):
            idx = perm[i:i + a.batch]
            pred = model(Xtr[idx]).squeeze(-1)
            loss = lossf(pred, ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if a.clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
            opt.step()
            tot += loss.detach() * idx.numel()
        train_loss = float(tot) / n
        val_loss, _ = evaluate(model, Xva, yva)
        sched.step(val_loss)
        lr = opt.param_groups[0]["lr"]
        improved = val_loss < best
        if improved:
            best, best_epoch = val_loss, ep
            _save_atomic(model.state_dict(), a.out)
        log["epochs"].append(dict(epoch=ep, train=train_loss, val=val_loss,
                                  lr=lr, s=round(time.time() - te, 2)))
        if ep <= 3 or ep % 10 == 0 or ep == a.epochs:
            print(f"epoch {ep:4d}  train={train_loss:.6e}  val={val_loss:.6e}  "
                  f"lr={lr:.2e}  {time.time()-te:.1f}s{'  *' if improved else ''}",
                  flush=True)
        if a.time_limit and time.time() - t0 > 3600 * a.time_limit:
            print(f"time limit at epoch {ep}", flush=True)
            break

    # final metrics of the BEST checkpoint, in physical units
    model.load_state_dict(torch.load(a.out, map_location=dev))
    val_loss, pred = evaluate(model, Xva, yva)
    ys = float(stats["y_sigma"]); ym = float(stats["y_mu"])
    p_t = 10.0 ** (yva.double() * ys + ym)
    p_p = 10.0 ** (pred.double() * ys + ym)
    rel = ((p_p - p_t).abs() / p_t).cpu().numpy()
    log.update(best_val=best, best_epoch=best_epoch, final_train=train_loss,
               val_rel_err=dict(median=float(np.median(rel)),
                                p90=float(np.quantile(rel, 0.9)),
                                p99=float(np.quantile(rel, 0.99)),
                                max=float(rel.max())),
               wall_s=round(time.time() - t0, 1))
    json.dump(log, open(a.out + ".json", "w"), indent=1)
    print(f"== best val {best:.6e} at epoch {best_epoch}; val rel err "
          f"median {log['val_rel_err']['median']:.3e} p99 "
          f"{log['val_rel_err']['p99']:.3e} max {log['val_rel_err']['max']:.3e}; "
          f"wall {log['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
