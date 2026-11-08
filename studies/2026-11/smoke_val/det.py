"""Deterministic default-path check: torch seeded, 2 epochs, same data.
    PYTHONPATH=<tree> python det.py OUT.pt"""
import sys

import torch

torch.manual_seed(0)
torch.set_num_threads(2)
from rmhd import ml_guess as mg  # noqa: E402

m0, s0 = mg.load("/Users/miler/Codes/rmhd_final/data/ml_guess_rotor_ft.pt")
mdl, scl = mg.train_from_dataset(
    "/Users/miler/Codes/claude_scratch/r2d_retrain/smoke_audit/it3/lost_it2.npz",
    epochs=2, batch_size=64, lr=2e-4, val_frac=0.1, seed=42, device="cpu",
    init_model=m0, init_scaler=s0, verbose=False)
torch.save({k: v.clone() for k, v in mdl.state_dict().items()}, sys.argv[1])
print(mg.__file__)
