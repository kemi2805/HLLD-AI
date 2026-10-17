"""
PyTorch Dataset / DataLoader definitions.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class HLLDDataset(Dataset):
    """
    Dataset for HLLD total pressure prediction.

    Each sample:
        x : (19,)  — left + right R-vectors + velocities + shared Bx
                      [RL_tau, RL_Sx, RL_Sy, RL_Sz, RL_By, RL_Bz, vx_L, vy_L, vz_L,
                       RR_tau, RR_Sx, RR_Sy, RR_Sz, RR_By, RR_Bz, vx_R, vy_R, vz_R,
                       Bx]
        y : (1,)   — total pressure  p_tot*
    """

    def __init__(self, path: str):
        data = np.load(path)
        self.X = torch.tensor(data["inputs"], dtype=torch.float32)
        self.y = torch.tensor(data["targets"], dtype=torch.float32).unsqueeze(-1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dataloaders(split_dir: str, batch_size: int = 256, num_workers: int = None,
                    device: str = "cpu"):
    """Return train/val/test DataLoaders."""
    pin = (device == "cuda")
    # Multiprocessing workers require torch_shm_manager which breaks on macOS
    # for non-CUDA devices. Default to 0 workers on cpu/mps.
    if num_workers is None:
        num_workers = 4 if device == "cuda" else 0
    loaders = {}
    for name in ["train", "val", "test"]:
        ds = HLLDDataset(f"{split_dir}/{name}.npz")
        loaders[name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=num_workers,
            pin_memory=pin,
        )
    return loaders
