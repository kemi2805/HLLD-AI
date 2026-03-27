"""
Loss functions for total pressure prediction.

Besides standard MSE you may want physics-informed losses, e.g.
enforcing positivity or conservation-law constraints.
"""

import torch
import torch.nn as nn


class RelativeMSELoss(nn.Module):
    """MSE on the relative error — useful when p_tot spans orders of magnitude."""
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        rel_err = (pred - target) / (target.abs() + self.eps)
        return torch.mean(rel_err ** 2)


class PhysicsInformedLoss(nn.Module):
    """
    Combines data loss with a penalty for non-physical predictions.
    Extend this with your specific constraints.
    """
    def __init__(self, lambda_phys: float = 0.1):
        super().__init__()
        self.lambda_phys = lambda_phys
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        data_loss = self.mse(pred, target)

        # Penalty: total pressure must be positive
        # Note: with Softplus output this should always be ~0,
        # but kept as a safeguard.
        positivity_penalty = torch.mean(torch.relu(-pred) ** 2)

        # TODO: add more physics constraints as needed

        return data_loss + self.lambda_phys * positivity_penalty
