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


class RelativeMSELoss(nn.Module):
    """
    Relative MSE loss in normalized log10(p*) space.

    Penalises relative errors rather than absolute ones, so weak and
    strong shock cases are weighted equally regardless of pressure magnitude.

    eps: small constant to avoid division by zero for near-zero targets.
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(((pred - target) / (target.abs() + self.eps)) ** 2)
    
class PressureLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        return self.mse(pred, target)

