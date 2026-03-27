"""
Neural network architectures for total pressure prediction.
"""

import torch
import torch.nn as nn


class PressureNet(nn.Module):
    """
    Simple fully-connected network.
    Input:  15 features (7 per side + 1 shared Bx)
            [rho_L, vx_L, vy_L, vz_L, p_L, By_L, Bz_L,
             rho_R, vx_R, vy_R, vz_R, p_R, By_R, Bz_R,
             Bx]
    Output: 1  (total pressure p_tot*)
    """

    def __init__(self, n_input: int = 15, hidden: list = None, activation: str = "silu"):
        super().__init__()
        if hidden is None:
            hidden = [128, 128, 64]

        act_fn = {"relu": nn.ReLU, "silu": nn.SiLU, "gelu": nn.GELU, "tanh": nn.Tanh}
        act = act_fn.get(activation, nn.SiLU)

        layers = []
        prev = n_input
        for h in hidden:
            layers += [nn.Linear(prev, h), act()]
            prev = h
        #layers += [nn.Linear(prev, 1), nn.Softplus()]
        layers += [nn.Linear(prev, 1)]  # no Softplus


        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
