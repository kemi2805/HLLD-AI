"""
Neural network architectures for total pressure prediction.
"""

import torch
import torch.nn as nn


class PressureNet(nn.Module):
    """
    Simple fully-connected network.
    Input:  15 features
            [RL_tau, RL_Sx, RL_St, RL_Bt, vx_L, vt_L,
             RR_tau, RR_Sx, RR_St, RR_Bt, vx_R, vt_R,
             Bx, cmin, cmax]
    Output: 1  (normalised log10 total pressure p_tot*)
    """

    def __init__(self, n_input: int = 15, hidden: list = None, activation: str = "silu",
                 dropout: float = 0.0):
        super().__init__()
        if hidden is None:
            hidden = [128, 128, 64]

        act_fn = {"relu": nn.ReLU, "silu": nn.SiLU, "gelu": nn.GELU, "tanh": nn.Tanh}
        act = act_fn.get(activation, nn.SiLU)

        layers = []
        prev = n_input
        for h in hidden:
            #layers += [nn.Linear(prev, h), nn.LayerNorm(h), act()]
            layers += [nn.Linear(prev, h), act(), nn.LayerNorm(h), nn.Dropout(p=dropout)]
            #layers += [nn.Linear(prev, h), act()]
            prev = h
        #layers += [nn.Linear(prev, 1), nn.Softplus()]
        layers += [nn.Linear(prev, 1)]  # no Softplus


        self.net = nn.Sequential(*layers)

        # Weight initialization
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
