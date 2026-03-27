"""
Equation of State.

Hybrid EOS: cold polytropic + thermal ideal gas component.
    p_cold = K * rho^gamma
    eps_cold = p_cold / (rho * (gamma - 1))
    p = p_cold + (eps - eps_cold) * rho * (gamma_th - 1)

For cold-only matter (gamma_th=None or 1.0), the thermal term drops out.
"""

import torch


class hybrid_eos:
    def __init__(self, K, gamma, gamma_th=None):
        self.K = K
        self.gamma = gamma
        self.gamma_th = gamma_th
        self.is_cold = (gamma_th is None or gamma_th == 1.0)

    def press_cold_eps_cold__rho(self, rho):
        press_cold = self.K * rho**self.gamma
        eps_cold = press_cold / (rho * (self.gamma - 1))
        return press_cold, eps_cold

    def eps_th__temp(self, temp):
        if self.is_cold:
            return torch.zeros_like(temp)
        return torch.maximum(torch.zeros_like(temp), temp / (self.gamma_th - 1))  # pyright: ignore[reportOptionalOperand]

    def press__eps_rho(self, eps, rho):
        press_cold, eps_cold = self.press_cold_eps_cold__rho(rho)
        eps = torch.maximum(eps, eps_cold)

        if self.is_cold:
            return press_cold

        return press_cold + (eps - eps_cold) * rho * (self.gamma_th - 1)  # pyright: ignore[reportOptionalOperand]

    def eps__press_rho(self, press, rho):
        """Inverse EOS: specific internal energy from pressure and density."""
        press_cold, eps_cold = self.press_cold_eps_cold__rho(rho)
        if self.is_cold:
            return eps_cold
        return eps_cold + (press - press_cold) / (rho * (self.gamma_th - 1))  # pyright: ignore[reportOptionalOperand]

    def eps_range__rho(self, rho):
        press_cold = self.K * rho**self.gamma
        eps_cold = press_cold / (rho * (self.gamma - 1))
        return eps_cold, torch.full_like(rho, 1e5)

    def press_eps__temp_rho(self, temp, rho):
        press_cold, eps_cold = self.press_cold_eps_cold__rho(rho)

        if self.is_cold:
            return press_cold, eps_cold

        temp = torch.maximum(temp, torch.zeros_like(temp))
        eps_th = self.eps_th__temp(temp)
        press = press_cold + eps_th * rho * (self.gamma_th - 1)  # pyright: ignore[reportOptionalOperand]
        eps = eps_cold + eps_th
        return press, eps
    
    def press_and_cs2(self, eps, rho):
        press = self.press__eps_rho(eps, rho)
        h = 1.0 + eps + press / rho
    
        if self.is_cold:
            cs2 = self.gamma * press / (rho * h)
        else:
            press_cold, _ = self.press_cold_eps_cold__rho(rho)
            press_th = press - press_cold
            cs2 = (self.gamma * press_cold + self.gamma_th * press_th) / (rho * h)
    
        return press, cs2
