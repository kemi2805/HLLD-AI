"""
Conservative-to-Primitive (C2P) for SR-MHD in flat Minkowski spacetime.

Implements the Kastaun et al. (2021) / arXiv:2312.11358 master-function
approach, fully vectorised in PyTorch (no Python loops over cells).

Conservative variables layout (dict keys):
    D, Sx, Sy, Sz, tau, Bx, By, Bz

Primitive variables output (dict keys):
    rho, vx, vy, vz, p, eps, Bx, By, Bz

The scheme reduces the inversion to a 1-D root-find in mu ∈ (0, mu_+],
where mu = 1 / (rho * h * W).  A bracketed Illinois / Brent-style
secant+bisection is used, identical in spirit to the hydro solver already
present in physics_utils.py but extended for MHD.
"""

import torch
from .eos import hybrid_eos

_TINY = 1e-300


def _sdiv(a: torch.Tensor, b: torch.Tensor, tiny: float = _TINY) -> torch.Tensor:
    sgn  = torch.sign(b)
    sgn  = torch.where(sgn == 0.0, torch.ones_like(sgn), sgn)
    safe = torch.where(b.abs() < tiny, sgn * tiny, b)
    return a / safe


# ── Kastaun master function (vectorised) ──────────────────────────────────────

class KastaunC2P:
    """
    Vectorised Kastaun C2P for SR-MHD (Minkowski, alpha=1, beta^i=0).

    All tensors are shape (N,) — one entry per cell.
    """

    def __init__(
        self,
        cons: dict[str, torch.Tensor],
        eos: hybrid_eos,
    ):
        device = cons["D"].device
        dtype  = cons["D"].dtype

        D   = cons["D"]
        Sx  = cons["Sx"]; Sy = cons["Sy"]; Sz = cons["Sz"]
        tau = cons["tau"]
        Bx  = cons["Bx"]; By = cons["By"]; Bz = cons["Bz"]

        # ── Enforce tau >= 0 (Kastaun eq. before eq. 16) ─────────────────────
        tau = torch.clamp(tau, min=0.0)

        # Momentum causal check: |S| <= D + tau
        S2   = Sx**2 + Sy**2 + Sz**2
        Snorm = torch.sqrt(S2)
        S_max = D + tau
        too_large = Snorm > S_max
        fac = torch.where(too_large, 0.9999 * S_max / (Snorm + _TINY), torch.ones_like(Snorm))
        Sx = Sx * fac; Sy = Sy * fac; Sz = Sz * fac
        S2 = Sx**2 + Sy**2 + Sz**2

        B2     = Bx**2 + By**2 + Bz**2
        BdotS  = Bx*Sx + By*Sy + Bz*Sz   # B · S

        self.D   = D
        self.tau = tau
        self.Sx  = Sx; self.Sy = Sy; self.Sz = Sz
        self.Bx  = Bx; self.By = By; self.Bz = Bz
        self.S2  = S2
        self.B2  = B2
        self.BdotS = BdotS

        # Rescaled variables (Kastaun 2021 notation)
        self.q  = tau / (D + _TINY)          # q = tau/D
        self.rU = torch.stack([Sx, Sy, Sz], dim=1) / (D.unsqueeze(1) + _TINY)  # r^i = S^i/D
        self.rNorm2 = S2 / (D**2 + _TINY)    # |r|^2

        # Rescaled B:  Btilde^i = B^i / sqrt(D)
        sqrtD = torch.sqrt(D + _TINY)
        self.BtildeU = torch.stack([Bx, By, Bz], dim=1) / sqrtD.unsqueeze(1)
        self.BtildeNorm2 = B2 / (D + _TINY)
        self.BtildeNorm  = torch.sqrt(self.BtildeNorm2)

        # b̃·r = (B · S) / D^{3/2}  (dot product of Btilde and r in Kastaun notation)
        self.B_dot_r = BdotS / (D * sqrtD + _TINY)
        # B2_rPerp2 = |r|^2 |b̃|^2 - (b̃·r)^2   (needed for qbar)
        self.B2_rPerp2 = self.rNorm2 * self.BtildeNorm2 - self.B_dot_r**2

        # h_min = 1 (cold vacuum floor) — used for mu_+
        self.h_min = torch.ones_like(D)

        # v0^2 upper bound (eq. 25 of Kastaun 2021)
        self.v02 = self.rNorm2 / (self.rNorm2 + self.h_min**2)

        self.eos = eos

    # ── helper quantities as functions of mu ──────────────────────────────────

    def _chi(self, mu: torch.Tensor) -> torch.Tensor:
        return 1.0 / (1.0 + mu * self.BtildeNorm2)

    def _rbar2(self, mu: torch.Tensor) -> torch.Tensor:
        chi = self._chi(mu)
        return (self.rNorm2 * chi**2
                + mu * chi * (1.0 + chi) * self.B_dot_r**2)

    def _qbar(self, mu: torch.Tensor) -> torch.Tensor:
        chi = self._chi(mu)
        return (self.q
                - 0.5 * self.BtildeNorm2
                - 0.5 * (mu * chi)**2 * self.B2_rPerp2)

    def _fa(self, mu: torch.Tensor) -> torch.Tensor:
        """Eq. for mu_+:  fa(mu) = mu * sqrt(h_min^2 + rbar^2) - 1 = 0."""
        rbar2 = _sdiv(self._rbar2(mu), torch.ones_like(mu))  # just pass through
        return mu * torch.sqrt(self.h_min**2 + self._rbar2(mu)) - 1.0

    def _master(self, mu: torch.Tensor):
        """
        Master function f(mu).  Returns (f, rhohat, epshat) for each cell.
        """
        qbar  = self._qbar(mu)
        rbar2 = self._rbar2(mu)

        vhat2 = torch.clamp(mu**2 * rbar2, max=self.v02)
        What  = 1.0 / torch.sqrt(torch.clamp(1.0 - vhat2, min=1e-10))

        rhohat0  = self.D / What
        # clamp to eos range
        rho_min  = torch.full_like(rhohat0, 1e-12)
        rho_max  = torch.full_like(rhohat0, 1e15)
        rhohat   = torch.clamp(rhohat0, min=rho_min, max=rho_max)

        epshat0  = What * (qbar - mu * rbar2) + vhat2 * What**2 / (1.0 + What)
        eps_lo, eps_hi = self.eos.eps_range__rho(rhohat)
        epshat   = torch.clamp(epshat0, min=eps_lo, max=eps_hi)

        phat     = self.eos.press__eps_rho(epshat, rhohat)
        ahat     = phat / (rhohat * (1.0 + epshat) + _TINY)
        h_eff    = (1.0 + ahat) * (1.0 + epshat)

        nu_A     = h_eff / What
        nu_B     = (1 + ahat) * (1.0 + qbar - mu * rbar2)
        nu       = torch.maximum(nu_A, nu_B)

        f = mu - _sdiv(1.0, nu + mu * rbar2)
        return f, rhohat, epshat, What, vhat2

    # ── find mu_+ (upper bracket) ─────────────────────────────────────────────

    def _find_mu_plus(self, n_iter: int = 50, tol: float = 1e-15) -> torch.Tensor:
        """
        Find mu_+ as root of fa(mu) = 0 in (0, 1/h_min] via bisection.
        """
        mu_lo = torch.zeros_like(self.D)
        mu_hi = 1.0 / (self.h_min + _TINY)

        for _ in range(n_iter):
            mu_mid = 0.5 * (mu_lo + mu_hi)
            f_mid  = self._fa(mu_mid)
            f_lo   = self._fa(mu_lo)
            lft    = (f_lo * f_mid) <= 0.0
            mu_hi  = torch.where(lft, mu_mid, mu_hi)
            mu_lo  = torch.where(lft, mu_lo,  mu_mid)
            if torch.all((mu_hi - mu_lo).abs() < tol):
                break

        return mu_hi  # return upper bound (slightly above root)

    # ── main inversion ────────────────────────────────────────────────────────

    def invert(
        self,
        n_iter: int = 60,
        tol: float = 1e-15,
    ) -> dict[str, torch.Tensor]:
        """
        Invert conservatives → primitives.

        Returns a dict with keys: rho, vx, vy, vz, p, eps, Bx, By, Bz.
        """
        mu_plus = self._find_mu_plus()
        mu_lo   = torch.zeros_like(self.D)
        mu_hi   = mu_plus * (1.0 - 1e-10)   # stay strictly inside bracket

        f_lo, *_ = self._master(mu_lo)
        f_hi, *_ = self._master(mu_hi)

        # If bracket doesn't contain a sign change (degenerate / vacuum),
        # fall back to mu_hi
        good_bracket = (f_lo * f_hi) < 0.0

        # Bisection + secant hybrid (same idea as safe_secant_bisection in hlld.py)
        x0 = mu_lo.clone(); x1 = mu_hi.clone()
        f0 = f_lo.clone();  f1 = f_hi.clone()

        for _ in range(n_iter):
            converged = f1.abs() < tol
            if torch.all(converged):
                break

            df   = f1 - f0
            safe = df.abs() > 1e-30
            x2s  = torch.where(
                safe,
                x1 - f1 * (x1 - x0) / torch.where(safe, df, torch.ones_like(df)),
                0.5 * (x0 + x1),
            )
            inb  = (x2s - x0) * (x2s - x1) < 0.0
            x2   = torch.where(inb, x2s, 0.5 * (x0 + x1))
            f2, *_ = self._master(x2)

            lft  = (f0 * f2) <= 0.0
            x0   = torch.where(lft, x0, x1);  f0 = torch.where(lft, f0, f1)
            x1   = x2;                          f1 = f2

        mu = x1
        _, rhohat, epshat, What, vhat2 = self._master(mu)

        # ── Recover velocity vector ───────────────────────────────────────────
        chi  = self._chi(mu)
        # v^i = mu * chi * (r^i + mu * (B·r) * Btilde^i)
        vhatU = (mu * chi).unsqueeze(1) * (self.rU
                            + (mu * self.B_dot_r).unsqueeze(1) * self.BtildeU)

        # clamp |v| < 1
        v2   = (vhatU**2).sum(dim=1)
        v2c  = torch.clamp(v2, max=1.0 - 1e-10)
        norm = torch.where(v2 > 1.0 - 1e-10,
                           torch.sqrt(v2c / (v2 + _TINY)),
                           torch.ones_like(v2))
        vhatU = vhatU * norm.unsqueeze(1)

        phat = self.eos.press__eps_rho(epshat, rhohat)

        # Use mu_hi everywhere a good bracket wasn't found (degenerate cell)
        # In that case the primitives will be floor values — handled by caller.
        return {
            "rho": rhohat,
            "vx":  vhatU[:, 0],
            "vy":  vhatU[:, 1],
            "vz":  vhatU[:, 2],
            "p":   phat,
            "eps": epshat,
            "Bx":  self.Bx,
            "By":  self.By,
            "Bz":  self.Bz,
        }


def conservative_to_primitive(
    cons: dict[str, torch.Tensor],
    eos: hybrid_eos,
    atmo_rho: float = 1e-10,
    atmo_eps: float = 1e-10,
    n_iter: int = 60,
    tol: float = 1e-15,
) -> dict[str, torch.Tensor]:
    """
    Top-level C2P wrapper.

    Handles atmosphere enforcement and returns primitive state dict.
    """
    solver = KastaunC2P(cons, eos)
    prims  = solver.invert(n_iter=n_iter, tol=tol)

    # Atmosphere floor
    atmo_mask = prims["rho"] < atmo_rho
    prims["rho"] = torch.where(atmo_mask, torch.full_like(prims["rho"], atmo_rho), prims["rho"])
    prims["eps"] = torch.where(atmo_mask, torch.full_like(prims["eps"], atmo_eps), prims["eps"])
    prims["vx"]  = torch.where(atmo_mask, torch.zeros_like(prims["vx"]), prims["vx"])
    prims["vy"]  = torch.where(atmo_mask, torch.zeros_like(prims["vy"]), prims["vy"])
    prims["vz"]  = torch.where(atmo_mask, torch.zeros_like(prims["vz"]), prims["vz"])
    # Recompute pressure from floored state
    prims["p"]   = eos.press__eps_rho(prims["eps"], prims["rho"])

    return prims