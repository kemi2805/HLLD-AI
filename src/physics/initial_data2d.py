"""
2D initial conditions.

Every initial condition also returns a CORNER-valued vector potential
``Az``, and the staggered field is always built from it as

    Bxf[i, j] =  (Az[i, j+1] - Az[i, j]) / dy
    Byf[i, j] = -(Az[i+1, j] - Az[i, j]) / dx

Sampling B at faces directly would leave ``div B`` at truncation error;
taking a discrete curl of a corner potential makes it exactly zero at t=0,
for free, for every problem.  Constrained transport then preserves that to
machine precision, so any nonzero divergence later is a real bug rather
than an initialisation artefact.
"""

from __future__ import annotations

import math

import torch


def b_from_potential(Az: torch.Tensor, dx: float, dy: float
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """Discrete curl of a corner-valued ``Az``: ``(Bxf, Byf)``."""
    Bxf = (Az[:, 1:] - Az[:, :-1]) / dy
    Byf = -(Az[1:, :] - Az[:-1, :]) / dx
    return Bxf, Byf


def _corner_mesh(grid):
    XF = grid.xf.unsqueeze(1).expand(grid.nxt + 1, grid.nyt + 1)
    YF = grid.yf.unsqueeze(0).expand(grid.nxt + 1, grid.nyt + 1)
    return XF, YF


def _fill(grid, eos, mask, sL: dict, sR: dict) -> dict:
    """Cell-centred primitives: ``sL`` where ``mask``, else ``sR``."""
    prims = {}
    for k in ("rho", "p", "vx", "vy", "vz", "Bx", "By", "Bz"):
        prims[k] = torch.where(
            mask,
            torch.full(grid.shape, float(sL[k]), dtype=torch.float64,
                       device=grid.device),
            torch.full(grid.shape, float(sR[k]), dtype=torch.float64,
                       device=grid.device),
        )
    prims["eps"] = eos.eps__press_rho(prims["p"], prims["rho"])
    return prims


def shocktube_2d(grid, eos, sL: dict, sR: dict, direction: str = "x",
                 interface: float = 0.5):
    """A planar shock tube with its normal along ``direction``.

    The state dicts use the usual component names; for ``direction="y"`` the
    caller is responsible for supplying states already expressed in that
    frame (see ``tests/test_sweep2d.py``, which builds them by applying the
    same cyclic relabelling used to verify solver direction-genericity).
    """
    if direction not in ("x", "y"):
        raise ValueError("direction must be 'x' or 'y'")

    mask = (grid.X < interface) if direction == "x" else (grid.Y < interface)
    prims = _fill(grid, eos, mask, sL, sR)

    XF, YF = _corner_mesh(grid)
    if direction == "x":
        # Bn = Bx uniform;  By piecewise constant  ->  Az = Bx*y - int By dx
        By = torch.where(XF < interface, torch.full_like(XF, float(sL["By"])),
                         torch.full_like(XF, float(sR["By"])))
        x_rel = XF - interface
        int_By = torch.where(
            XF < interface, float(sL["By"]) * x_rel, float(sR["By"]) * x_rel
        )
        Az = float(sL["Bx"]) * YF - int_By
    else:
        # Bn = By uniform;  Bx piecewise constant  ->  Az = int Bx dy - By*x
        y_rel = YF - interface
        int_Bx = torch.where(
            YF < interface, float(sL["Bx"]) * y_rel, float(sR["Bx"]) * y_rel
        )
        Az = int_Bx - float(sL["By"]) * XF
    return prims, Az


def magnetic_rotor(grid, eos, r0: float = 0.1, rho_in: float = 10.0,
                   rho_out: float = 1.0, omega: float = 9.95,
                   press: float = 1.0, B0: float = 1.0):
    """Magnetized rotor (Balsara & Spicer; relativistic version).

    A dense cylinder of radius ``r0`` in rigid rotation inside a static,
    lighter ambient medium, threaded by a uniform field ``B = (B0, 0, 0)``.
    Parameters follow the reference setup: ``rho_in=10``, ``rho_out=1``,
    ``omega=9.95`` (so ``v_max = r0*omega = 0.995``, ``W ~ 10``), uniform
    ``p=1``, ideal gas ``gamma=5/3``.

    The cylinder edge is SHARP — no taper.  That is deliberate and matches
    the reference; a tapered rotor is a different (easier) problem.
    """
    X, Y = grid.X, grid.Y
    r2 = X**2 + Y**2
    # relative tolerance keeps the comparison symmetric under the discrete
    # pi-rotation, which the run-time symmetry check relies on
    inside = r2 <= r0 * r0 * (1.0 + 16 * torch.finfo(torch.float64).eps)

    zeros = torch.zeros(grid.shape, dtype=torch.float64, device=grid.device)
    prims = {
        "rho": torch.where(inside, torch.full_like(zeros, rho_in),
                           torch.full_like(zeros, rho_out)),
        "p": torch.full_like(zeros, press),
        "vx": torch.where(inside, -omega * Y, zeros),
        "vy": torch.where(inside, omega * X, zeros),
        "vz": zeros.clone(),
        "Bx": torch.full_like(zeros, B0),
        "By": zeros.clone(),
        "Bz": zeros.clone(),
    }
    prims["eps"] = eos.eps__press_rho(prims["p"], prims["rho"])

    # uniform Bx  ->  Az = B0 * y
    _, YF = _corner_mesh(grid)
    Az = B0 * YF
    return prims, Az


def orszag_tang(grid, eos, gamma: float = 5.0 / 3.0, v0: float = 0.99,
                B0: float | None = None, press: float = 1.0,
                rho: float = 1.0):
    """Relativistic Orszag-Tang vortex on a periodic unit square.

    Smooth initial data whose vector potential
    ``Az = B0*(cos(4 pi x)/(4 pi) + cos(2 pi y)/(2 pi))`` gives
    ``B = B0*(-sin(2 pi y), sin(4 pi x), 0)``.
    """
    if B0 is None:
        B0 = 1.0
    X, Y = grid.X, grid.Y
    zeros = torch.zeros(grid.shape, dtype=torch.float64, device=grid.device)
    prims = {
        "rho": torch.full_like(zeros, rho),
        "p": torch.full_like(zeros, press),
        "vx": -v0 * torch.sin(2 * math.pi * Y) / 2.0,
        "vy": v0 * torch.sin(2 * math.pi * X) / 2.0,
        "vz": zeros.clone(),
        "Bx": -B0 * torch.sin(2 * math.pi * Y),
        "By": B0 * torch.sin(4 * math.pi * X),
        "Bz": zeros.clone(),
    }
    prims["eps"] = eos.eps__press_rho(prims["p"], prims["rho"])

    XF, YF = _corner_mesh(grid)
    Az = B0 * (torch.cos(4 * math.pi * XF) / (4 * math.pi)
               + torch.cos(2 * math.pi * YF) / (2 * math.pi))
    return prims, Az
