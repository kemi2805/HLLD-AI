"""
State containers and shape helpers for the 2D solver.

Data layout decision
--------------------
Cell-centred variables are ``dict[str, Tensor]`` with each value a 2-D
``(nxt, nyt)`` tensor (x-major, ghosts included).  Two consumers require
flat ``(N,)`` batches instead:

  * ``compute_srmhd_fluxes`` / ``HLLDComputation`` — batched over interfaces
  * ``KastaunC2P`` — uses ``torch.stack([Sx,Sy,Sz], dim=1)`` and ``v[:, 0]``

Rather than rewrite either (they are the most delicate code in the repo),
callers wrap them in :func:`flat` / :func:`unflat`.  This is free: every
array here is produced by elementwise arithmetic and is therefore
contiguous, so ``reshape`` returns a *view*, not a copy.

Keeping the 2-D form everywhere else is what makes the constrained-transport
stencils (``Ez[i, j+1]``, ``Efx[i, j-1]``, ``Ec[i-1, j-1]``) readable; the
same expressions written in flat index arithmetic are where sign and
off-by-one bugs hide.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import torch

# Cell-centred variables evolved by the finite-volume update in 2D.
#
# Bx and By are NOT here: with constrained transport they are derived from
# the face-staggered field (see ct.cell_centred_B), not evolved as ordinary
# conserved variables.  This is what removes the 1D-only hacks in
# driver.rk2_step (`fHLLD["Bx"] = 0` and the `cons["Bx"]` copy-back).
#
# Bz IS an ordinary conserved variable: in 2D d/dz = 0, so
# div B = dx Bx + dy By only, and Bz is correctly advanced by its fluxes
# F^x(Bz) = Bz vx - Bx vz and F^y(Bz) = Bz vy - By vz.
EVOLVED_KEYS = ["D", "Sx", "Sy", "Sz", "tau", "Bz"]


def flat(d: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """View every entry as a flat ``(N,)`` tensor (no copy when contiguous)."""
    return {k: v.reshape(-1) for k, v in d.items()}


def unflat(d: dict[str, torch.Tensor], shape: Sequence[int]) -> dict[str, torch.Tensor]:
    """Inverse of :func:`flat`."""
    return {k: v.reshape(*shape) for k, v in d.items()}


@dataclass
class State2D:
    """A complete solver state: cell-centred conserved vars + staggered B.

    ``Bxf`` has shape ``(nxt + 1, nyt)`` and ``Byf`` shape ``(nxt, nyt + 1)``;
    index ``i`` denotes the *lower* face of cell ``i`` in the staggered
    direction (matching the GRACE convention).
    """

    cons: dict[str, torch.Tensor]
    Bxf: torch.Tensor
    Byf: torch.Tensor
    prims: dict[str, torch.Tensor] = field(default_factory=dict)

    def clone(self) -> "State2D":
        return State2D(
            cons={k: v.clone() for k, v in self.cons.items()},
            Bxf=self.Bxf.clone(),
            Byf=self.Byf.clone(),
            prims={k: v.clone() for k, v in self.prims.items()},
        )


def combine(states: Iterable["State2D"], weights: Sequence[float]) -> State2D:
    """Affine combination ``sum_k w_k * state_k`` of solver states.

    The SAME weights are applied to the cell-centred conserved variables and
    to the staggered field.  That is the entirety of the constrained-transport
    Runge-Kutta consistency argument: the divergence operator is linear, so
    any affine combination of divergence-free staggered states is itself
    divergence-free.  No special CT-RK machinery is needed — but only as long
    as the weights really are identical, which is why this function is the
    only place they are applied.
    """
    states = list(states)
    if len(states) != len(weights):
        raise ValueError(f"{len(states)} states but {len(weights)} weights")
    if not states:
        raise ValueError("combine() needs at least one state")

    w0, s0 = weights[0], states[0]
    cons = {k: w0 * s0.cons[k] for k in s0.cons}
    Bxf = w0 * s0.Bxf
    Byf = w0 * s0.Byf
    for w, s in zip(weights[1:], states[1:]):
        for k in cons:
            cons[k] = cons[k] + w * s.cons[k]
        Bxf = Bxf + w * s.Bxf
        Byf = Byf + w * s.Byf
    return State2D(cons=cons, Bxf=Bxf, Byf=Byf)
