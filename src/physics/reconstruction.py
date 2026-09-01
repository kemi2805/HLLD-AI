"""
Piecewise-constant (PCM) and piecewise-linear (PLM) reconstruction.

Face indexing convention
------------------------
All reconstruction output uses ONE convention, shared with the staggered
magnetic field in Grid2D:

    face i  =  the face between cell i-1 and cell i   ("lower" face of cell i)

so an axis with ``n`` cells has ``n + 1`` faces, and ``QL[i]`` / ``QR[i]``
are the states on the left/right side of face ``i``.  Entries 0 and n are
pure ghost (they would need cells -1 and n) and are filled with the
adjacent cell value; with ng >= 1 they are never read.

This uniformity is deliberate.  Previously ``reconstruct_pcm`` returned
``ntotal-1`` entries and ``reconstruct_plm`` returned ``ntotal-2``, but
``get_interface_states`` sliced both at the same offset — so PLM was shifted
by one cell.  The bug was latent only because the driver hardcoded PCM.
"""

import torch

# ── slope limiters ────────────────────────────────────────────────────────────


def minmod(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Minmod limiter: returns the smaller-magnitude slope if same sign, else 0."""
    return torch.where(
        a * b > 0.0,
        torch.where(a.abs() < b.abs(), a, b),
        torch.zeros_like(a),
    )


def mc_limiter(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Monotonized-Central (van Leer) limiter."""
    c = 0.5 * (a + b)
    return torch.where(
        a * b > 0.0,
        torch.sign(c)
        * torch.minimum(2.0 * a.abs(), torch.minimum(2.0 * b.abs(), c.abs())),
        torch.zeros_like(a),
    )


_LIMITERS = {"mc": mc_limiter, "minmod": minmod}


# ── core reconstruction along one axis ────────────────────────────────────────


def _sl(axis: int, s, ndim: int) -> tuple:
    out = [slice(None)] * ndim
    out[axis] = s
    return tuple(out)


def cell_slopes(Q: torch.Tensor, axis: int = 0, limiter: str = "mc") -> torch.Tensor:
    """Limited slopes at cell centres; same shape as ``Q``, zero on the edges."""
    lim = _LIMITERS.get(limiter)
    if lim is None:
        raise ValueError(f"unknown limiter {limiter!r}")
    nd = Q.ndim
    fwd = Q[_sl(axis, slice(1, None), nd)] - Q[_sl(axis, slice(None, -1), nd)]
    slope = torch.zeros_like(Q)
    slope[_sl(axis, slice(1, -1), nd)] = lim(
        fwd[_sl(axis, slice(None, -1), nd)], fwd[_sl(axis, slice(1, None), nd)]
    )
    return slope


def reconstruct(
    Q: torch.Tensor, axis: int = 0, limiter: str = "pcm"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct ``Q`` to faces along ``axis``.

    Returns ``(QL, QR)``, each with ``Q.shape[axis] + 1`` entries along
    ``axis``: the states on the left and right side of each face.
    """
    nd = Q.ndim
    if limiter == "pcm":
        slope = torch.zeros_like(Q)
    else:
        slope = cell_slopes(Q, axis, limiter)

    left = Q + 0.5 * slope  # right edge of each cell
    right = Q - 0.5 * slope  # left edge of each cell

    # face i is between cell i-1 and cell i
    edge_lo = left[_sl(axis, slice(0, 1), nd)]
    edge_hi = right[_sl(axis, slice(-1, None), nd)]
    QL = torch.cat([edge_lo, left], dim=axis)
    QR = torch.cat([right, edge_hi], dim=axis)
    return QL, QR


# ── primitive-variable reconstruction ────────────────────────────────────────

_RECON_KEYS = ("rho", "p", "vx", "vy", "vz", "Bx", "By", "Bz")


def reconstruct_prims(
    prims: dict[str, torch.Tensor],
    axis: int,
    eos,
    limiter: str = "mc",
    vel_var: str = "Wv",
    floor_rho: float = 1e-12,
    floor_p: float = 1e-14,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Reconstruct primitives to faces along ``axis``.

    Two things this does that a naive variable-by-variable PLM does not:

    ``vel_var="Wv"``
        Reconstruct the spatial four-velocity ``z^i = W v^i`` and recover
        ``v^i = z^i / sqrt(1 + |z|^2)``.  This makes ``|v| < 1`` true at every
        face *by construction*.  Reconstructing ``v`` directly can produce a
        superluminal interface state wherever the limiter is active near a
        steep velocity gradient — fatal for the magnetic rotor, whose initial
        data already sits at W ~ 10.

    EOS consistency
        ``eps`` is recomputed from the reconstructed ``(p, rho)`` rather than
        reconstructed independently, which would leave the face state off the
        EOS surface.

    Cells where the limited state would violate the density or pressure floor
    fall back to piecewise-constant on that face.
    """
    nd = next(iter(prims.values())).ndim

    work = {k: prims[k] for k in _RECON_KEYS if k in prims}
    if vel_var == "Wv":
        v2 = prims["vx"] ** 2 + prims["vy"] ** 2 + prims["vz"] ** 2
        W = 1.0 / torch.sqrt(torch.clamp(1.0 - v2, min=1e-14))
        for c in ("vx", "vy", "vz"):
            work[c] = W * prims[c]

    L: dict[str, torch.Tensor] = {}
    R: dict[str, torch.Tensor] = {}
    for k, v in work.items():
        L[k], R[k] = reconstruct(v, axis=axis, limiter=limiter)

    # positivity fallback: revert to PCM on faces where the limited state
    # would be unphysical
    if limiter != "pcm":
        pL, pR = reconstruct(prims["rho"], axis=axis, limiter="pcm")
        bad = (L["rho"] <= floor_rho) | (R["rho"] <= floor_rho)
        pLp, pRp = reconstruct(prims["p"], axis=axis, limiter="pcm")
        bad = bad | (L["p"] <= floor_p) | (R["p"] <= floor_p)
        if bool(bad.any()):
            for k in work:
                cL, cR = reconstruct(work[k], axis=axis, limiter="pcm")
                L[k] = torch.where(bad, cL, L[k])
                R[k] = torch.where(bad, cR, R[k])

    for side in (L, R):
        if vel_var == "Wv":
            z2 = side["vx"] ** 2 + side["vy"] ** 2 + side["vz"] ** 2
            fac = 1.0 / torch.sqrt(1.0 + z2)
            for c in ("vx", "vy", "vz"):
                side[c] = side[c] * fac
        side["rho"] = torch.clamp(side["rho"], min=floor_rho)
        side["p"] = torch.clamp(side["p"], min=floor_p)
        side["eps"] = eos.eps__press_rho(side["p"], side["rho"])

    return L, R


# ── 1D compatibility wrapper ─────────────────────────────────────────────────


def reconstruct_plm(Q: torch.Tensor, limiter: str = "mc"):
    """Backwards-compatible 1D PLM (see module docstring for the convention)."""
    return reconstruct(Q, axis=0, limiter=limiter)


def reconstruct_pcm(Q: torch.Tensor):
    """Backwards-compatible 1D PCM."""
    return reconstruct(Q, axis=0, limiter="pcm")


def get_interface_states(
    prims: dict[str, torch.Tensor],
    ng: int,
    ncells: int,
    limiter: str = "pcm",
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """1D helper: interface states on the ``ncells+1`` physical faces.

    Physical face ``b`` (between cells ``b-1`` and ``b``) runs
    ``b = ng .. ng+ncells``, which is a direct slice under the face
    convention documented at the top of this module.
    """
    keys = list(prims.keys())
    Q = torch.stack([prims[k] for k in keys], dim=1)  # (ntotal, nvars)
    QL, QR = reconstruct(Q, axis=0, limiter=limiter)  # (ntotal+1, nvars)

    sl = slice(ng, ng + ncells + 1)
    primL = {k: QL[sl, j] for j, k in enumerate(keys)}
    primR = {k: QR[sl, j] for j, k in enumerate(keys)}
    return primL, primR
