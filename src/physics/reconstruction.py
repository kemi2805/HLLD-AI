"""
Piecewise-constant (PCM), piecewise-linear (PLM) and high-order (5th and 7th)
reconstruction.

High-order schemes
------------------
Three are available beside ``pcm``/``mc``/``minmod``, all reconstructing the
FACE value from cell averages:

``weno5z``
    Fifth-order WENO with the Borges (2008) "Z" weights.  Non-oscillatory by
    construction -- the sub-stencil whose smoothness indicator is large is
    weighted out -- but it is not monotonicity-preserving: near a
    discontinuity it can still overshoot by a little.
``mp5``
    Suresh & Huynh (1997): the fifth-order interpolation followed by their
    monotonicity-preserving limiter.  This is the one that answers "no
    spikes": the face value is clipped into an interval built from the
    median of the neighbouring values and a curvature estimate, so no new
    extremum is created at a discontinuity, while at a SMOOTH extremum the
    interval is wide enough to leave the fifth-order value untouched (which
    is what a plain TVD limiter like ``mc`` cannot do -- it clips smooth
    extrema and drops to first order there).
``mp7``
    The same limiter around the seventh-order interpolation.

Ghost cells.  The stencils are wider than PLM's, so a run needs more ghost
zones: ``ng >= 3`` for ``weno5z``/``mp5`` and ``ng >= 4`` for ``mp7``, from
``ghosts_needed(limiter)``.  Faces whose stencil would reach outside the
array fall back to PLM and then to PCM, so a too-narrow ``ng`` degrades the
boundary rows instead of reading garbage -- but it is a degraded run, and
``run_2d.py`` sizes the grid from ``ghosts_needed`` rather than relying on
that.

Face indexing convention

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

# ── high-order reconstruction ────────────────────────────────────────────────

# stencil half-width of each scheme = the ghost cells a run needs
_STENCIL = {"pcm": 0, "mc": 1, "minmod": 1, "weno5z": 2, "mp5": 2, "mp7": 3}
HIGH_ORDER = ("weno5z", "mp5", "mp7")


def ghosts_needed(limiter: str) -> int:
    """Ghost cells per side a run needs for ``limiter``.

    PLM needs 2 rather than its stencil's 1: constrained transport takes an
    extra row for the EMF corners (see ``Grid2D``'s docstring), and that
    argument adds the same one row to the wider stencils.
    """
    if limiter not in _STENCIL:
        raise ValueError(f"unknown limiter {limiter!r}")
    return max(2, _STENCIL[limiter] + 1)


def _shift(Q: torch.Tensor, k: int, axis: int) -> torch.Tensor:
    """``Q`` shifted so entry i holds ``Q[i + k]``.

    ``torch.roll`` wraps, which is wrong at the two ends -- every entry whose
    stencil reaches outside is overwritten by the PLM fallback in
    :func:`_high_order_edges`, so the wrapped values never survive.
    """
    return torch.roll(Q, shifts=-k, dims=axis)


def _weno5z_face(v, axis):
    """Fifth-order WENO-Z value at the RIGHT face of each cell."""
    vm2, vm1, v0, vp1, vp2 = (_shift(v, k, axis) for k in (-2, -1, 0, 1, 2))
    p0 = (2.0 * vm2 - 7.0 * vm1 + 11.0 * v0) / 6.0
    p1 = (-vm1 + 5.0 * v0 + 2.0 * vp1) / 6.0
    p2 = (2.0 * v0 + 5.0 * vp1 - vp2) / 6.0
    c1, c2 = 13.0 / 12.0, 0.25
    b0 = c1 * (vm2 - 2.0 * vm1 + v0) ** 2 + c2 * (vm2 - 4.0 * vm1 + 3.0 * v0) ** 2
    b1 = c1 * (vm1 - 2.0 * v0 + vp1) ** 2 + c2 * (vm1 - vp1) ** 2
    b2 = c1 * (v0 - 2.0 * vp1 + vp2) ** 2 + c2 * (3.0 * v0 - 4.0 * vp1 + vp2) ** 2
    # tau5 = |b0 - b2| is what makes the Z weights fifth order at a smooth
    # extremum, where the classic WENO-JS weights drop to third
    tau5 = (b0 - b2).abs()
    eps = 1e-40
    a0 = 0.1 * (1.0 + (tau5 / (b0 + eps)) ** 2)
    a1 = 0.6 * (1.0 + (tau5 / (b1 + eps)) ** 2)
    a2 = 0.3 * (1.0 + (tau5 / (b2 + eps)) ** 2)
    return (a0 * p0 + a1 * p1 + a2 * p2) / (a0 + a1 + a2)


def _median(a, b, c):
    """The median of three values, as Suresh & Huynh write it."""
    return a + minmod(b - a, c - a)


def _mp_limit(vl, v, axis, *, alpha: float = 4.0):
    """Suresh & Huynh's monotonicity-preserving clip of a face value ``vl``.

    ``vl`` is the unlimited high-order value at the right face of each cell,
    ``v`` the cell averages.  The clip is a no-op wherever ``vl`` already
    lies between the cell value and the monotonicity-preserving bound, which
    is the case in smooth regions INCLUDING at extrema -- that is the whole
    point of the scheme over a TVD limiter.
    """
    vm2, vm1, v0, vp1, vp2 = (_shift(v, k, axis) for k in (-2, -1, 0, 1, 2))
    vmp = v0 + minmod(vp1 - v0, alpha * (v0 - vm1))
    # cheap test: no clipping needed unless the value leaves [v0, vmp]
    need = (vl - v0) * (vl - vmp) > 1e-40
    d_m1 = vm2 - 2.0 * vm1 + v0
    d_0 = vm1 - 2.0 * v0 + vp1
    d_p1 = v0 - 2.0 * vp1 + vp2
    dm4_p = minmod(minmod(4.0 * d_0 - d_p1, 4.0 * d_p1 - d_0), minmod(d_0, d_p1))
    dm4_m = minmod(minmod(4.0 * d_m1 - d_0, 4.0 * d_0 - d_m1), minmod(d_m1, d_0))
    v_ul = v0 + alpha * (v0 - vm1)                      # upper limit
    v_av = 0.5 * (v0 + vp1)
    v_md = v_av - 0.5 * dm4_p                           # median
    v_lc = v0 + 0.5 * (v0 - vm1) + (4.0 / 3.0) * dm4_m  # large curvature
    vmin = torch.maximum(torch.minimum(torch.minimum(v0, vp1), v_md),
                         torch.minimum(torch.minimum(v0, v_ul), v_lc))
    vmax = torch.minimum(torch.maximum(torch.maximum(v0, vp1), v_md),
                         torch.maximum(torch.maximum(v0, v_ul), v_lc))
    return torch.where(need, _median(vl, vmin, vmax), vl)


def _mp5_face(v, axis):
    vm2, vm1, v0, vp1, vp2 = (_shift(v, k, axis) for k in (-2, -1, 0, 1, 2))
    vl = (2.0 * vm2 - 13.0 * vm1 + 47.0 * v0 + 27.0 * vp1 - 3.0 * vp2) / 60.0
    return _mp_limit(vl, v, axis)


def _mp7_face(v, axis):
    vm3, vm2, vm1, v0, vp1, vp2, vp3 = (_shift(v, k, axis)
                                        for k in (-3, -2, -1, 0, 1, 2, 3))
    vl = (-3.0 * vm3 + 25.0 * vm2 - 101.0 * vm1 + 319.0 * v0
          + 214.0 * vp1 - 38.0 * vp2 + 4.0 * vp3) / 420.0
    return _mp_limit(vl, v, axis)


_FACE_FN = {"weno5z": _weno5z_face, "mp5": _mp5_face, "mp7": _mp7_face}


def _high_order_edges(Q: torch.Tensor, axis: int, limiter: str):
    """(right edge, left edge) of every cell, same shape as ``Q``.

    The left edge is the right edge of the mirrored problem: reversing the
    axis turns "the face at i+1/2 seen from the left" into "the face at
    i-1/2 seen from the right", so one implementation serves both and the
    two sides cannot drift apart.

    Cells whose stencil reaches outside the array keep the PLM value (and
    the outermost cell the PCM one), so the wrap-around of ``_shift`` is
    never read.
    """
    fn = _FACE_FN[limiter]
    nd = Q.ndim
    right_edge = fn(Q, axis)                       # value at i+1/2
    rev = torch.flip(Q, dims=(axis,))
    left_edge = torch.flip(fn(rev, axis), dims=(axis,))   # value at i-1/2

    w = _STENCIL[limiter]
    slope = cell_slopes(Q, axis, "mc")             # zero on the outermost cells
    plm_r, plm_l = Q + 0.5 * slope, Q - 0.5 * slope
    n = Q.shape[axis]
    if n <= 2 * w:
        return plm_r, plm_l
    edge = torch.zeros(n, dtype=torch.bool, device=Q.device)
    edge[:w] = True
    edge[n - w:] = True
    shape = [1] * nd
    shape[axis] = n
    edge = edge.reshape(shape)
    return (torch.where(edge, plm_r, right_edge),
            torch.where(edge, plm_l, left_edge))


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
    if limiter in _FACE_FN:
        left, right = _high_order_edges(Q, axis, limiter)
    else:
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
