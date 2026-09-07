"""A starting guess for the exact solver, read off a batched 1D simulation.

The idea (user, 2026-09-07): the interfaces the Newton fails on are ordinary
Riemann problems.  Evolving each one as a small 1D shock tube produces the
real wave structure -- all seven waves, because a simulation solves the PDE
and does not care that HLLD's flux function knows only five -- and the zone
values can then be read off the profile and handed to the Newton as its
initial guess.  Unlike a scan of the unknown box, this guess is an
approximation of the answer rather than a point that happens to be near it.

The cost objection is answered by batching: one tube is a column, and N
tubes are an ``(ncells, N)`` array evolved together, so the whole population
of failures is one vectorised run rather than N small ones.

Two pieces:

``run_tubes``   evolve N Riemann problems, PLM + HLLD + SSP-RK2, no
                transverse direction (B_n is constant in 1D).
``read_zones``  turn each profile into the six unknowns.  The reader is the
                part the idea stands or falls on, and it is deliberately
                blunt: find the seven strongest jumps per column, keeping
                them apart, take the median of each of the eight segments
                they define, and read the unknowns off zones R2, R4, R6 and
                the rotations R2->R3, R6->R7.  A seed does not have to be
                accurate, only inside the basin, so a reader that misplaces
                a weak wave still helps.

Everything here is per column; nothing couples the tubes.
"""
from __future__ import annotations

import numpy as np
import torch

from .c2p import conservative_to_primitive
from .hlld import hlld_flux, primitive_to_conserved
from .reconstruction import reconstruct_prims

_KEYS = ("D", "Sx", "Sy", "Sz", "tau")


def _prims_from(UL, UR, Bn, eos, ncells, ng):
    """Two constant states, split at the middle of an ``(ncells+2ng, N)`` grid."""
    n = Bn.shape[0]
    nt = ncells + 2 * ng
    half = nt // 2
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64)
    prims = {}
    # solver-frame layout: rho, P_tot, vx, vy, vz, By, Bz  (+ Bn separately)
    for k, j in (("rho", 0), ("vx", 2), ("vy", 3), ("vz", 4), ("By", 5), ("Bz", 6)):
        col = torch.empty((nt, n), dtype=torch.float64)
        col[:half] = t(UL[:, j])
        col[half:] = t(UR[:, j])
        prims[k] = col
    prims["Bx"] = torch.empty((nt, n), dtype=torch.float64)
    prims["Bx"][:] = t(Bn)
    # the states carry TOTAL pressure; the evolution wants gas pressure
    for side, U, sl in ((0, UL, slice(None, half)), (1, UR, slice(half, None))):
        v2 = U[:, 2] ** 2 + U[:, 3] ** 2 + U[:, 4] ** 2
        W2 = 1.0 / np.maximum(1.0 - v2, 1e-300)
        eta = Bn * U[:, 2] + U[:, 5] * U[:, 3] + U[:, 6] * U[:, 4]
        b2 = (Bn ** 2 + U[:, 5] ** 2 + U[:, 6] ** 2) / W2 + eta ** 2
        pg = U[:, 1] - 0.5 * b2
        prims.setdefault("p", torch.empty((nt, n), dtype=torch.float64))
        prims["p"][sl] = t(pg)
    prims["eps"] = eos.eps__press_rho(prims["p"], prims["rho"])
    return prims


def _bc(d, ng):
    for v in d.values():
        v[:ng] = v[ng:ng + 1]
        v[-ng:] = v[-ng - 1:-ng]


def _rhs(prims, eos, dx, ng, limiter):
    L, R = reconstruct_prims(prims, axis=0, eos=eos, limiter=limiter)
    # B_n is single-valued at a face and constant in 1D
    L["Bx"] = R["Bx"] = prims["Bx"][:1].expand(L["rho"].shape).clone()
    F, _, _ = hlld_flux(L, R, eos, idir=0)
    out = {}
    for k in _KEYS + ("By", "Bz"):
        f = F[k]
        out[k] = -(f[1:] - f[:-1]) / dx
    return out


def run_tubes(UL, UR, Bn, eos, *, ncells=192, tend=0.18, cfl=0.3,
              limiter="mc", ng=2, max_steps=400):
    """Evolve N Riemann problems to ``tend`` on ``x in [-0.5, 0.5]``.

    ``UL``/``UR`` are ``(N, 7)`` solver-frame states (rho, P_tot, vx, vy, vz,
    By, Bz) and ``Bn`` is ``(N,)``.  Returns the primitive profiles, each an
    ``(ncells, N)`` tensor with the ghosts stripped.

    ``tend`` is chosen so the fastest wave stays inside the domain: the fast
    speed is below 1, so waves reach at most ``tend`` from the centre.
    """
    dx = 1.0 / ncells
    prims = _prims_from(np.asarray(UL, float), np.asarray(UR, float),
                        np.asarray(Bn, float), eos, ncells, ng)
    _bc(prims, ng)
    cons = primitive_to_conserved(prims)
    cons = {k: cons[k].clone() for k in _KEYS}
    cons["By"], cons["Bz"] = prims["By"].clone(), prims["Bz"].clone()
    Bx = prims["Bx"]

    def sync(c):
        full = {k: c[k] for k in _KEYS}
        full["Bx"], full["By"], full["Bz"] = Bx, c["By"], c["Bz"]
        p = conservative_to_primitive(full, eos)
        p["Bx"], p["By"], p["Bz"] = Bx, c["By"], c["Bz"]
        _bc(p, ng)
        return p

    t, step = 0.0, 0
    while t < tend - 1e-14 and step < max_steps:
        cs = torch.sqrt(torch.clamp(
            eos.gamma * prims["p"] / torch.clamp(
                prims["rho"] + eos.gamma * prims["p"] / (eos.gamma - 1.0), min=1e-30),
            min=0.0))
        # a bound, not the exact fast speed: |v| + c_ms <= |v| + 1
        smax = float((prims["vx"].abs() + torch.clamp(cs + 0.5, max=1.0)).max())
        dt = min(cfl * dx / max(smax, 1e-8), tend - t)

        k1 = _rhs(prims, eos, dx, ng, limiter)
        c1 = {k: cons[k] + dt * k1[k] for k in cons}
        p1 = sync(c1)
        k2 = _rhs(p1, eos, dx, ng, limiter)
        cons = {k: 0.5 * (cons[k] + c1[k] + dt * k2[k]) for k in cons}
        prims = sync(cons)
        t += dt
        step += 1

    ph = slice(ng, ng + ncells)
    return {k: v[ph] for k, v in prims.items()}, t


def _wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def read_zones(prof, Bn, gamma, *, sep=None):
    """Six unknowns per column, read off the profiles.

    Returns ``(unk6, ok)``: a 6-list of ``(N,)`` arrays and a mask of columns
    whose eight segments were all non-empty.
    """
    rho = prof["rho"].numpy()
    p = prof["p"].numpy()
    vx, vy, vz = (prof[k].numpy() for k in ("vx", "vy", "vz"))
    By, Bz = prof["By"].numpy(), prof["Bz"].numpy()
    nc, n = rho.shape
    Bn2 = np.asarray(Bn, float)[None, :]

    v2 = vx * vx + vy * vy + vz * vz
    W2 = 1.0 / np.maximum(1.0 - v2, 1e-300)
    eta = Bn2 * vx + By * vy + Bz * vz
    b2 = (Bn2 ** 2 + By ** 2 + Bz ** 2) / W2 + eta ** 2
    Ptot = p + 0.5 * b2

    # wave indicator: the normalised jump between adjacent cells, summed over
    # the quantities that no single wave family leaves untouched
    def jump(q):
        s = np.maximum(np.abs(q).max(axis=0, keepdims=True), 1e-30)
        return np.abs(np.diff(q, axis=0)) / s

    d = jump(rho) + jump(Ptot) + jump(vx) + jump(vy) + jump(vz) + jump(By) + jump(Bz)

    # seven strongest jumps, kept apart so one wave is not counted twice
    sep = sep if sep is not None else max(2, nc // 24)
    work = d.copy()
    pos = np.zeros((7, n), dtype=int)
    cols = np.arange(n)
    for w in range(7):
        i = np.argmax(work, axis=0)
        pos[w] = i
        for off in range(-sep, sep + 1):
            j = np.clip(i + off, 0, work.shape[0] - 1)
            work[j, cols] = -1.0
    pos = np.sort(pos, axis=0)

    # eight segment medians (segment z runs between wave z-1 and wave z)
    edges = np.concatenate([np.zeros((1, n), dtype=int), pos + 1,
                            np.full((1, n), nc, dtype=int)], axis=0)
    ok = np.ones(n, dtype=bool)
    seg = {}
    idx = np.arange(nc)[:, None]
    for z in range(8):
        lo, hi = edges[z], edges[z + 1]
        m = (idx >= lo[None, :]) & (idx < hi[None, :])
        cnt = m.sum(axis=0)
        ok &= cnt > 0
        w = np.where(m, 1.0, np.nan)
        seg[z] = {k: np.nanmedian(q * w, axis=0)
                  for k, q in (("P", Ptot), ("By", By), ("Bz", Bz))}

    psi = lambda z: np.arctan2(seg[z]["Bz"], seg[z]["By"])

    # A median over a merged pair of zones, with the smeared cells at each end
    # trimmed off.
    def merged(z0, z1, keys, trim=0.25):
        lo, hi = edges[z0], edges[z1 + 1]
        w = (hi - lo).astype(float)
        cut = np.maximum(np.floor(trim * w), 0.0).astype(int)
        a = lo + cut
        b = np.maximum(hi - cut, a + 1)
        m = (idx >= a[None, :]) & (idx < b[None, :])
        good = m.sum(axis=0) > 0
        wm = np.where(m, 1.0, np.nan)
        return {k: np.nanmedian(q * wm, axis=0) for k, q in keys.items()}, good

    # MEASURED (300 interfaces with known answers): merging helps the contact
    # field and HURTS the pressures.  |Bt_CD| improves from 1.0e-1 to 5.8e-2
    # in the log -- better than the network's 8.9e-2 -- because R4 and R5 are
    # one plateau and the contact between them is the least-resolved wave in
    # the profile.  The post-fast pressures are already read from a clean
    # plateau, and widening them across the rotational discontinuity dragged
    # p_RF from 1.2e-3 to 7.1e-3, so those stay on their own zone.
    cd, ok_cd = merged(3, 4, {"By": By, "Bz": Bz})
    ok &= ok_cd
    BtCD = np.hypot(cd["By"], cd["Bz"])

    unk6 = [np.log(np.maximum(seg[1]["P"], 1e-30)),
            np.log(np.maximum(BtCD, 1e-30)),
            np.arctan2(cd["Bz"], cd["By"]),
            np.log(np.maximum(seg[5]["P"], 1e-30)),
            _wrap(psi(2) - psi(1)),
            _wrap(psi(6) - psi(5))]
    for u in unk6:
        ok &= np.isfinite(u)
    return unk6, ok
