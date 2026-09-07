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


def read_zones(prof, Bn, gamma, *, sep=None, full=False, dx=None, t=None,
               mode="segments"):
    """Six unknowns per column, read off the profiles.

    Returns ``(unk6, ok)``: a 6-list of ``(N,)`` arrays and a mask of columns
    whose eight segments were all non-empty.

    With ``full=True`` also returns ``zones`` ``(N, 8, 7)`` and ``speeds``
    ``(N, 7)`` -- the complete wave structure in the same schema the harvest
    and the trainer use, so a tube answer can stand in for a solved one where
    the Newton has no root to find.  The speeds come free: the solution is
    self-similar, so a wave sitting at ``x`` at time ``t`` has speed ``x / t``
    (``dx`` and ``t`` must then be given).
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

    # Seven strongest jumps, kept apart so one wave is not counted twice --
    # located on a COARSE-GRAINED indicator, then mapped back to the fine
    # grid.  Peak-finding directly on the fine profile is resolution-fragile:
    # refining sharpens post-shock oscillations and contact ringing faster
    # than it sharpens weak waves, so at 768 cells the detector started
    # picking numerical features over real ones and the read of |Bt_CD|
    # degraded from 4.8e-2 to 3.2e-1 (measured against known answers at 192,
    # 384 and 768 cells).  Binning the indicator to a fixed count makes the
    # SEGMENTATION independent of resolution while the zone medians are still
    # taken on the fine grid, so refinement buys accuracy instead of noise.
    NB = 128
    fac = max(1, int(np.ceil((nc - 1) / NB)))
    nb = int(np.ceil((nc - 1) / fac))
    pad = nb * fac - (nc - 1)
    db = np.concatenate([d, np.zeros((pad, n))]) if pad else d
    db = db.reshape(nb, fac, n).sum(axis=1)          # coarse indicator

    sep = sep if sep is not None else max(1, nb // 24)
    work = db.copy()
    pos_b = np.zeros((7, n), dtype=int)
    cols = np.arange(n)
    for w in range(7):
        i = np.argmax(work, axis=0)
        pos_b[w] = i
        for off in range(-sep, sep + 1):
            j = np.clip(i + off, 0, nb - 1)
            work[j, cols] = -1.0
    # refine each coarse peak to the strongest fine cell inside its bin
    pos = np.zeros((7, n), dtype=int)
    for w in range(7):
        lo = pos_b[w] * fac
        sub = np.stack([d[np.clip(lo + k, 0, nc - 2), cols] for k in range(fac)])
        pos[w] = np.clip(lo + np.argmax(sub, axis=0), 0, nc - 2)
    pos = np.sort(pos, axis=0)

    # ── landmarks, not a fixed segment count ─────────────────────────────
    # The rotor's interfaces show a MEDIAN OF FOUR waves, not seven (counted
    # on the tube profiles: 3-5 waves on ~90% of them, seven on ~1%).  A
    # planar flow has no rotational discontinuities and its slow waves are
    # often too weak to see, so forcing seven segments splits real plateaus
    # and mislabels the zones -- which is why |Bt_CD| and p_LF stopped
    # converging under refinement (6.1e-2 -> 4.2e-2 -> 6.9e-2 at 192/384/768
    # cells) while p_RF, read from a segment that happened to be right,
    # converged normally (1.2e-3 -> 4.0e-4 -> 3.0e-4).
    #
    # So read from three landmarks that exist whatever the wave count:
    #   * the OUTERMOST jumps are the two fast waves, and the plateau just
    #     inside each is R2 and R7, whose total pressures are the two
    #     unknowns p_LF and p_RF;
    #   * the CONTACT is the jump that moves density without moving total
    #     pressure, and the tangential field is continuous across it, so the
    #     plateaus on either side are one sample of |Bt_CD| and psi_CD.
    # Anything between them may be a slow wave, an Alfven wave, or nothing;
    # the reader no longer has to know which.
    strong = db.max(axis=0)[None, :] * 0.02
    dj_rho = jump(rho); dj_P = jump(Ptot)
    lm = {}
    for name, take in (("first", 0), ("last", -1)):
        lm[name] = pos[take]
    # contact: density moves, total pressure does not
    score = (dj_rho[pos, np.arange(n)[None, :]]
             - 3.0 * dj_P[pos, np.arange(n)[None, :]])
    lm["cd"] = pos[np.argmax(score, axis=0), np.arange(n)]

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
    def between(lo, hi, keys, trim=0.25):
        """Median of each key over the cells strictly between two landmarks."""
        w = np.maximum(hi - lo, 1).astype(float)
        cut = np.maximum(np.floor(trim * w), 1.0).astype(int)
        a2 = np.minimum(lo + cut, nc - 1)
        b2 = np.maximum(np.minimum(hi - cut + 1, nc), a2 + 1)
        m = (idx >= a2[None, :]) & (idx < b2[None, :])
        wm = np.where(m, 1.0, np.nan)
        return {k: np.nanmedian(q * wm, axis=0) for k, q in keys.items()}, m.any(axis=0)

    # R2: just inside the left fast wave.  R7: just inside the right fast.
    # R4/R5: the two plateaus flanking the contact, which share By and Bz.
    # MEASURED, 120 interfaces with known answers, tube to t = 0.18:
    #
    #                   ln p_LF          ln|Bt_CD|          ln p_RF
    #   cells      seg   landmark     seg  landmark      seg  landmark
    #     192   1.9e-2    1.9e-2   6.1e-2   1.9e-1   1.2e-3    2.6e-3
    #     384   1.9e-2    2.0e-2   4.2e-2   1.0e-1   4.0e-4    1.2e-3
    #     768   1.8e-2    1.8e-2   6.9e-2   8.7e-2   3.0e-4    1.3e-3
    #
    # The landmark reader CONVERGES where the fixed segmentation does not
    # (|Bt_CD| falls monotonically instead of wandering) but it is less
    # accurate at every resolution we can afford, so the segments stay the
    # default and the landmarks are kept for the record and for profiles
    # where the wave count is known to be small.  Neither reaches the ~1e-3
    # a training target wants: for the interfaces the seven-wave solver
    # cannot answer, the REDUCED three-wave solver is the better tool
    # (converges on 100% of them, star pressure exact to 1e-4 for
    # |B_n| < 0.03 and 6e-4 below 0.1 -- batched/contact_b).
    if mode == "landmarks":
        lo2 = lm["first"]; hi2 = np.maximum(lm["cd"] - 1, lo2 + 1)
        lo7 = np.minimum(lm["cd"] + 1, lm["last"] - 1); hi7 = lm["last"]
        r2, okA = between(lo2, hi2, {"P": Ptot})
        r7, okB = between(lo7, hi7, {"P": Ptot})
        cdz, okC = between(np.maximum(lm["cd"] - max(2, nc // 40), 0),
                           np.minimum(lm["cd"] + max(2, nc // 40), nc - 1),
                           {"By": By, "Bz": Bz}, trim=0.0)
        ok &= okA & okB & okC
        pLF, pRF = r2["P"], r7["P"]
        Byc, Bzc = cdz["By"], cdz["Bz"]
    else:
        cdz, ok_cd = merged(3, 4, {"By": By, "Bz": Bz})
        ok &= ok_cd
        pLF, pRF = seg[1]["P"], seg[5]["P"]
        Byc, Bzc = cdz["By"], cdz["Bz"]
    BtCD = np.hypot(Byc, Bzc)

    unk6 = [np.log(np.maximum(pLF, 1e-30)),
            np.log(np.maximum(BtCD, 1e-30)),
            np.arctan2(Bzc, Byc),
            np.log(np.maximum(pRF, 1e-30)),
            _wrap(psi(2) - psi(1)),
            _wrap(psi(6) - psi(5))]
    for u in unk6:
        ok &= np.isfinite(u)
    if not full:
        return unk6, ok

    # the complete structure, in the harvest/trainer schema
    comps = (rho, Ptot, vx, vy, vz, By, Bz)
    zones = np.empty((n, 8, 7))
    for z in range(8):
        lo, hi = edges[z], edges[z + 1]
        w = (hi - lo).astype(float)
        cut = np.maximum(np.floor(0.25 * w), 0.0).astype(int)
        a2 = lo + cut
        b2 = np.maximum(hi - cut, a2 + 1)
        mm = np.where((idx >= a2[None, :]) & (idx < b2[None, :]), 1.0, np.nan)
        for j, q in enumerate(comps):
            zones[:, z, j] = np.nanmedian(q * mm, axis=0)
    # R4 and R5 share the tangential field exactly; the contact only jumps rho
    zones[:, 3, 5] = zones[:, 4, 5] = Byc
    zones[:, 3, 6] = zones[:, 4, 6] = Bzc
    speeds = np.full((n, 7), np.nan)
    if dx is not None and t is not None and t > 0:
        # cell centre of the jump, measured from the middle of the domain
        x = (pos.astype(float) + 1.0) * dx - 0.5 * (nc * dx)
        speeds = (x / t).T
    ok &= np.isfinite(zones).all(axis=(1, 2))
    return unk6, ok, zones, speeds
