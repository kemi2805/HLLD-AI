"""The high-resolution tube test (science plan, item C).

The question
------------
Item B left 83% of the genuinely stubborn interfaces unexplained: the Newton
cannot start on them even tilted out of the plane, so nothing in the
seven-wave formulation says what their solution looks like -- or whether the
elementary seven-wave family contains it at all.  A simulation does not care.
Evolving the interface as a 1D shock tube solves the PDE, so whatever the
real structure is, it appears in the profile.

Method
------
For each interface, ``tube_seed.run_tubes`` evolves the Riemann problem on
x in [-0.5, 0.5] to ``tend``, at TWO resolutions.  The solution is
self-similar, so a feature sitting at x at time t has speed xi = x / t, and
the two runs can be compared in xi.

Features are found on a coarse-grained jump indicator (the same device
``read_zones`` uses, for the same reason: refining sharpens post-shock
ringing faster than it sharpens weak waves, so a fine-grid detector is
resolution-fragile).  Every run of cells above the threshold is one feature,
with its xi window, its strength, and its width.

Each feature is then assigned to a wave family by SPEED, not by order: the
seven characteristic speeds (fast-, Alfven-, slow-, contact, slow+, Alfven+,
fast+) are evaluated in the states just ahead of and just behind the feature,
and the family whose speed interval overlaps the feature's xi window wins.  A
feature matching no family is reported as unassigned rather than forced.

What would be evidence of what
------------------------------
* **Seven or fewer features, one per family:** the elementary family contains
  the solution; the Newton's failure is a basin problem, not a structural one.
* **Two features inside one family** (the spec's "two features between a fast
  wave and the contact on one side"): a compound wave, which the seven-wave
  formulation cannot represent -- a structural answer to why no root exists.
* **Unassigned features:** something outside the classification; look at it.

Uncertainty comes from the resolution pair: a feature is CONFIRMED when it
appears at both resolutions with matching speed.  Counting only confirmed
features is what keeps ringing and contact noise out of the conclusion.
Controls -- interfaces the solver does answer -- are run identically, so the
detector's own bias cancels in the comparison.

The tube read is also handed to the Newton (``read_zones`` -> fullcontact6),
which measures directly how many of these lanes a tube seed can rescue at
today's solver.

Usage (calea, through Slurm: scripts/calea_tube_features.sh)
    python scripts/tube_features.py <coplanar_limit dir> --n 100 \
        --shard 0 --nshards 64 --out results/tube_features/shard_000.npz
    python scripts/tube_features.py --summarise results/tube_features
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
import socket
import subprocess
import sys
import time
import warnings

import numpy as np
import torch

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)
from rmhd.eos import set_eos                                       # noqa: E402
set_eos("ideal")
from rmhd.batched import classify as CL                            # noqa: E402
from rmhd.batched import fullcontact_b as FB                       # noqa: E402
from rmhd.batched import wave_speeds_b as WB                       # noqa: E402
from rmhd.util.angles import wrap                                  # noqa: E402
from src.physics.eos import hybrid_eos                             # noqa: E402
from src.physics.exact_flux import _self_crossing                  # noqa: E402
from src.physics.tube_seed import read_zones, run_tubes            # noqa: E402

warnings.filterwarnings("ignore")
np.seterr(all="ignore")
torch.set_num_threads(1)

GAMMA = 5.0 / 3.0
VERIFY = 1e-8
WAVE_ORDER_TOL = 1e-6
FAMILIES = ("F-", "A-", "S-", "CD", "S+", "A+", "F+")
# groups, in the order the summary prints them
GROUPS = ("control", "stub-nostart", "stub-lost", "stub-exact0")
MAXF = 16          # features kept per lane (padded output)
STRENGTH_CUTS = (1e-6, 1e-4, 1e-3, 1e-2, 1e-1)


# ── population ───────────────────────────────────────────────────────────

def load_lanes(d, n_per_group, seed=0, pick=None):
    """Lanes from a coplanar_limit run, grouped by what item B made of them.

    ``pick`` selects exact lanes by their index in that run's concatenation
    (the ``gidx`` this writes out), which is how the plotting pass asks for
    the lanes worth looking at.
    """
    fs = sorted(glob.glob(os.path.join(d, "shard_*.npz")))
    if not fs:
        raise SystemExit("no shard_*.npz in %s" % d)
    Z = [np.load(f) for f in fs]
    cat = lambda k: np.concatenate([z[k] for z in Z])
    grp, cls, route = cat("group"), cat("cls"), cat("pol_route")
    solvable = cat("prod_ok") | cat("ref_ok")          # production answers it today
    still = (grp == 1) & ~solvable
    masks = {
        "control": (grp == 0) & solvable,
        "stub-nostart": still & (cls == 0),
        "stub-lost": still & (cls == 1),
        "stub-exact0": still & (route > 0),
    }
    UL, UR, Bn = cat("UL"), cat("UR"), cat("Bn")
    out = {k: [] for k in ("UL", "UR", "Bn", "group", "orig", "gidx")}
    orig = cat("orig")
    for gi, g in enumerate(GROUPS):
        i = np.flatnonzero(masks[g])
        if pick is not None:
            i = np.array([j for j in i if j in pick], dtype=int)
        elif n_per_group and i.size > n_per_group:
            i = np.sort(np.random.default_rng(seed + gi).choice(i, n_per_group,
                                                                replace=False))
        out["UL"].append(UL[i]); out["UR"].append(UR[i]); out["Bn"].append(Bn[i])
        out["orig"].append(orig[i]); out["gidx"].append(i)
        out["group"].append(np.full(i.size, gi, np.int8))
    return {k: np.concatenate(v) for k, v in out.items()}


# ── characteristic speeds ────────────────────────────────────────────────

def speeds7(state, Bn):
    """The seven characteristic speeds per lane, in spatial order."""
    eig, _, _, ok = WB.xi_all(*state, Bn, GAMMA)
    la, ra = CL.alfven_speeds(*state, Bn, GAMMA)
    return np.stack([eig[:, 0], la, eig[:, 1], state[2], eig[:, 2], ra,
                     eig[:, 3]], axis=1), ok.all(axis=1)


def state_at(prof, Ptot, j, i):
    """Column ``j``'s solver-frame state at cell ``i`` (clamped)."""
    i = int(np.clip(i, 0, prof["rho"].shape[0] - 1))
    g = lambda k: np.array([float(prof[k][i, j])])
    return [g("rho"), np.array([float(Ptot[i, j])]), g("vx"), g("vy"), g("vz"),
            g("By"), g("Bz")]


# ── feature detection ────────────────────────────────────────────────────

def indicator(prof, Ptot):
    """Coarse-grained jump indicator, (nbins, N), resolution-independent."""
    q = [prof[k].numpy() for k in ("rho", "vx", "vy", "vz", "By", "Bz")]
    q.append(Ptot)
    d = 0.0
    for a in q:
        s = np.maximum(np.abs(a).max(axis=0, keepdims=True), 1e-30)
        d = d + np.abs(np.diff(a, axis=0)) / s
    nc1, n = d.shape
    NB = 256
    fac = max(1, int(np.ceil(nc1 / NB)))
    nb = int(np.ceil(nc1 / fac))
    pad = nb * fac - nc1
    db = np.concatenate([d, np.zeros((pad, n))]) if pad else d
    return db.reshape(nb, fac, n).sum(axis=1), fac


def find_features(db, fac, nc, t, thresh):
    """Runs of active bins -> [(xi_lo, xi_hi, strength, i_lo, i_hi)].

    The bar is a fraction of the column's TOTAL variation, not of its largest
    bin: a rotation or a slow wave beside a strong shock carries a percent of
    the total and would never clear a threshold set by the shock.  A noise
    floor of three times the median bin keeps post-shock ringing out.
    """
    out = []
    nb = db.shape[0]
    bar = max(thresh * float(db.sum()), 3.0 * float(np.median(db)), 1e-30)
    strong = db > bar
    i = 0
    while i < nb:
        if not strong[i]:
            i += 1
            continue
        j = i
        while j + 1 < nb and (strong[j + 1] or (j + 2 < nb and strong[j + 2])):
            j += 1                                   # bridge one quiet bin
        lo_c, hi_c = i * fac, min((j + 1) * fac, nc - 1)
        x_lo = -0.5 + lo_c / nc
        x_hi = -0.5 + (hi_c + 1) / nc
        out.append((x_lo / t, x_hi / t, float(db[i:j + 1].sum()), lo_c, hi_c))
        i = j + 1
    return out


def assign(feat, prof, Ptot, j, tol):
    """``(family, n_overlapping)`` for one feature; family -1 = unassigned.

    The seven speeds can be nearly degenerate -- rotor interfaces routinely
    have the Alfven, slow and contact speeds inside 0.02 of each other -- so
    a feature can sit in more than one family's interval.  The closest centre
    wins, and the count of overlapping families is returned with it, because
    a "two features in one family" verdict is only worth anything when the
    assignment was unambiguous.
    """
    xlo, xhi, _, ilo, ihi = feat
    pad = max(2, (ihi - ilo) // 2)
    sL, okL = speeds7(state_at(prof, Ptot, j, ilo - pad), np.array([BN_J[j]]))
    sR, okR = speeds7(state_at(prof, Ptot, j, ihi + pad), np.array([BN_J[j]]))
    if not (okL[0] and okR[0]):
        return -1, 0
    mid = 0.5 * (xlo + xhi)
    best, bestd, nov = -1, np.inf, 0
    for k in range(7):
        a, b = sorted((float(sL[0, k]), float(sR[0, k])))
        if max(a - tol - xhi, xlo - (b + tol), 0.0) > 0.0:
            continue                                  # no overlap with family k
        nov += 1
        d = abs(mid - 0.5 * (a + b))
        if d < bestd:
            best, bestd = k, d
    return best, nov


def elementary(feat, prof, Ptot, j, bt_floor=1e-3):
    """Is one feature consistent with an ELEMENTARY wave?

    The seven-wave family allows exactly two kinds of jump.  A rotational
    discontinuity turns the tangential field and leaves rho, P_tot and |B_t|
    alone; a magnetosonic wave changes them and does NOT turn the field (in a
    planar problem it may only reverse it -- and a reversal through |B_t| = 0
    is precisely the switch-off structure an elementary wave cannot carry,
    because the rotation is then glued to the magnitude change).

    Returns ``(kind, dpsi, dmag, bt_min_rel)`` with kind
        0 magnetosonic (no rotation)      1 rotation only (an RD)
        2 REVERSING     rotation by ~pi carried together with a magnitude
                        or density jump -- compound / intermediate
        3 mixed         an intermediate rotation with a magnitude jump
        -1 undecidable  (the field vanishes on one side)
    """
    _, _, _, ilo, ihi = feat
    pad = max(2, (ihi - ilo) // 2)
    a = state_at(prof, Ptot, j, ilo - pad)
    b = state_at(prof, Ptot, j, ihi + pad)
    # NB every element of a state here is a 1-element ARRAY; numpy 2 (the
    # cluster venvs) refuses float() on those, where numpy 1.26 (the laptop)
    # allowed it, so index before converting.
    sc = lambda x: float(np.asarray(x).reshape(-1)[0])
    BtA, BtB = sc(np.hypot(a[5], a[6])), sc(np.hypot(b[5], b[6]))
    scale = max(BtA, BtB)
    lo = max(0, ilo - pad)
    hi = min(prof["rho"].shape[0], ihi + pad + 1)
    bt_in = np.hypot(prof["By"][lo:hi, j].numpy(), prof["Bz"][lo:hi, j].numpy())
    bt_min_rel = float(bt_in.min()) / max(scale, 1e-30)
    if scale < bt_floor:
        return -1, 0.0, 0.0, bt_min_rel
    dpsi = abs(sc(wrap(np.arctan2(b[6], b[5]) - np.arctan2(a[6], a[5]))))
    dmag = max(abs(BtB - BtA) / max(scale, 1e-30),
               abs(sc(b[0]) - sc(a[0])) / max(abs(sc(a[0])), 1e-30),
               abs(sc(b[1]) - sc(a[1])) / max(abs(sc(a[1])), 1e-30))
    if dpsi < 0.05:
        return 0, dpsi, dmag, bt_min_rel
    if dmag < 0.05:
        return 1, dpsi, dmag, bt_min_rel
    return (2 if dpsi > np.pi - 0.3 else 3), dpsi, dmag, bt_min_rel


def match(f1, f2, tol=0.02):
    """Features confirmed at both resolutions, matched by speed window."""
    used, conf = set(), []
    for a in f1:
        ca = 0.5 * (a[0] + a[1])
        for j, b in enumerate(f2):
            if j in used:
                continue
            cb = 0.5 * (b[0] + b[1])
            if abs(ca - cb) <= tol + 0.5 * ((a[1] - a[0]) + (b[1] - b[0])):
                used.add(j)
                conf.append((a, b))
                break
    return conf


def wave_strengths(res, left, right):
    """Per-family strength and speed of a seven-wave answer.

    Strength is the three-way split exact_flux.relative_jump uses (rho and
    P_tot relative, v as a vector, B against the field and pressure scale),
    evaluated across each wave; speed is the wave's own speed, the contact's
    read off the two states beside it.
    """
    Z = res["zones"]
    st = [left] + [[c for c in z] for z in Z] + [right]      # L, R2..R7, R
    def jump(a, b):
        t = 1e-30
        d = np.abs(a[0] - b[0]) / (np.abs(a[0]) + np.abs(b[0]) + t)
        d = np.maximum(d, np.abs(a[1] - b[1]) / (np.abs(a[1]) + np.abs(b[1]) + t))
        d = np.maximum(d, np.sqrt(sum((a[k] - b[k]) ** 2 for k in (2, 3, 4))))
        dB = np.hypot(a[5] - b[5], a[6] - b[6])
        sc = (np.hypot(a[5], a[6]) + np.hypot(b[5], b[6])
              + np.sqrt(np.abs(a[1]) + np.abs(b[1])))
        return np.maximum(d, dB / (sc + t))
    strength = np.stack([jump(st[k], st[k + 1]) for k in range(7)], axis=1)
    cd = 0.5 * (Z[2][2] + Z[3][2])
    speed = np.stack([res["VsLv"][0], res["VsLv"][1], res["VsLv"][2], cd,
                      res["VsRv"][2], res["VsRv"][1], res["VsRv"][0]], axis=1)
    return strength, speed


def calibrate(UL, UR, Bn, models, max_iter):
    """Solve each lane the production way; returns (ok, strength, speed)."""
    from rmhd.batched import api as API
    from rmhd.batched import ml_b as MB
    from rmhd import ml_guess as mg, paths                  # noqa: F401
    left = [UL[:, j].copy() for j in range(7)]
    right = [UR[:, j].copy() for j in range(7)]
    (m0, s0), rest = models[0], models[1:]
    cands, feats = MB.predict_unk6_k(m0, s0, UL[:, :7], UR[:, :7], Bn)
    seed, _ = MB.clamp_seed_physical(cands[0], left, right, Bn)
    extra = [MB.clamp_seed_physical(u, left, right, Bn)[0] for u in cands[1:]]
    for m, sc in rest:
        u, _ = MB.predict_unk6(m, sc, UL[:, :7], UR[:, :7], Bn)
        extra.append(MB.clamp_seed_physical(u, left, right, Bn)[0])
    res, _ = API.make_solver(GAMMA)(left, right, Bn, seed6=seed,
                                    keys=MB.lane_keys(feats), accuracy=1e-8,
                                    max_iter=max_iter, n_retries=2,
                                    seeds_extra=(extra or None))
    ok = np.asarray(res["converged"], bool) & ~_self_crossing(
        res["zones"], res["VsLv"], res["VsRv"], WAVE_ORDER_TOL)
    strength, speed = wave_strengths(res, left, right)
    return ok, strength, speed


# ── one shard ────────────────────────────────────────────────────────────

BN_J = None            # per-column Bn, set in run_shard (assign() reads it)


def tube(UL, UR, Bn, eos, ncells, tend, limiter, max_steps):
    prof, t = run_tubes(UL, UR, Bn, eos, ncells=ncells, tend=tend,
                        limiter=limiter, max_steps=max_steps)
    rho = prof["rho"].numpy()
    vx, vy, vz = (prof[k].numpy() for k in ("vx", "vy", "vz"))
    By, Bz = prof["By"].numpy(), prof["Bz"].numpy()
    v2 = vx * vx + vy * vy + vz * vz
    W2 = 1.0 / np.maximum(1.0 - v2, 1e-300)
    eta = Bn[None, :] * vx + By * vy + Bz * vz
    b2 = (Bn[None, :] ** 2 + By ** 2 + Bz ** 2) / W2 + eta ** 2
    Ptot = prof["p"].numpy() + 0.5 * b2
    return prof, Ptot, t, rho.shape[0]


def run_shard(a):
    global BN_J
    t0 = time.time()
    tag = "[%s shard %d/%d]" % (socket.gethostname(), a.shard, a.nshards)
    log = lambda s: print("%s %6.0fs  %s" % (tag, time.time() - t0, s), flush=True)
    pick = (set(int(x) for x in a.pick.split(",")) if a.pick else None)
    pop = load_lanes(a.lanes, a.n, pick=pick)
    sel = np.arange(pop["Bn"].size)[a.shard::a.nshards]
    pop = {k: v[sel] for k, v in pop.items()}
    n = sel.size
    if n == 0:
        raise SystemExit("no lanes in this shard")
    log("%d lanes: %s" % (n, ", ".join("%s %d" % (g, (pop["group"] == i).sum())
                                       for i, g in enumerate(GROUPS))))
    UL, UR, Bn = pop["UL"][:, :7], pop["UR"][:, :7], pop["Bn"]
    BN_J = Bn
    eos = hybrid_eos(K=0.0, gamma=GAMMA, gamma_th=GAMMA)
    F = FB.make_solver(GAMMA)
    from rmhd import ml_guess as mg, paths
    models = [mg.load(str(paths.resolve(c))) for c in
              [os.environ.get("RMHD_ML_CKPT", "data/ml_guess_rotor_ft.pt")]
              + [c for c in os.environ.get(
                  "RMHD_ML_CKPTS",
                  "data/ml_guess_rotor_big_s42.pt,data/ml_guess_rotor_big_s47.pt"
              ).split(",") if c]]
    known_ok, known_str, known_spd = calibrate(pop["UL"], pop["UR"], Bn, models,
                                               a.max_iter)
    log("calibration: %d/%d lanes have a verified seven-wave answer to check "
        "the detector against" % (known_ok.sum(), n))

    runs = []
    for nc in (a.ncells, 2 * a.ncells):
        prof, Ptot, t_end, ncc = tube(UL, UR, Bn, eos, nc, a.tend, a.limiter,
                                      a.max_steps)
        db, fac = indicator(prof, Ptot)
        runs.append(dict(prof=prof, Ptot=Ptot, t=t_end, nc=ncc, db=db, fac=fac))
        log("tube at %d cells: t = %.3f" % (nc, t_end))

    nfeat = np.zeros((n, 2), np.int16)
    counts = np.zeros((n, 7), np.int16)            # confirmed features per family
    counts_sure = np.zeros((n, 7), np.int16)       # ... assigned unambiguously
    unassigned = np.zeros(n, np.int16)
    fnov = np.zeros((n, MAXF), np.int8)
    fkind = np.full((n, MAXF), -2, np.int8)          # elementary() verdict
    fdpsi = np.full((n, MAXF), np.nan)
    fdmag = np.full((n, MAXF), np.nan)
    fbtmin = np.full((n, MAXF), np.nan)
    fspeed = np.full((n, MAXF), np.nan)
    fwidth = np.full((n, MAXF), np.nan)
    ffam = np.full((n, MAXF), -2, np.int8)
    fstr = np.full((n, MAXF), np.nan)
    for j in range(n):
        feats = []
        for r in runs:
            feats.append(find_features(r["db"][:, j], r["fac"], r["nc"], r["t"],
                                       a.thresh))
        nfeat[j] = (len(feats[0]), len(feats[1]))
        conf = match(feats[0], feats[1], a.match_tol)
        r = runs[1]                                  # classify on the finer run
        for m, (_, fb) in enumerate(conf[:MAXF]):
            k, nov = assign(fb, r["prof"], r["Ptot"], j, a.speed_tol)
            fnov[j, m] = nov
            (fkind[j, m], fdpsi[j, m], fdmag[j, m],
             fbtmin[j, m]) = elementary(fb, r["prof"], r["Ptot"], j)
            fspeed[j, m] = 0.5 * (fb[0] + fb[1])
            fwidth[j, m] = fb[1] - fb[0]
            fstr[j, m] = fb[2]
            ffam[j, m] = k
            if k < 0:
                unassigned[j] += 1
            else:
                counts[j, k] += 1
                if nov == 1:
                    counts_sure[j, k] += 1
        if a.debug and j < a.debug:
            sL, _ = speeds7([UL[:, c][j:j + 1] for c in range(7)], Bn[j:j + 1])
            sR, _ = speeds7([UR[:, c][j:j + 1] for c in range(7)], Bn[j:j + 1])
            print("  lane %d (%s): raw %d/%d, confirmed %d" % (
                j, GROUPS[pop["group"][j]], len(feats[0]), len(feats[1]), len(conf)))
            print("    speeds in L: %s" % np.array2string(sL[0], precision=3))
            print("    speeds in R: %s" % np.array2string(sR[0], precision=3))
            for m in range(MAXF):
                if ffam[j, m] == -2:
                    break
                print("    xi %+.4f  width %.4f  strength %7.3f  family %s"
                      " (%d families overlap)"
                      % (fspeed[j, m], fwidth[j, m], fstr[j, m],
                         FAMILIES[ffam[j, m]] if ffam[j, m] >= 0 else "UNASSIGNED",
                         fnov[j, m]))
    # Detection, measured two ways, because the loose one is worthless: the
    # seven speeds are often crowded inside 0.02 and a fan is 0.1-0.4 wide in
    # xi, so "a feature covers this wave's speed" is satisfied by accident
    # (it called 94% of waves below strength 1e-4 detected).
    #   known_seen  STRICT: a feature was ASSIGNED to that family.
    #   n_above[S]  the number of waves above strength S, to compare with the
    #               number of confirmed features per lane.
    known_seen = np.zeros((n, 7), bool)
    for j in range(n):
        if not known_ok[j]:
            continue
        for m in range(MAXF):
            if ffam[j, m] == -2:
                break
            if ffam[j, m] >= 0:
                known_seen[j, ffam[j, m]] = True
    n_above = np.stack([(known_str > S).sum(axis=1) for S in STRENGTH_CUTS],
                       axis=1)
    log("non-elementary features: %d lanes carry a reversal with a magnitude "
        "jump, %d a mixed rotation" % (int((fkind == 2).any(axis=1).sum()),
                                       int((fkind == 3).any(axis=1).sum())))
    log("features: median confirmed %.1f; lanes with a doubled family %d; "
        "with an unassigned feature %d"
        % (np.median((ffam >= 0).sum(axis=1)), int((counts > 1).any(axis=1).sum()),
           int((unassigned > 0).sum())))

    # the tube read, handed to the Newton
    r = runs[1]
    unk, okread, zt, st = read_zones(r["prof"], Bn, GAMMA, full=True,
                                     dx=1.0 / r["nc"], t=r["t"])
    left = [UL[:, k].copy() for k in range(7)]
    right = [UR[:, k].copy() for k in range(7)]
    u0 = [np.asarray(c, float) for c in unk]
    fv0, _, _, _, e0 = F["fullfuncv6"](left, right, u0, Bn)
    read_nrm = np.where(e0, np.inf, np.max(np.abs(fv0), axis=0))
    res = F["fullcontact6"](left, right, u0, Bn,
                            accuracy=1e-10, max_iter=a.max_iter)
    tube_ok = (np.asarray(res["nrm"]) <= VERIFY) & okread & ~_self_crossing(
        res["zones"], res["VsLv"], res["VsRv"], WAVE_ORDER_TOL)
    log("tube seed -> Newton: %d/%d verified" % (tube_ok.sum(), n))

    np.savez_compressed(
        a.out, group=pop["group"], orig=pop["orig"], gidx=pop["gidx"],
        UL=pop["UL"], UR=pop["UR"],
        Bn=Bn, nfeat=nfeat, counts=counts, counts_sure=counts_sure,
        fnov=fnov, fkind=fkind, fdpsi=fdpsi, fdmag=fdmag, fbtmin=fbtmin,
        unassigned=unassigned, known_ok=known_ok,
        known_str=known_str, known_spd=known_spd, known_seen=known_seen,
        n_above=n_above, strength_cuts=np.array(STRENGTH_CUTS),
        tube_read_nrm=read_nrm,
        fspeed=fspeed, fwidth=fwidth, ffam=ffam, fstr=fstr,
        tube_ok=tube_ok, tube_read_ok=okread, tube_nrm=np.asarray(res["nrm"]),
        ncells=a.ncells, tend=a.tend, t_reached=r["t"], thresh=a.thresh,
        wall=time.time() - t0, host=socket.gethostname(),
        hlld_rev=subprocess.run(["git", "-C", _HERE, "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True).stdout.strip(),
        script_md5=hashlib.md5(open(__file__, "rb").read()).hexdigest(),
        argv=" ".join(sys.argv))
    if a.dump:
        r = runs[1]
        prof = {k: r["prof"][k].numpy() for k in
                ("rho", "p", "vx", "vy", "vz", "By", "Bz")}
        xi = (-0.5 + (np.arange(r["nc"]) + 0.5) / r["nc"]) / r["t"]
        np.savez_compressed(
            a.dump, xi=xi, Ptot=r["Ptot"], t=r["t"], nc=r["nc"],
            group=pop["group"], gidx=pop["gidx"], Bn=Bn, UL=pop["UL"],
            UR=pop["UR"], fspeed=fspeed, fwidth=fwidth, ffam=ffam, fstr=fstr,
            known_ok=known_ok, known_spd=known_spd, known_str=known_str,
            tube_ok=tube_ok, tube_read_nrm=read_nrm, **prof)
        log("dumped %d fine profiles to %s" % (n, a.dump))
    log("wrote %s" % a.out)


# ── summary ──────────────────────────────────────────────────────────────

def summarise(d):
    fs = sorted(glob.glob(os.path.join(d, "shard_*.npz")))
    if not fs:
        raise SystemExit("no shards in %s" % d)
    Z = [np.load(f) for f in fs]
    cat = lambda k: np.concatenate([z[k] for z in Z])
    grp, counts, unas = cat("group"), cat("counts"), cat("unassigned")
    nfeat, ffam, tube_ok = cat("nfeat"), cat("ffam"), cat("tube_ok")
    conf = (ffam >= 0).sum(axis=1)
    print("tube features: %d shards, %d lanes, %d + %d cells, t = %.3f"
          % (len(fs), grp.size, int(Z[0]["ncells"]), 2 * int(Z[0]["ncells"]),
             float(Z[0]["t_reached"])))
    print("\n  %-14s %5s  %-9s  %-9s  %s" % ("group", "n", "confirmed",
                                             "raw lo/hi", "features per family "
                                             "(mean +- sd)"))
    for gi, g in enumerate(GROUPS):
        m = grp == gi
        if not m.any():
            continue
        per = "  ".join("%s %.2f+-%.2f" % (FAMILIES[k], counts[m, k].mean(),
                                           counts[m, k].std()) for k in range(7))
        print("  %-14s %5d  %4.2f+-%.2f  %4.1f/%4.1f   %s"
              % (g, m.sum(), conf[m].mean(), conf[m].std(),
                 nfeat[m, 0].mean(), nfeat[m, 1].mean(), per))
    sure = cat("counts_sure")
    print("\n  %-14s %10s %10s %8s %8s %8s" % (
        "group", "2+ in one", "2+ unambig.", "unassigned", ">7 conf.",
        "tube->Newton"))
    for gi, g in enumerate(GROUPS):
        m = grp == gi
        if not m.any():
            continue
        print("  %-14s %9.1f%% %9.1f%% %7.1f%% %7.1f%% %7.1f%%"
              % (g, 100 * (counts[m] > 1).any(axis=1).mean(),
                 100 * (sure[m] > 1).any(axis=1).mean(),
                 100 * (unas[m] > 0).mean(),
                 100 * (conf[m] > 7).mean(), 100 * tube_ok[m].mean()))
    print("\n  features seen only at the FINE resolution (weak waves the coarse"
          " run misses), mean per lane:")
    for gi, g in enumerate(GROUPS):
        m = grp == gi
        if m.any():
            print("  %-14s %.2f   (coarse %.2f, fine %.2f, confirmed %.2f)"
                  % (g, (nfeat[m, 1] - conf[m]).mean(), nfeat[m, 0].mean(),
                     nfeat[m, 1].mean(), conf[m].mean()))
    if "known_ok" in Z[0]:
        ko, ks, kseen = cat("known_ok"), cat("known_str"), cat("known_seen")
        st, sn = ks[ko].ravel(), kseen[ko].ravel()
        print("\n  DETECTOR SENSITIVITY -- waves of a KNOWN answer (%d lanes, "
              "%d waves), family actually assigned:" % (ko.sum(), st.size))
        print("  %-20s %7s %9s" % ("wave strength", "waves", "found"))
        for lo, hi in ((0, 1e-4), (1e-4, 1e-3), (1e-3, 1e-2), (1e-2, 1e-1),
                       (1e-1, 1.0), (1.0, np.inf)):
            m = (st >= lo) & (st < hi)
            if m.any():
                print("  [%7.0e, %7.0e)   %7d %8.1f%%" % (lo, hi, m.sum(),
                                                          100 * sn[m].mean()))
        na, cuts = cat("n_above"), Z[0]["strength_cuts"]
        print("\n  ... and counted: confirmed features %.2f per lane against "
              "the number of waves above" % conf[ko].mean())
        print("  " + "   ".join("S=%.0e: %.2f" % (c, na[ko, i].mean())
                                for i, c in enumerate(cuts)))
        rn = cat("tube_read_nrm")
        print("\n  seven-wave residual AT the tube read (median):  "
              + "   ".join("%s %.1e" % (g, np.median(rn[(grp == gi) & np.isfinite(rn)]))
                           for gi, g in enumerate(GROUPS)
                           if ((grp == gi) & np.isfinite(rn)).any()))

    if "fkind" in Z[0]:
        fk, fbt = cat("fkind"), cat("fbtmin")
        print("\n  IS EVERY FEATURE AN ELEMENTARY WAVE?  (a rotation must leave"
              " rho, P and |Bt| alone;\n  a magnetosonic wave must not turn the"
              " field -- anything else is compound/intermediate)")
        print("  %-14s %7s %10s %10s %12s %12s" % (
            "group", "lanes", "rotation", "reversing", "mixed rot.",
            "|Bt|->0 inside"))
        for gi, g in enumerate(GROUPS):
            m = grp == gi
            if not m.any():
                continue
            print("  %-14s %7d %9.1f%% %9.1f%% %11.1f%% %11.1f%%" % (
                g, m.sum(), 100 * (fk[m] == 1).any(axis=1).mean(),
                100 * (fk[m] == 2).any(axis=1).mean(),
                100 * (fk[m] == 3).any(axis=1).mean(),
                100 * ((fbt[m] < 0.1) & (fk[m] >= 0)).any(axis=1).mean()))

    print("\n  doubled families, by family (share of lanes in the group):")
    for gi, g in enumerate(GROUPS):
        m = grp == gi
        if m.any():
            print("  %-14s %s" % (g, "  ".join(
                "%s %4.1f%%" % (FAMILIES[k], 100 * np.mean(counts[m, k] > 1))
                for k in range(7))))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("lanes", nargs="?", help="a coplanar_limit output directory")
    ap.add_argument("--n", type=int, default=100, help="lanes per group (0 = all)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--out")
    ap.add_argument("--ncells", type=int, default=1024,
                    help="coarse resolution; the fine run uses twice this")
    ap.add_argument("--tend", type=float, default=0.30)
    ap.add_argument("--limiter", default="mc", choices=["mc", "minmod", "pcm"])
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--max-iter", type=int, default=60)
    ap.add_argument("--thresh", type=float, default=2e-3,
                    help="a bin is active above this fraction of the column's "
                         "total variation (floored at 3x the median bin)")
    ap.add_argument("--debug", type=int, default=0,
                    help="print the features of the first N lanes and stop")
    ap.add_argument("--match-tol", type=float, default=0.02,
                    help="speed tolerance when matching the two resolutions")
    ap.add_argument("--speed-tol", type=float, default=0.01,
                    help="slack when testing a family's speed interval")
    ap.add_argument("--pick", help="comma-separated gidx values: run exactly "
                                   "these lanes (for --dump)")
    ap.add_argument("--dump", help="save the fine profiles of the lanes run, "
                                   "for plotting elsewhere")
    ap.add_argument("--summarise", metavar="DIR")
    a = ap.parse_args()
    if a.summarise:
        summarise(a.summarise)
        return
    if not a.lanes or not a.out:
        ap.error("need the coplanar_limit dir and --out (or --summarise DIR)")
    run_shard(a)


if __name__ == "__main__":
    main()
