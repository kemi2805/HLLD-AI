"""The eps-regularised coplanar limit (science plan, item B).

The question
------------
The rotor's stubborn interfaces -- the seven-wave AND the planar solver both
fail -- are coplanar to ~1e-11.  The premise was that the seven-wave
formulation is badly posed there, the three angle unknowns (psi_CD, phi_L,
phi_R) being unobservable with the tangential fields parallel.  So tilt the
problem out of its plane by a small angle eps, solve it, and follow the
solution as eps -> 0.  What the branch does in that limit is the evidence the
brute-force box could not give (a grid residual says nothing about existence
-- see what_we_solve.md section 5).

MEASURED 2026-09-19 (pilot, 64 + 64 lanes, calea jobs 533-539), and it cuts
against the premise: on controls the 6x6 condition number stays ~21 all the
way down to eps = 1e-6 (the angles are observable at finite |B_t|; only
|B_t| -> 0 loses them), and tilting the stubborn lanes does not make them
solvable -- from the production warm start 22-33% start at any eps from
0.03 to 1 rad (fields up to 2 rad apart), against 86-98% of controls.

Construction
------------
In the planar frame (``planar5_b.to_planar``: the left tangential field along
+y) the left tangential FIELD is rotated by +eps and the right one by -eps
about the normal, as a rotation matrix, so ``|B_t|``, rho, gas pressure and
velocity are unchanged and eps = 0 is the planar-frame problem bit for bit.
The total pressure the solver takes is moved by the change in b^2, which is
O(eps^2).  The fields then make an angle of 2 eps (pi - 2 eps across a
reversal), far above ``classify.TOL_COPLANAR = 1e-12`` on every rung, so the
tilted problem is an ordinary FULL7 one.

The ladder
----------
eps = 1e-1, 3e-2, 1e-2, ..., 1e-6.  The first rung is seeded the production
way (primary checkpoint, retries seeded by the ensemble, per-lane keys), each
later rung by the previous rung's answer.  A lane that fails a rung walks to
it again from its last converged eps in 2, then 4, then 8 geometric
sub-steps before it is declared lost; ``eps_star`` is the smallest eps it
reached.  At every converged point the finite-difference 6x6 Jacobian's
condition number is kept, because a fold shows up as kappa -> inf as
eps -> eps_star, and a plain Newton failure does not.

Every lane that started is then polished at eps = 0 (the untilted problem)
three ways, first verified wins: the seven-wave Newton from the limit
unknowns as they are; the same with psi_CD, phi_L, phi_R snapped to {0, pi};
and the planar solver seeded from the limit with the flips the limit's
rotations imply.  Verified means what production means by it: full
seven-wave residual <= 1e-8 (RMHD_VERIFY_TOL) AND waves in order
(``exact_flux._self_crossing``, the production rule -- a zero residual is
not a solution).

Classes
-------
  NOSTART   the first rung never converged: no information
  LOST      started, lost below eps_star.  Evidence for a fold -- (iii), no
            continuous limit -- only if kappa grows toward eps_star; the
            summary prints that ratio (the pilot's were all 1.00: Newton
            losses, not folds)
  SINGULAR  (ii)  reached eps_min, but eps = 0 does not verify: the limit of
                  the tilted family is not a solution of the coplanar problem
  EXACT0    (i)   verifies at eps = 0 with rotations at {0, pi}: an
                  elementary solution the planar Newton missed
  EXACTROT  (i')  verifies at eps = 0 with an intermediate rotation
A LOST lane whose polish verifies anyway is counted separately.

The branch matters: a seed can put a lane on a branch of the TILTED problem
that does not reach the planar answer (pilot: 2 of 58 controls, rotations
near pi at eps = 0.1, both ending SINGULAR), so a SINGULAR verdict is a
statement about the branch followed, not yet about the interface.

The validation gate
-------------------
Run on CONTROL lanes the planar solver solves (the same 1280 + 1280 lanes as
brute_force_stubborn.py, read from its output, so the two are paired lane by
lane).  The ladder must reach eps_min and approach the planar answer as
eps -> 0.  No state tolerance is fixed in advance: a 1e-8 residual pins the
seven-wave state only to ~kappa x residual, and kappa reaches 7.9e3 on
coplanar lanes (measured 2026-09-09, residual-is-not-a-state-tolerance), so
``--summarise`` prints the state difference per rung beside kappa x residual
and the gate is judged from those numbers.  If the controls do not converge
to their planar answers, nothing the stubborn lanes do counts.

State differences use the three-way split of ``exact_flux.relative_jump``
(rho and P_tot relative, v as an absolute vector, B as a vector against the
field and pressure scale), never a per-component relative difference with a
floor: vz and Bz are zero in a planar flow.

Usage (calea, through Slurm: scripts/calea_coplanar_limit.sh)
    python scripts/coplanar_limit.py results/brute_hlld --n 64 \
        --shard 0 --nshards 64 --out results/coplanar_limit/shard_000.npz
    python scripts/coplanar_limit.py --summarise results/coplanar_limit
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

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)
from rmhd.eos import set_eos                                       # noqa: E402
set_eos("ideal")
from rmhd import ml_guess as mg, paths                             # noqa: E402
from rmhd.batched import fullcontact_b as FB                       # noqa: E402
from rmhd.batched import ml_b as MB                                # noqa: E402
from rmhd.batched import planar5_b as P5                           # noqa: E402
from rmhd.util.angles import wrap                                  # noqa: E402
from src.physics.exact_flux import _self_crossing                  # noqa: E402

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

GAMMA = 5.0 / 3.0
EPS_LADDER = (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6)
SUBSTEPS = (2, 4, 8)
ACC = 1e-10             # Newton target on every rung and in the polish
VERIFY = 1e-8           # what counts as solved: exact_flux's RMHD_VERIFY_TOL, also
                        # the rung acceptance -- the pilot lost 4 controls at
                        # rung 0 with residuals 6e-9..7e-5 under a 1e-10 bar
WAVE_ORDER_TOL = 1e-6   # exact_flux's RMHD_WAVE_ORDER_TOL
ROT_ZERO, ROT_PI = 1e-6, 1e-3      # brute_force_stubborn's rotation classes
CLASSES = ("NOSTART", "LOST", "SINGULAR", "EXACT0", "EXACTROT")
GROUPS = ("control", "stubborn")
DEFAULT_CKPT = "data/ml_guess_rotor_ft.pt"
DEFAULT_EXTRA = "data/ml_guess_rotor_big_s42.pt,data/ml_guess_rotor_big_s47.pt"


# ── states ───────────────────────────────────────────────────────────────

def sub(state, i):
    return [c[i] for c in state]


def b2_of(s, Bn):
    """Comoving b^2, as classify.alfven_speeds and the solver compute it."""
    _, _, vx, vy, vz, By, Bz = s
    v2 = vx * vx + vy * vy + vz * vz
    eta = Bn * vx + By * vy + Bz * vz
    return (Bn * Bn + By * By + Bz * Bz) * (1.0 - v2) + eta * eta


def tilt(s, Bn, ang):
    """Rotate the tangential field by ``ang`` about the normal.

    rho, gas pressure and v are held; the total pressure moves with b^2.
    ``ang = 0`` returns the input bit for bit.
    """
    out = [np.array(c, dtype=float, copy=True) for c in s]
    c, sn = np.cos(ang), np.sin(ang)
    out[5] = c * s[5] - sn * s[6]
    out[6] = sn * s[5] + c * s[6]
    out[1] = s[1] + 0.5 * (b2_of(out, Bn) - b2_of(s, Bn))
    return out


def tilted(L, R, Bn, eps):
    return tilt(L, Bn, eps), tilt(R, Bn, -eps)


def state_diff(a, b, Bn):
    """Three-way split of exact_flux.relative_jump, on solver-frame states."""
    tiny = 1e-30
    d = np.abs(a[0] - b[0]) / (np.abs(a[0]) + np.abs(b[0]) + tiny)
    d = np.maximum(d, np.abs(a[1] - b[1]) / (np.abs(a[1]) + np.abs(b[1]) + tiny))
    d = np.maximum(d, np.sqrt(sum((a[k] - b[k]) ** 2 for k in (2, 3, 4))))
    dB = np.hypot(a[5] - b[5], a[6] - b[6])
    scale = (np.sqrt(Bn * Bn + a[5] ** 2 + a[6] ** 2)
             + np.sqrt(Bn * Bn + b[5] ** 2 + b[6] ** 2)
             + np.sqrt(np.abs(a[1]) + np.abs(b[1])))
    return np.maximum(d, dB / (scale + tiny))


def zones_diff(Z7, Z4, Bn):
    """Seven-wave zones R2..R7 against planar zones [A, B, C, D]:
    R2~A, R4~B, R5~C, R7~D (the rotations of a planar answer are silent)."""
    return np.max(np.stack([state_diff(Z7[a], Z4[b], Bn)
                            for a, b in ((0, 0), (2, 1), (3, 2), (5, 3))]), axis=0)


def signature(L, R, Z, tol=1e-10):
    """Wave types [LF, LS, RS, RF]: +1 shock, -1 rarefaction, 0 no wave."""
    R2, R3, R4, R5, R6, R7 = Z
    bt = lambda s: np.hypot(s[5], s[6])

    def sgn(up, ahead):                       # up > 0: a shock
        return np.where(np.abs(up) <= tol * np.maximum(np.abs(ahead), 1e-30),
                        0, np.sign(up)).astype(np.int8)
    return np.stack([sgn(R2[1] - L[1], L[1]),          # fast: p rises
                     sgn(bt(R3) - bt(R4), bt(R3)),      # slow: |Bt| falls
                     sgn(bt(R6) - bt(R5), bt(R6)),
                     sgn(R7[1] - R[1], R[1])], axis=-1)


def rot_class(phiL, phiR):
    """0: both rotations 0 or pi; 1: an intermediate rotation (brute's rule)."""
    out = np.zeros(phiL.shape, dtype=np.int8)
    for p in (phiL, phiR):
        a = np.abs(wrap(p))
        out |= ((a >= ROT_ZERO) & (np.abs(a - np.pi) >= ROT_PI)).astype(np.int8)
    return out


def snap(a):
    """Nearest of {0, pi}."""
    return np.where(np.abs(wrap(a)) <= 0.5 * np.pi, 0.0, np.pi)


# ── solver pieces ────────────────────────────────────────────────────────

def solved(r):
    """Rung acceptance: residual at production's verification threshold."""
    return np.asarray(r["nrm"]) <= VERIFY


def admissible(r):
    return ~_self_crossing(r["zones"], r["VsLv"], r["VsRv"], WAVE_ORDER_TOL)


def kappa(F, L, R, Bn, unk):
    if Bn.size == 0:
        return np.zeros(0)
    _, J, _, _, _, err = F["fullfuncd6"](L, R, unk, Bn)
    with np.errstate(all="ignore"):
        k = np.linalg.cond(J)
    return np.where(err | ~np.isfinite(k), np.inf, k)


def load_models(ckpt, extra):
    ms = [mg.load(str(paths.resolve(ckpt)))]
    for c in [c.strip() for c in extra.split(",") if c.strip()]:
        ms.append(mg.load(str(paths.resolve(c))))
    return ms


def warm_start(L, R, Bn, models):
    """exact_flux_batched's seeding: primary candidate 0, the ensemble for
    the retries, per-lane keys from each lane's own features."""
    UL, UR = np.stack(L, axis=1), np.stack(R, axis=1)
    (m0, s0), rest = models[0], models[1:]
    cands, feats = MB.predict_unk6_k(m0, s0, UL, UR, Bn)
    seed, _ = MB.clamp_seed_physical(cands[0], L, R, Bn)
    extra = [MB.clamp_seed_physical(u, L, R, Bn)[0] for u in cands[1:]]
    for m, s in rest:
        u, _ = MB.predict_unk6(m, s, UL, UR, Bn)
        extra.append(MB.clamp_seed_physical(u, L, R, Bn)[0])
    return seed, (extra or None), MB.lane_keys(feats)


# ── the ladder ───────────────────────────────────────────────────────────

class Record:
    """Per-lane, per-rung arrays."""

    def __init__(self, n, K):
        self.conv = np.zeros((n, K), bool)
        self.nrm = np.full((n, K), np.inf)
        self.nit = np.zeros((n, K), np.int16)
        self.unk = np.full((n, K, 6), np.nan)
        self.zones = np.full((n, K, 6, 7), np.nan)
        self.VsL = np.full((n, K, 3), np.nan)
        self.VsR = np.full((n, K, 3), np.nan)
        self.kappa = np.full((n, K), np.nan)
        self.adm = np.zeros((n, K), bool)
        self.sig = np.zeros((n, K, 4), np.int8)
        self.nsub = np.zeros((n, K), np.int8)

    def put(self, F, k, idx, r, sel, Lt, Rt, Bn, nsub=1):
        """Store result ``r`` rows ``sel`` for global lanes ``idx``; Lt/Rt/Bn
        are the tilted problems of exactly those rows."""
        if idx.size == 0:
            return
        u = [c[sel] for c in r["unk"]]
        Z = [[c[sel] for c in z] for z in r["zones"]]
        VL = [c[sel] for c in r["VsLv"]]
        VR = [c[sel] for c in r["VsRv"]]
        self.conv[idx, k] = solved(r)[sel]
        self.nrm[idx, k] = r["nrm"][sel]
        self.nit[idx, k] = r["n_iter"][sel]
        self.unk[idx, k] = np.stack(u, axis=1)
        self.zones[idx, k] = np.stack([np.stack(z, axis=1) for z in Z], axis=1)
        self.VsL[idx, k] = np.stack(VL, axis=1)
        self.VsR[idx, k] = np.stack(VR, axis=1)
        self.adm[idx, k] = ~_self_crossing(Z, VL, VR, WAVE_ORDER_TOL)
        self.kappa[idx, k] = kappa(F, Lt, Rt, Bn, u)
        self.sig[idx, k] = signature(Lt, Rt, Z)
        self.nsub[idx, k] = nsub


def walk(F, L0, R0, Bn, u_from, e_from, e_to, nsub, max_iter):
    """Step lanes from ``e_from`` (per lane) to ``e_to`` in ``nsub`` geometric
    steps, each seeded by the last.  Returns ``(reached, u, e_last, (i, r))``:
    ``u``/``e_last`` the furthest converged point of every lane, ``r`` the
    final-step result on the lanes ``i`` that attempted it."""
    m = Bn.size
    u = [c.copy() for c in u_from]
    e_last = e_from.copy()
    going = np.ones(m, bool)
    last = (np.zeros(0, int), None)
    for j in range(1, nsub + 1):
        i = np.flatnonzero(going)
        if i.size == 0:
            break
        e = (np.full(i.size, e_to) if j == nsub
             else e_from[i] * (e_to / e_from[i]) ** (j / nsub))
        Lt, Rt = tilted(sub(L0, i), sub(R0, i), Bn[i], e)
        r = F["fullcontact6"](Lt, Rt, sub(u, i), Bn[i], accuracy=ACC,
                              max_iter=max_iter)
        ok = solved(r)
        for c in range(6):
            u[c][i[ok]] = r["unk"][c][ok]
        e_last[i[ok]] = e[ok]
        going[i[~ok]] = False
        if j == nsub:
            last = (i, r)
    return going, u, e_last, last


def ladder(F, models, L0, R0, Bn, eps, retries, max_iter, log):
    n, K = Bn.size, len(eps)
    rec = Record(n, K)
    all_i = np.arange(n)
    Lt, Rt = tilted(L0, R0, Bn, eps[0])
    seed, extra, keys = warm_start(Lt, Rt, Bn, models)
    r = MB.solve_with_retries(F["fullcontact6"], Lt, Rt, Bn, seed, keys,
                              accuracy=ACC, max_iter=max_iter,
                              n_retries=retries, seeds_extra=extra)
    rec.put(F, 0, all_i, r, all_i, Lt, Rt, Bn)
    attempts = np.asarray(r.get("attempts", np.ones(n, int)))
    alive = solved(r).copy()
    cur = [np.array(c, dtype=float, copy=True) for c in r["unk"]]
    e_last = np.where(alive, eps[0], np.nan)
    log("rung 0  eps=%.0e  converged %d/%d" % (eps[0], alive.sum(), n))

    for k in range(1, K):
        i = np.flatnonzero(alive)
        if i.size == 0:
            break
        Lt, Rt = tilted(sub(L0, i), sub(R0, i), Bn[i], eps[k])
        r = F["fullcontact6"](Lt, Rt, sub(cur, i), Bn[i], accuracy=ACC,
                              max_iter=max_iter)
        ok = solved(r)
        g = np.flatnonzero(ok)
        rec.put(F, k, i[g], r, g, sub(Lt, g), sub(Rt, g), Bn[i[g]])
        for c in range(6):
            cur[c][i[g]] = r["unk"][c][g]
        e_last[i[g]] = eps[k]
        fail = i[~ok]
        n_direct = g.size
        for ns in SUBSTEPS:
            if fail.size == 0:
                break
            reached, u, el, (ii, rf) = walk(F, sub(L0, fail), sub(R0, fail),
                                            Bn[fail], sub(cur, fail),
                                            e_last[fail], eps[k], ns, max_iter)
            for c in range(6):
                cur[c][fail] = u[c]
            e_last[fail] = el
            if rf is not None and bool(solved(rf).any()):
                okf = np.flatnonzero(solved(rf))
                gl = fail[ii[okf]]
                Lg, Rg = tilted(sub(L0, gl), sub(R0, gl), Bn[gl], eps[k])
                rec.put(F, k, gl, rf, okf, Lg, Rg, Bn[gl], nsub=ns)
            fail = fail[~reached]
        alive[fail] = False
        log("rung %d  eps=%.0e  %d direct + %d by sub-steps, %d lost"
            % (k, eps[k], n_direct, i.size - n_direct - fail.size, fail.size))
    return rec, alive, cur, e_last, attempts


def polish(F, S5, L0, R0, Bn, u, max_iter):
    """Solve the UNTILTED problem from the limit unknowns ``u``; see the
    module docstring.  Returns route (0 none, 1 as-is, 2 snapped, 3 planar),
    the verified unk6, residual, zones, speeds; all per lane."""
    n = Bn.size
    route = np.zeros(n, np.int8)
    unk = np.full((n, 6), np.nan)
    nrm = np.full(n, np.inf)
    zones = np.full((n, 6, 7), np.nan)
    VsL = np.full((n, 3), np.nan)
    VsR = np.full((n, 3), np.nan)

    def take(sel, r, rows, code, u6=None):
        """Lanes ``sel`` (global) take rows ``rows`` of result ``r``."""
        route[sel] = code
        src = r["unk"] if u6 is None else u6
        unk[sel] = np.stack([c[rows] for c in src], axis=1)
        nrm[sel] = r["nrm"][rows]
        zones[sel] = np.stack([np.stack([c[rows] for c in z], axis=1)
                               for z in r["zones"]], axis=1)
        VsL[sel] = np.stack([c[rows] for c in r["VsLv"]], axis=1)
        VsR[sel] = np.stack([c[rows] for c in r["VsRv"]], axis=1)

    seeds = (u, [u[0], u[1], snap(u[2]), u[3], snap(u[4]), snap(u[5])])
    for code, s in ((1, seeds[0]), (2, seeds[1])):
        i = np.flatnonzero(route == 0)
        if i.size == 0:
            break
        r = F["fullcontact6"](sub(L0, i), sub(R0, i), sub(s, i), Bn[i],
                              accuracy=ACC, max_iter=max_iter)
        good = np.flatnonzero((r["nrm"] <= VERIFY) & admissible(r))
        take(i[good], r, good, code)

    # the planar solver, seeded from the limit, flips from its rotations
    sgn = np.where(np.cos(u[2]) >= 0.0, 1.0, -1.0)
    u3 = [u[0], sgn * np.exp(u[1]), u[3]]
    fl = np.stack([np.abs(wrap(u[4])) > 0.5 * np.pi,
                   np.abs(wrap(u[5])) > 0.5 * np.pi], axis=1)
    for fL in (False, True):
        for fR in (False, True):
            i = np.flatnonzero((route == 0) & (fl[:, 0] == fL) & (fl[:, 1] == fR))
            if i.size == 0:
                continue
            p = S5["solve"](sub(L0, i), sub(R0, i), Bn[i], unk3_init=sub(u3, i),
                            accuracy=ACC, max_iter=max_iter, flip=(fL, fR))
            # the equivalent seven-wave unknowns, as planar5_b.verify_full
            # writes them, give the zones and speeds the order gate needs
            Bt = p["unk"][1]
            u6 = [p["unk"][0], np.log(np.maximum(np.abs(Bt), 1e-30)),
                  np.where(Bt >= 0.0, 0.0, np.pi), p["unk"][2],
                  np.full(i.size, np.pi if fL else 0.0),
                  np.full(i.size, np.pi if fR else 0.0)]
            fv, Z, VL, VR, e = F["fullfuncv6"](sub(L0, i), sub(R0, i), u6, Bn[i])
            r = dict(unk=u6, zones=Z, VsLv=VL, VsRv=VR,
                     nrm=np.where(e, np.inf, np.max(np.abs(fv), axis=0)))
            good = np.flatnonzero(p["converged"] & ~e & admissible(r))
            take(i[good], r, good, 3, u6)
    return route, unk, nrm, zones, VsL, VsR


# ── population ───────────────────────────────────────────────────────────

def load_population(brute_dir, n_cap, seed=0):
    """The brute-force run's lanes, in their original order, capped per group
    by a fixed-seed random subset (the original order is harvest order, so a
    plain head would be one stretch of one sweep)."""
    fs = sorted(glob.glob(os.path.join(brute_dir, "shard_*.npz")))
    if not fs:
        raise SystemExit("no shard_*.npz in %s" % brute_dir)
    ns = len(fs)
    out = {}
    for gi, (g, key) in enumerate((("control", "control"), ("stubborn", "stub"))):
        parts = []
        for s, f in enumerate(fs):
            z = np.load(f)
            m = z["%s_UL" % key].shape[0]
            parts.append(dict(orig=s + ns * np.arange(m),
                              UL=z["%s_UL" % key], UR=z["%s_UR" % key],
                              best=z["%s_best" % key], unk=z["%s_unk" % key]))
        cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        o = np.argsort(cat["orig"], kind="stable")
        cat = {k: v[o] for k, v in cat.items()}
        N = cat["orig"].size
        if n_cap and n_cap < N:
            pick = np.sort(np.random.default_rng(seed + gi).choice(N, n_cap,
                                                                   replace=False))
            cat = {k: v[pick] for k, v in cat.items()}
        cat["group"] = np.full(cat["orig"].size, gi, np.int8)
        out[g] = cat
    return {k: np.concatenate([out[g][k] for g in GROUPS]) for k in out["control"]}


def brute_verdict(F, UL, UR, best, unk):
    """Brute force's answer, re-gated: residual <= VERIFY and waves in order
    (its runs predate the in-loop gate).  Original frame, as it ran."""
    L = [UL[:, j].copy() for j in range(7)]
    R = [UR[:, j].copy() for j in range(7)]
    Bn = UL[:, 7].copy()
    ok = np.isfinite(best) & (best <= VERIFY)
    zones = np.full((Bn.size, 6, 7), np.nan)
    i = np.flatnonzero(ok)
    if i.size:
        u = [unk[i, c].copy() for c in range(6)]
        fv, Z, VL, VR, e = F["fullfuncv6"](sub(L, i), sub(R, i), u, Bn[i])
        adm = ~e & ~_self_crossing(Z, VL, VR, WAVE_ORDER_TOL)
        ok[i] = adm
        zones[i] = np.stack([np.stack(z, axis=1) for z in Z], axis=1)
    return ok, zones


def production_retest(UL, UR, models, max_iter):
    """Does the CURRENT production path solve this interface?

    The lanes come from a run that predates the Alfven sign fix (rmhd
    d09ffe2), which changed who is stubborn: on Bn < 0 it took the 32^2
    rotor's exact fluxes from 15,113 to 39,183.  So "stubborn" has to be
    re-measured before anything is concluded from it.  Mirrors
    exact_flux_batched: original frame, clamped ML seed, the ensemble on two
    retries, accuracy 1e-8, then the production wave-order gate.  The planar
    solver is checked by the caller (its verified answer is a production
    rescue too).
    """
    from rmhd.batched import api as API
    left = [UL[:, j].copy() for j in range(7)]
    right = [UR[:, j].copy() for j in range(7)]
    Bn = UL[:, 7].copy()
    seed, extra, keys = warm_start(left, right, Bn, models)
    res, diag = API.make_solver(GAMMA)(left, right, Bn, seed6=seed, keys=keys,
                                       accuracy=1e-8, max_iter=max_iter,
                                       n_retries=2, seeds_extra=extra)
    ok = np.asarray(res["converged"], bool) & admissible(res)
    return ok, np.asarray(res["cls"]), np.asarray(res["attempts"])


def rh_defect(UL, UR):
    """How close an interface is to ONE discontinuity: the best single speed
    ``s`` and the Rankine-Hugoniot defect ``|dF - s dU| / |dU|``.

    Lab-frame conserved variables and fluxes from HLLD's own
    ``compute_srmhd_fluxes``, each component scaled by its magnitude on the
    two sides so no variable dominates; ``s`` is the least-squares speed in
    that scaling and the defect is relative to the jump, so 0 means (L, R)
    are exactly one discontinuity and O(1) means several waves.  Necessary,
    not sufficient, for "no solve needed": an expansion shock satisfies it too.
    """
    import torch
    from src.physics.eos import hybrid_eos
    from src.physics.exact_flux import from_solver_frame
    from src.physics.hlld import compute_srmhd_fluxes
    eos = hybrid_eos(K=0.0, gamma=GAMMA, gamma_th=GAMMA)
    T = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float64)
    sL = from_solver_frame([T(UL[:, j]) for j in range(7)], T(UL[:, 7]), eos, 0, None)
    sR = from_solver_frame([T(UR[:, j]) for j in range(7)], T(UR[:, 7]), eos, 0, None)
    uL, uR, fL, fR, _, _ = compute_srmhd_fluxes(sL, sR, eos, 0)
    keys = list(uL)
    col = lambda d: np.stack([d[k].detach().numpy() for k in keys], axis=1)
    c = (np.abs(col(uL)) + np.abs(col(uR)) + np.abs(col(fL)) + np.abs(col(fR))
         + 1e-300)
    a, b = (col(uR) - col(uL)) / c, (col(fR) - col(fL)) / c
    s = (a * b).sum(1) / np.maximum((a * a).sum(1), 1e-300)
    defect = (np.abs(b - s[:, None] * a).max(1)
              / np.maximum(np.abs(a).max(1), 1e-300))
    return s, defect


# ── one shard ────────────────────────────────────────────────────────────

def _git(path):
    try:
        return subprocess.run(["git", "-C", path, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except OSError:
        return "?"


def run_shard(a):
    t0 = time.time()
    tag = "[%s shard %d/%d]" % (socket.gethostname(), a.shard, a.nshards)
    log = lambda s: print("%s %6.0fs  %s" % (tag, time.time() - t0, s), flush=True)
    pop = load_population(a.brute, a.n)
    sel = np.arange(pop["orig"].size)[a.shard::a.nshards]
    pop = {k: v[sel] for k, v in pop.items()}
    n = sel.size
    log("%d lanes (%d control, %d stubborn) from %s"
        % (n, (pop["group"] == 0).sum(), (pop["group"] == 1).sum(), a.brute))

    F = FB.make_solver(GAMMA)
    S5 = P5.make_solver(GAMMA)
    models = load_models(a.ckpt, a.extra_ckpts)
    eps = np.array(a.eps, dtype=float)

    UL, UR = pop["UL"], pop["UR"]
    Bn = UL[:, 7].copy()
    L0, R0, alpha, presid = P5.to_planar([UL[:, j].copy() for j in range(7)],
                                         [UR[:, j].copy() for j in range(7)], Bn)
    reversal = R0[5] < 0.0

    # the planar reference (default flips, as production and brute ran it)
    ref = S5["solve"](L0, R0, Bn, accuracy=ACC, max_iter=a.max_iter)
    ref_zones = np.stack([np.stack(z, axis=1) for z in ref["zones"]], axis=1)
    log("planar reference: %d/%d verified" % (ref["converged"].sum(), n))
    brute_ok, brute_zones = brute_verdict(F, UL, UR, pop["best"], pop["unk"])
    rh_s, rh_def = rh_defect(UL, UR)
    prod_ok, prod_cls, prod_att = production_retest(UL, UR, models, a.max_iter)
    log("production re-test at the current solver: %d/%d solve (%d of them "
        "were the stubborn group)" % (prod_ok.sum(), n,
                                      (prod_ok & (pop["group"] == 1)).sum()))
    t_ref = time.time() - t0

    rec, reached, cur, eps_star, attempts = ladder(
        F, models, L0, R0, Bn, eps, a.retries, a.max_iter, log)
    t_ladder = time.time() - t0 - t_ref

    started = rec.conv[:, 0]
    pol = [np.zeros(n, np.int8), np.full((n, 6), np.nan), np.full(n, np.inf),
           np.full((n, 6, 7), np.nan), np.full((n, 3), np.nan),
           np.full((n, 3), np.nan)]
    i = np.flatnonzero(started)
    if i.size:
        out = polish(F, S5, sub(L0, i), sub(R0, i), Bn[i], sub(cur, i),
                     a.max_iter)
        for dst, src in zip(pol, out):
            dst[i] = src
    route, p_unk, p_nrm, p_zones, p_VsL, p_VsR = pol
    verified = route > 0
    rot = np.where(verified, rot_class(p_unk[:, 4], p_unk[:, 5]), 0)
    cls = np.where(~started, 0, np.where(~reached, 1, np.where(
        ~verified, 2, np.where(rot == 0, 3, 4)))).astype(np.int8)
    log("classes: " + "  ".join("%s %d" % (c, (cls == k).sum())
                                for k, c in enumerate(CLASSES))
        + "   (LOST but verified at 0: %d)" % ((cls == 1) & verified).sum())

    # state differences: ladder vs planar reference, polish vs reference,
    # polish vs brute's root (brute is in the ORIGINAL frame: rotate ours back)
    K = eps.size
    d_ref = np.full((n, K), np.nan)
    for k in range(K):
        Z7 = [[rec.zones[:, k, z, j] for j in range(7)] for z in range(6)]
        Z4 = [[ref_zones[:, z, j] for j in range(7)] for z in range(4)]
        d_ref[:, k] = zones_diff(Z7, Z4, Bn)
    Zp = [[p_zones[:, z, j] for j in range(7)] for z in range(6)]
    d_pol_ref = zones_diff(Zp, [[ref_zones[:, z, j] for j in range(7)]
                                for z in range(4)], Bn)
    Zp_orig = [P5.from_planar(z, alpha) for z in Zp]
    Zb = [[brute_zones[:, z, j] for j in range(7)] for z in range(6)]
    d_pol_brute = np.max(np.stack([state_diff(Zp_orig[z], Zb[z], Bn)
                                   for z in (0, 2, 3, 5)]), axis=0)

    np.savez_compressed(
        a.out, eps=eps, group=pop["group"], orig=pop["orig"], UL=UL, UR=UR,
        Bn=Bn, alpha=alpha, presid=presid, reversal=reversal,
        ref_ok=ref["converged"], ref_unk=np.stack(ref["unk"], axis=1),
        ref_zones=ref_zones, ref_full_resid=ref["full_resid"],
        brute_best=pop["best"], brute_ok=brute_ok, rh_s=rh_s, rh_defect=rh_def,
        prod_ok=prod_ok, prod_cls=prod_cls, prod_att=prod_att,
        conv=rec.conv, nrm=rec.nrm, nit=rec.nit, unk=rec.unk, zones=rec.zones,
        VsL=rec.VsL, VsR=rec.VsR, kappa=rec.kappa, adm=rec.adm, sig=rec.sig,
        nsub=rec.nsub, attempts=attempts, reached=reached, eps_star=eps_star,
        pol_route=route, pol_unk=p_unk, pol_nrm=p_nrm, pol_zones=p_zones,
        pol_VsL=p_VsL, pol_VsR=p_VsR, rot=rot, cls=cls,
        d_ref=d_ref, d_pol_ref=d_pol_ref, d_pol_brute=d_pol_brute,
        t_ref=t_ref, t_ladder=t_ladder, t_total=time.time() - t0,
        host=socket.gethostname(), rmhd_rev=paths.git_rev(), hlld_rev=_git(_HERE),
        script_md5=hashlib.md5(open(__file__, "rb").read()).hexdigest(),
        argv=" ".join(sys.argv))
    log("wrote %s  (reference %.0fs, ladder %.0fs)" % (a.out, t_ref, t_ladder))


# ── summary ──────────────────────────────────────────────────────────────

def _q(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return "      --        --        --"
    return "%9.2e %9.2e %9.2e" % (np.median(x), np.percentile(x, 99), x.max())


def summarise(d):
    fs = sorted(glob.glob(os.path.join(d, "shard_*.npz")))
    if not fs:
        raise SystemExit("no shards in %s" % d)
    Z = [np.load(f, allow_pickle=False) for f in fs]
    cat = lambda k: np.concatenate([z[k] for z in Z])
    eps = Z[0]["eps"]
    grp, cls, reached = cat("group"), cat("cls"), cat("reached")
    conv, nrm, kap = cat("conv"), cat("nrm"), cat("kappa")
    d_ref, ref_ok = cat("d_ref"), cat("ref_ok")
    Bn, rev, eps_star = np.abs(cat("Bn")), cat("reversal"), cat("eps_star")
    route, rot = cat("pol_route"), cat("rot")
    brute_ok = cat("brute_ok")
    tt = np.array([float(z["t_total"]) for z in Z])
    revs = {(str(z["rmhd_rev"]), str(z["hlld_rev"]), str(z["script_md5"])) for z in Z}
    print("coplanar limit: %d shards, %d lanes; rmhd/HLLD/script %s" % (
        len(fs), grp.size, sorted(revs)))
    print("wall per shard: median %.0f s, max %.0f s; %.1f s per lane summed"
          % (np.median(tt), tt.max(), tt.sum() / max(grp.size, 1)))

    # ── the validation gate ──
    g = (grp == 0) & ref_ok
    print("\nVALIDATION GATE -- %d control lanes the planar solver verifies "
          "(of %d controls)" % (g.sum(), (grp == 0).sum()))
    print("  rung  eps      reached   diff to planar: median   p99      max"
          "     kappa*resid: median   p99    kappa median")
    for k, e in enumerate(eps):
        c = g & conv[:, k]
        kr = kap[c, k] * nrm[c, k]
        print("  %2d   %.0e   %4d/%-4d  %s   %s   %9.2e" % (
            k, e, c.sum(), g.sum(), _q(d_ref[c, k]), _q(kr)[:19],
            np.median(kap[c, k]) if c.any() else np.nan))
    v = g & (route > 0)
    print("  eps=0 polish verified: %d/%d (routes as-is/snapped/planar: %d/%d/%d); "
          "diff to planar: %s" % (v.sum(), g.sum(), (g & (route == 1)).sum(),
                                  (g & (route == 2)).sum(), (g & (route == 3)).sum(),
                                  _q(cat("d_pol_ref")[v])))
    print("  controls by class: " + "  ".join(
        "%s %d" % (c, ((grp == 0) & (cls == k)).sum()) for k, c in enumerate(CLASSES)))

    # ── is the lane still stubborn at the current solver? ──
    if "prod_ok" in Z[0]:
        po, ro = cat("prod_ok"), ref_ok
        print("\nSTILL STUBBORN? the same lanes through TODAY's production path")
        for label, m in (("control", grp == 0), ("stubborn", grp == 1)):
            print("  %-10s n=%4d   seven-wave solves %4d (%4.1f%%)   planar verifies "
                  "%4d   either %4d (%4.1f%%)" % (
                      label, m.sum(), (m & po).sum(), 100 * np.mean(po[m]),
                      (m & ro).sum(), (m & (po | ro)).sum(),
                      100 * np.mean((po | ro)[m])))
        st = (grp == 1) & ~(po | ro)
        print("  -> %d of %d former stubborn lanes are still stubborn; %d were the "
              "old solver's doing" % (st.sum(), (grp == 1).sum(),
                                      (grp == 1).sum() - st.sum()))

    # ── the stubborn lanes ──
    s = grp == 1
    print("\nSTUBBORN -- %d lanes (planar re-solved here: %d verified, i.e. not "
          "stubborn on this host)" % (s.sum(), (s & ref_ok).sum()))
    hdr = "  %-22s %5s " + " ".join("%9s" % c for c in CLASSES) + "   LOST+root@0"
    print(hdr % ("", "n"))

    def row(label, m):
        print("  %-22s %5d " % (label, m.sum()) + " ".join(
            "%8.1f%%" % (100 * np.mean(cls[m] == k) if m.any() else 0)
            for k in range(len(CLASSES)))
            + "   %d" % (m & (cls == 1) & (route > 0)).sum())
    row("all", s)
    for lo, hi in ((0, 1e-2), (1e-2, 1e-1), (1e-1, 1.0), (1.0, np.inf)):
        row("|Bn| in [%g, %g)" % (lo, hi), s & (Bn >= lo) & (Bn < hi))
    row("field reversal", s & rev)
    row("no reversal", s & ~rev)
    f = s & (cls == 1)
    if f.any():         # LOST
        k_last = np.array([kap[j, np.flatnonzero(conv[j])[-1]] for j in np.flatnonzero(f)])
        k_first = kap[f, 0]
        print("  LOST: eps_star %s (median/p99/max);  kappa at the last converged "
              "rung / kappa at rung 0: %s" % (_q(eps_star[f]), _q(k_last / k_first)))
    sig = cat("sig")
    sw = np.array([len({tuple(x) for x, c in zip(sig[j], conv[j]) if c}) > 1
                   for j in range(grp.size)])
    print("  wave-type switch along the ladder: stubborn %d/%d, controls %d/%d"
          % ((s & sw).sum(), s.sum(), ((grp == 0) & sw).sum(), (grp == 0).sum()))

    # ── how close to ONE discontinuity (Rankine-Hugoniot defect) ──
    rd = cat("rh_defect")
    cuts = (1e-8, 1e-6, 1e-3, 1e-2, 1e-1)
    print("\nRANKINE-HUGONIOT DEFECT |dF - s dU|/|dU| (0 = a single discontinuity)")
    print("  %-22s %5s   median     " % ("", "n") + "  ".join("<=%.0e" % c for c in cuts))
    for label, m in (("control", grp == 0), ("stubborn", s)) + tuple(
            ("stubborn " + c, s & (cls == k)) for k, c in enumerate(CLASSES)):
        if m.any():
            print("  %-22s %5d  %8.1e  " % (label, m.sum(), np.median(rd[m]))
                  + "  ".join("%6.1f%%" % (100 * np.mean(rd[m] <= c)) for c in cuts))

    # ── paired with brute force ──
    ex = route > 0
    print("\nPAIRED WITH BRUTE FORCE (its roots re-gated: <=1e-8 and in order)")
    for gi, gname in enumerate(GROUPS):
        m = grp == gi
        both = m & ex & brute_ok
        print("  %-8s  both %4d   eps=0 only %4d   brute only %4d   neither %4d"
              "   same root where both: %d/%d (diff <= 1e-6)" % (
                  gname, both.sum(), (m & ex & ~brute_ok).sum(),
                  (m & ~ex & brute_ok).sum(), (m & ~ex & ~brute_ok).sum(),
                  (both & (cat("d_pol_brute") <= 1e-6)).sum(), both.sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("brute", nargs="?", help="brute_force_stubborn output dir")
    ap.add_argument("--n", type=int, default=0,
                    help="lanes per group, a fixed random subset (0 = all)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--out")
    ap.add_argument("--eps", type=float, nargs="+", default=list(EPS_LADDER))
    ap.add_argument("--retries", type=int, default=2,
                    help="first rung only; production RETRIES")
    ap.add_argument("--max-iter", type=int, default=60)
    ap.add_argument("--ckpt", default=os.environ.get("RMHD_ML_CKPT", DEFAULT_CKPT))
    ap.add_argument("--extra-ckpts",
                    default=os.environ.get("RMHD_ML_CKPTS", DEFAULT_EXTRA))
    ap.add_argument("--summarise", metavar="DIR")
    a = ap.parse_args()
    if a.summarise:
        summarise(a.summarise)
        return
    if not a.brute or not a.out:
        ap.error("need the brute-force dir and --out (or --summarise DIR)")
    if list(a.eps) != sorted(a.eps, reverse=True):
        ap.error("--eps must decrease")
    run_shard(a)


if __name__ == "__main__":
    main()
