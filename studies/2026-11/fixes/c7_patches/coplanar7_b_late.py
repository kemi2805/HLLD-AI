"""
The seven-wave family on COPLANAR interfaces, written in the four-unknown
planar form.  ``RMHD_COPLANAR7=1`` makes it the first solve `api.solve_batch`
tries on the ``COPLANAR`` class; lanes it cannot answer fall back to
``fullcontact6`` as before.  ``=after`` runs it after ``fullcontact6`` on the
lanes that one lost or answered with an entropy-lowering jump (`after`);
``=late`` keeps it out of `api.solve_batch` altogether and lets the exact
flux (HLLD) offer it as a rescue rung after the five-wave planar solver
(`late`).  Off by default.

Why a second formulation of the same seven waves
------------------------------------------------
``fullcontact6`` solves for one tangential field at the contact (magnitude and
angle) that BOTH slow waves are built towards, plus the two rotation angles.
On a coplanar interface -- every interface of a 2D run -- that cannot express
three things the 4Q 128^2 failure ledger needs (2026-11-04/05):

* a slow wave steered by anything but the field: at beta ~100 a total-pressure
  change of 1e-5 swings B_t by 0.1, and with the field nearly normal B_t is the
  wrong coordinate on the other faces -- each side needs its own steering;
* a slow fan that carries the lab-frame field THROUGH ZERO (a |B_t|-steered
  wave cannot), which the near-normal faces need on one or both sides;
* the coplanar rotation that turns the field by pi in the fluid frame but
  KEEPS its lab-frame sign: the rotation is pinned to a lab angle, and at the
  same angle that is the identity (fullcontact_b.coplanar_rotation).

Without them the solver either fails or fuses the rotation into a slow jump
that crosses the Alfven speed and lowers the entropy by 1e-9..7e-5, which the
admissibility gate refuses.  The four-unknown form (planar4_b) has all three:
the unknowns are the two fast-wave pressures and one parameter per slow wave
(here the signed field behind it), the contact's field continuity is an
equation, and a rotation in front of either slow wave is a discrete branch
with no unknown.  Re-solved that way from their own refused answers, 112 of
112 refused ledger faces became admissible regular answers whose flux moved by
a median 9e-9 (fixes/rd_solve.py, calea 79423).

The stages
----------
Every stage is ONE batched planar4 solve over (lane, branch) pairs, central
differences, ``MAX_ITER`` iterations; a lane is answered by the first branch
whose answer converges, passes planar4's certificate (jump conditions, fan
order) and raises the entropy at every shock (``ENTROPY_TOL``):

  1. the ML seed, with the rotations it suggests (|phi| > pi/2 on a side);
  2. the ML seed, the other three rotation choices (fewest rotations first);
  3. the ML seed with PRESSURE-steered slow waves (the total pressures behind
     them read off the signed-field structure at the seed), all four rotation
     choices, the suggested one first -- at moderate beta B_t is the wrong
     coordinate (``RMHD_COPLANAR7_P=0`` leaves this stage out);
  4. each extra checkpoint's seed, all four rotation choices.

The answer is returned in the seven-wave layout ``[R2..R7] = [A, A2, B, C,
D2, D]`` (lab frame), so the ray, the wave-order gate, the admissibility gate
and the harvest read it like any other seven-wave answer; ``unk6`` is its
equivalent ``[ln p_LF, ln|Bt_CD|, psi_CD, ln p_RF, phi_L, phi_R]`` (a rotation
that keeps the lab sign has phi = 0 there -- the zones carry the truth).
"""
from __future__ import annotations

import os as _os

import numpy as np

from . import planar4_b as P4
from .planar5_b import to_planar
from ..util.angles import wrap as _wrap

MODE = _os.environ.get("RMHD_COPLANAR7", "0")
ENTROPY_TOL = float(_os.environ.get("RMHD_COPLANAR7_DS", "1e-9"))
MAX_ITER = int(_os.environ.get("RMHD_COPLANAR7_ITER", "25"))
# the pressure-steered stage (3 below); RMHD_COPLANAR7_P=0 leaves it out
P_STAGE = _os.environ.get("RMHD_COPLANAR7_P", "1") not in ("0", "", "off")
_COMBOS = ((False, False), (True, False), (False, True), (True, True))


def wanted():
    """The stage inside `api.solve_batch` (first or `after`)."""
    return MODE not in ("0", "", "off", "late")


def after():
    """``RMHD_COPLANAR7=after``: the planar form runs AFTER fullcontact6, on
    the COPLANAR lanes it lost or answered with a jump across which the
    entropy falls (`api.solve_batch`), instead of first on every COPLANAR
    lane.  Measured on the 64^2 runs (2026-11-05): first, it cut the wall per
    step against the rotation runs from 4.3x to 1.6x the control on
    Orszag-Tang and from 2.9x to 1.8x on the rotor, but on the four-quadrant
    problem's early steps, where every face is easy, it went from 1.0x to
    4.6x -- it replaces a Newton that converges in a few cheap iterations."""
    return MODE == "after"


def late():
    """``RMHD_COPLANAR7=late``: not a stage of `api.solve_batch` but a rescue
    rung of the exact flux (HLLD ``exact_flux``), offered the COPLANAR lanes
    that neither the six-unknown Newton nor the five-wave planar solver
    answered admissibly.  Measured 2026-11-05: ``after`` cost the 4Q 64^2 run
    3.7x the control because the faces it took -- nearly at rest (median W
    1.00) in the 45-degree field, wave speeds ~0.25 c apart, where the
    production network's seed leaves the Newton at iteration 0 -- are
    answered by the five-wave solver in 4 iterations, and by this one only at stage 3 after ~30 (96 faces: 79 s
    against 19 s, flux the same to 3e-9, no rotation used)."""
    return MODE == "late"


def entropy_min7(left, right, zones, Bn, gamma):
    """The smallest K_behind / K_ahead (K = p / rho^gamma) over the six waves
    of seven-wave answers (zones R2..R7 between ``left`` and ``right``):
    fans and rotations keep K, so a value below 1 is a jump that lowers the
    entropy."""
    def K(X):
        rho, Pt, vx, vy, vz, By, Bz = X
        W2 = 1.0 / np.maximum(1.0 - (vx * vx + vy * vy + vz * vz), 1e-300)
        vB = vx * Bn + vy * By + vz * Bz
        return (Pt - 0.5 * ((Bn ** 2 + By ** 2 + Bz ** 2) / W2 + vB ** 2)) / rho ** gamma

    R2, R3, R4, R5, R6, R7 = zones
    m = np.ones(Bn.shape[0])
    with np.errstate(all="ignore"):
        for ah, bh in ((left, R2), (R2, R3), (R3, R4), (R6, R5), (R7, R6), (right, R7)):
            q = K(bh) / K(ah)
            m = np.minimum(m, np.where(np.isfinite(q), q, 1.0))
    return m


def make_solver(gamma):
    S4 = P4.make_solver(gamma)

    def _K(X, Bn):
        rho, Pt, vx, vy, vz, By, Bz = X
        W2 = 1.0 / np.maximum(1.0 - (vx * vx + vy * vy + vz * vz), 1e-300)
        vB = vx * Bn + vy * By + vz * Bz
        return (Pt - 0.5 * ((Bn ** 2 + By ** 2 + Bz ** 2) / W2 + vB ** 2)) / rho ** gamma

    def entropy_min(r, left, right, Bn):
        """The smallest K_behind / K_ahead over the shocks of a planar4 answer
        (rotation answers: the slow shocks start behind the rotation); 1
        where there is none."""
        A, B_, C_, D = r["zones"]
        A2, D2 = r["zones_rot"] if "zones_rot" in r else (A, D)
        m = np.ones(Bn.shape[0])
        with np.errstate(all="ignore"):
            for row, (ah, bh) in enumerate(((left, A), (A2, B_), (D2, C_), (right, D))):
                q = _K(bh, Bn) / _K(ah, Bn)
                q = np.where(np.isfinite(q), q, 0.0)
                m = np.where(r["shocks"][row], np.minimum(m, q), m)
        return m

    def planar_seed(left, right, Bn, s6):
        """A seven-wave seed as four planar unknowns (signed-field slow
        waves) and the rotations it suggests."""
        alpha = to_planar(left, right, Bn)[2]
        bt = np.exp(s6[1]) * np.cos(s6[2] + alpha)
        u = [np.array(s6[0], dtype=float, copy=True), bt.copy(), bt.copy(),
             np.array(s6[3], dtype=float, copy=True)]
        rot = np.stack([np.abs(_wrap(s6[4])) > 0.5 * np.pi,
                        np.abs(_wrap(s6[5])) > 0.5 * np.pi])
        return u, rot

    def solve(left, right, Bn, s6, seeds_extra=None, accuracy=1e-10,
              max_iter=MAX_ITER, planar_tol=1e-6, verify_tol=1e-8):
        """Returns a dict over the lanes: ``converged``, ``zones`` (six
        states, lab frame), ``VsLv``, ``VsRv``, ``unk6``, ``shocks`` (4
        per-lane masks LF, LS, RS, RF), ``kinds`` (2 per-lane steerings),
        ``rot`` (2 per-lane flags), ``stage`` (1-4, 0 unanswered),
        ``n_iter``."""
        n = Bn.shape[0]
        conv = np.zeros(n, dtype=bool)
        stage = np.zeros(n, dtype=np.int8)
        n_iter = np.zeros(n, dtype=int)
        zones = [[np.zeros(n) for _ in range(7)] for _ in range(6)]
        VsLv = [np.zeros(n) for _ in range(3)]
        VsRv = [np.zeros(n) for _ in range(3)]
        unk6 = [np.zeros(n) for _ in range(6)]
        shocks = [np.zeros(n, dtype=bool) for _ in range(4)]
        kinds = [np.zeros(n, dtype=np.int8) for _ in range(2)]
        rotf = [np.zeros(n, dtype=bool) for _ in range(2)]

        u0, rot0 = planar_seed(left, right, Bn, s6)
        combos = np.array(_COMBOS, dtype=bool)                     # (4, 2)
        # which of the four combinations each lane's seed suggests (index into
        # _COMBOS: left rotation +1, right rotation +2)
        sugg = rot0[0].astype(int) + 2 * rot0[1].astype(int)
        # seeds: (four planar unknowns, slow-wave steering).  0 the ML seed
        # with signed-field slow waves; 1 the same seed with PRESSURE-steered
        # slow waves (the total pressures behind them, read off the signed-
        # field structure at the seed; the faces where a total-pressure
        # change of 1e-5 does not swing B_t -- moderate beta -- need it); then
        # each extra checkpoint's seed, signed field
        seeds = [(u0, P4.KIND_S)]
        if P_STAGE:
            L0, R0 = to_planar(left, right, Bn)[:2]
            with np.errstate(all="ignore"):
                st0 = S4["structure"](L0, R0, u0, Bn, np.full((2, n), P4.KIND_S, np.int8),
                                      rot=combos[sugg].T)
            e0 = st0[3]
            lp_mean = np.log(np.maximum(0.5 * (L0[1] + R0[1]), 1e-300))
            lpB = np.where(e0, lp_mean, np.log(np.maximum(st0[1][1][1], 1e-300)))
            lpC = np.where(e0, lp_mean, np.log(np.maximum(st0[1][2][1], 1e-300)))
            seeds.append(([u0[0].copy(), lpB, lpC, u0[3].copy()], P4.KIND_P))
        n_main = len(seeds)
        seeds += [(planar_seed(left, right, Bn, se)[0], P4.KIND_S)
                  for se in (seeds_extra or [])]

        def branches(st):
            """(lane, seed index, combination index) of a stage, each lane's
            branches together and in preference order."""
            k = np.arange(n)
            if st == 1:
                return k, np.zeros(n, dtype=int), sugg
            if st == 2:
                lane = np.repeat(k, 4)
                c = np.tile(np.arange(4), n)
                keep = c != sugg[lane]
                return lane[keep], np.zeros(keep.sum(), dtype=int), c[keep]
            if st == 3:
                if not P_STAGE:
                    return np.zeros(0, int), np.zeros(0, int), np.zeros(0, int)
                # the pressure-steered seed, the suggested combination first
                lane = np.repeat(k, 4)
                c = (np.tile(np.arange(4), n) + np.repeat(sugg, 4)) % 4
                return lane, np.ones(lane.size, dtype=int), c
            ns = len(seeds) - n_main
            if ns <= 0:
                return np.zeros(0, int), np.zeros(0, int), np.zeros(0, int)
            lane = np.repeat(k, 4 * ns)
            s = np.tile(np.repeat(np.arange(n_main, n_main + ns), 4), n)
            c = np.tile(np.arange(4), n * ns)
            return lane, s, c

        for st in (1, 2, 3, 4):
            lane, sidx, cidx = branches(st)
            keep = ~conv[lane]
            lane, sidx, cidx = lane[keep], sidx[keep], cidx[keep]
            if lane.size == 0:
                continue
            rot = combos[cidx].T                                   # (2, m)
            ui = [np.zeros(lane.size) for _ in range(4)]
            kd = np.zeros((2, lane.size), dtype=np.int8)
            for s in np.unique(sidx):
                m = sidx == s
                for c in range(4):
                    ui[c][m] = seeds[s][0][c][lane[m]]
                kd[:, m] = seeds[s][1]
            L = [c[lane] for c in left]
            R = [c[lane] for c in right]
            B = Bn[lane]
            r = S4["solve"](L, R, B, accuracy=accuracy, max_iter=max_iter,
                            planar_tol=planar_tol, verify_tol=verify_tol, kinds=kd,
                            retry=False, unk_init=ui, central=True, rot=rot)
            np.add.at(n_iter, lane, np.asarray(r["n_iter"]))
            good = np.asarray(r["converged"], bool)
            if good.any():
                good &= entropy_min(r, L, R, B) >= 1.0 - ENTROPY_TOL
            gi = np.flatnonzero(good)
            if gi.size == 0:
                continue
            # per lane the FIRST good branch: branches are in preference order
            _, first = np.unique(lane[gi], return_index=True)
            j = gi[first]
            k = lane[j]
            conv[k] = True
            stage[k] = st
            A, B_, C_, D = r["zones"]
            A2, D2 = r["zones_rot"]
            for z, X in enumerate((A, A2, B_, C_, D2, D)):
                for c in range(7):
                    zones[z][c][k] = X[c][j]
            VsLv[0][k], VsLv[1][k], VsLv[2][k] = r["speeds"][0][j], r["speeds_rot"][0][j], r["speeds"][1][j]
            VsRv[0][k], VsRv[1][k], VsRv[2][k] = r["speeds"][4][j], r["speeds_rot"][1][j], r["speeds"][3][j]
            for q in range(4):
                shocks[q][k] = r["shocks"][q][j]
            for q in range(2):
                kinds[q][k] = r["kinds"][q][j]
                rotf[q][k] = rot[q][j]
            unk6[0][k], unk6[3][k] = r["unk"][0][j], r["unk"][3][j]
        # the seven-wave equivalent of each answer (diagnostics, the harvest)
        if conv.any():
            k = np.flatnonzero(conv)
            R2, R3, R4, R5, R6, R7 = zones
            psi = lambda X: np.arctan2(X[6][k], X[5][k])
            unk6[1][k] = np.log(np.maximum(np.hypot(R4[5][k], R4[6][k]), 1e-300))
            unk6[2][k] = psi(R4)
            unk6[4][k] = _wrap(psi(R3) - psi(R2))
            unk6[5][k] = _wrap(psi(R7) - psi(R6))
        return dict(converged=conv, zones=zones, VsLv=VsLv, VsRv=VsRv, unk6=unk6,
                    shocks=shocks, kinds=kinds, rot=rotf, stage=stage, n_iter=n_iter)

    return dict(solve=solve, planar_seed=planar_seed, entropy_min=entropy_min)
