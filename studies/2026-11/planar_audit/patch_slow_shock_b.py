"""
Batched slow-shock solve: the Bt-method joint 2-D root-find.

Port of ``slow_shock.solve_slow_shock``.  Given the tangential field
``(By_t, Bz_t)`` behind the wave, the slow shock is fixed by two equations --
the B-Hugoniot ``f_hug(Vs, vxb)`` and Giacomazzo & Rezzolla equation (4.16)
for the normal velocity -- which the scalar code solves JOINTLY as a 2-D
system rather than in the nested scan-then-scan form it originally had.  The
nesting silently locked onto unphysical branches whenever either equation had
several roots, which is common.

The three-part structure is therefore preserved exactly, because each part
exists to stop a specific wrong answer:

* a **coarse grid** over the ``(Vs, vxb)`` window supplies seeds.  It is not
  an optimisation -- it is what makes the search global, and the scalar
  code's own note is explicit that "correctness requires the full grid +
  ranking on every call";
* a **damped 2-D Newton** runs from the best cells;
* every converged root is ranked by the **full Rankine-Hugoniot residual**.
  Both Bt-method equations are satisfied by all of the roots; only the
  physical one conserves mass, momentum and energy.  Ranking on the two
  equations that were solved would be no ranking at all.

What batching changes
---------------------
The grid becomes a ``(NV*NX, N)`` array and "the six best cells" an
``argsort``.  The six Newton solves are then not run one after another: the
seeds are stacked INTO the batch dimension, so one ``6N``-lane Newton
replaces six ``N``-lane ones, and the ``argmin`` over the RH residual happens
after a single reshape.

The scalar escalation ladder -- 12x16, then 24x32, then a fine grid around
``vxa`` for the highly magnetised states whose window is only ~0.01 wide --
is kept, but each rung runs on the COMPACTED subset of lanes that the
previous rung failed.  Compaction is safe here for the reason it is safe
everywhere in this package: a lane's result depends only on that lane's data,
so which other lanes travel with it cannot change its answer.
"""

from __future__ import annotations

import os as _os

import numpy as np

from . import srcport
from .jump import rh_resid

MARGIN = 0.03          # window overshoot past the Alfven bound, as in scalar
N_SEEDS = 6

# Lanes the ladder FAILS are retried on a window scaled to the state ahead
# (``retry_window``).  ON by default since 2026-10-30 (RMHD_SLOWSHOCK_RETRY=0
# turns it off).
# Where the slow and Alfven speeds ahead nearly coincide -- a small B_n, or
# fast tangential flow -- a slow shock lives in a sliver next to the Alfven
# speed far narrower than the ladder's cells: on the 2D runs' lost faces the
# sliver is ~1e-4 wide against cells of ~3e-3 in Vs and 0.13 in vxb, and the
# ladder fails on the very inputs of the shock (failure ledger, 2026-10-29).
# The window is the 2026-09-02 adaptive one, which then REPLACED the ladder's
# and moved root selection on the outer Newton's off-family iterates; as a
# retry it touches failed lanes only, so every lane the ladder answers keeps
# its bits.  A retried root is kept if its full RH residual is <= RETRY_RH.
# One rung: on the ledgers of the three 2D runs the 12 x 16 grid made 130,740
# of the 130,749 rescues, 24 x 32 none and 48 x 64 nine, while the two finer
# grids ran on every lane the first one could not rescue -- most of them
# mid-Newton iterates with no shock to find (calea jobs 60931, 60932).
# Full 64^2 runs off -> on (calea 60938-60940): faces exact of attempted, rotor
# 98.54 -> 98.69%, Orszag-Tang 99.11 -> 99.24%, four-quadrant 96.00 -> 99.26%;
# L1 against 512^2 unchanged (on vs off <= 0.09% of it); wall +6 / +4 / +21%.
SLOW_RETRY = _os.environ.get("RMHD_SLOWSHOCK_RETRY", "1") == "1"
RETRY_RH = 1e-8
RETRY_RUNGS = ((12, 16),)

# Admissible roots only (opt-in, RMHD_SLOWSHOCK_ADMISSIBLE=1; numpy ladder).
# The RH ranking below cannot tell a slow shock from an EXPANSION shock: both
# satisfy the jump conditions to round-off.  MEASURED 2026-11-03 on the
# four-quadrant 64^2 run's harvest: 0.40% of its exact faces hold a slow
# "shock" across which p/rho^gamma falls -- typically one riding on a strong
# fast shock at its speed, beyond the Alfven speed ahead (inside MARGIN),
# undoing most of its compression.  Under the switch a candidate competes only
# if the entropy does not fall (ADM_ENTROPY_TOL, relative) and it meets the
# Lax conditions of the slow family to ADM_LAX_TOL: lam_s(ahead) > Vs >
# lam_s(behind), beyond the Alfven speed ahead and short of the contact.
SLOW_ADMISSIBLE = _os.environ.get("RMHD_SLOWSHOCK_ADMISSIBLE", "0") == "1"
ADM_ENTROPY_TOL = 1e-6
ADM_LAX_TOL = 1e-4


def _load():
    from rmhd import postshock as PS
    from rmhd import slow_shock as SL
    return PS, SL


def _patch_line(old, new):
    """Line-precise source patch that fails loudly if the line moved.

    Same discipline as ``rarefaction_b._patch_source``: a silent no-op here
    would leave a scalar-only construct in the recompiled function and the
    error would surface far away, as a wrong number.
    """
    def patch(src):
        lines = src.split("\n")
        n = 0
        for i, ln in enumerate(lines):
            if ln.strip() == old:
                lines[i] = ln.replace(old, new)
                n += 1
        if n != 1:
            raise RuntimeError(
                f"expected exactly one occurrence of {old!r}, found {n}; "
                f"the scalar source has changed and the port must be rechecked")
        return "\n".join(lines)
    return patch


def make_postshock_bt(gamma):
    """Batched ``postshock.postshock_bt``.

    Returns ``behind(Vs, Byb, Bzb, vxb, ahead, Bx) -> (state7, err)``.  All
    five scalar early-return guards are neutralised in the recompiled source
    and recomputed here as masks -- including ``Vb``, which is internal to the
    scalar function and so has to be re-derived.  It is only four lines, and
    the alternative (inferring it from ``rhob <= 0``) would miss the
    ``|Vb| < 1e-30`` corner, where the density comes back huge but finite.
    """
    PS, _ = _load()
    raw = srcport.compile_batched(
        PS.postshock_bt,
        guards=("v2a >= 1.0", "abs(Ws) < 1e-30", "Bx == 0.0",
                "v2b >= 1.0 or abs(Vs) >= 1.0", "abs(Vb) < 1e-30 or Vb <= 0"),
        min_ifexp=2,
    )

    def behind(Vs, Byb, Bzb, vxb, ahead, Bx, g=gamma):
        rhoa, pa, vxa, vya, vza, Bya, Bza = ahead
        out, _ = raw(Vs, Byb, Bzb, vxb, list(ahead), Bx, g)
        out = list(np.broadcast_arrays(*out))

        v2a = vxa * vxa + vya * vya + vza * vza
        err = (v2a >= 1.0) | (Bx == 0.0) | (np.abs(Vs) >= 1.0)
        Wa = 1.0 / np.sqrt(np.maximum(1.0 - v2a, 1e-300))
        Va = 1.0 / (rhoa * Wa)
        bigF = rhoa * Wa * (Vs - vxa)              # == j / Ws
        Vb = np.where(np.abs(bigF) > 1e-30, Va + (vxa - vxb) / bigF, Va)
        err = err | (np.abs(Vb) < 1e-30) | (Vb <= 0.0)

        v2b = out[2] ** 2 + out[3] ** 2 + out[4] ** 2
        err = err | (v2b >= 1.0)
        for c in out:
            err = err | ~np.isfinite(c)
        return out, err

    return behind


def make_eq416(gamma):
    """Batched ``slow_shock._eq416`` (the y-momentum jump condition).

    Its trailing ``(f, False) if isfinite(f) else (nan, True)`` is a
    conditional over TUPLES, which the ``IfExp -> np.where`` pass would turn
    into a ragged array; it is patched out and the finiteness test is done by
    the caller as a mask instead.
    """
    _, SL = _load()
    raw = srcport.compile_batched(
        SL._eq416,
        guards=("v2a >= 1.0", "v2b >= 1.0", "abs(J) < 1e-30"),
        patch=_patch_line(
            "return (f, False) if math.isfinite(f) else (float('nan'), True)",
            "return f, False"),
    )

    def eq416(Vs, behind, ahead, Bx, g=gamma):
        rho, P, vx, vy, vz, By, Bz = ahead
        f, _ = raw(Vs, list(behind), list(ahead), Bx, g)
        v2a = vx * vx + vy * vy + vz * vz
        v2b = behind[2] ** 2 + behind[3] ** 2 + behind[4] ** 2
        Wa = 1.0 / np.sqrt(np.maximum(1.0 - v2a, 1e-300))
        Ws = 1.0 / np.sqrt(np.maximum(1.0 - Vs * Vs, 1e-30))
        J = Ws * rho * Wa * (Vs - vx)
        bad = (v2a >= 1.0) | (v2b >= 1.0) | (np.abs(J) < 1e-30) | ~np.isfinite(f)
        return f, bad

    return eq416


def make_eq417(gamma):
    """Batched ``slow_shock._eq417`` (the z-momentum jump condition).

    This is the ONE jump condition the Bt-method sub-solver leaves
    unenforced, so at a physical slow shock it vanishes and off the shock
    family it measures the conservation slack.  The 6-unknown outer Newton
    uses it as that wave's residual, which is why it is here and not just in
    the validation code.
    """
    _, SL = _load()
    raw = srcport.compile_batched(
        SL._eq417,
        guards=("v2a >= 1.0", "v2b >= 1.0", "abs(J) < 1e-30"),
        patch=_patch_line(
            "return (f, False) if math.isfinite(f) else (float('nan'), True)",
            "return f, False"),
    )

    def eq417(Vs, behind, ahead, Bx, g=gamma):
        rho, P, vx, vy, vz, By, Bz = ahead
        f, _ = raw(Vs, list(behind), list(ahead), Bx, g)
        v2a = vx * vx + vy * vy + vz * vz
        v2b = behind[2] ** 2 + behind[3] ** 2 + behind[4] ** 2
        Wa = 1.0 / np.sqrt(np.maximum(1.0 - v2a, 1e-300))
        Ws = 1.0 / np.sqrt(np.maximum(1.0 - Vs * Vs, 1e-30))
        J = Ws * rho * Wa * (Vs - vx)
        bad = (v2a >= 1.0) | (v2b >= 1.0) | (np.abs(J) < 1e-30) | ~np.isfinite(f)
        return f, bad

    return eq417


def make_hug(gamma):
    """Batched ``slow_shock._hug_residual`` (the B-Hugoniot on a known state)."""
    _, SL = _load()
    raw = srcport.compile_batched(
        SL._hug_residual,
        guards=("v2b >= 1.0 or pb <= 0 or rhob <= 0",),
        min_ifexp=1,
    )

    def hug(Vs, behind, ahead, Bx, g=gamma):
        rhob, pb = behind[0], behind[1]
        v2b = behind[2] ** 2 + behind[3] ** 2 + behind[4] ** 2
        f = raw(Vs, list(behind), list(ahead), Bx, g)
        bad = (v2b >= 1.0) | (pb <= 0.0) | (rhob <= 0.0) | ~np.isfinite(f)
        return f, bad

    return hug


def alfven_bounds(ahead, Bx, gamma):
    """Batched ``slow_shock._alfven_bounds``."""
    rho, P, vx, vy, vz, By, Bz = ahead
    v2 = vx * vx + vy * vy + vz * vz
    B2 = Bx * Bx + By * By + Bz * Bz
    eta = Bx * vx + By * vy + Bz * vz
    b2 = B2 * (1.0 - v2) + eta * eta
    h = 1.0 + gamma * (P - 0.5 * b2) / ((gamma - 1.0) * rho)
    sw = np.sqrt(np.maximum(rho * h + b2, 1e-30))
    dm, dp = eta - sw, eta + sw
    la = np.where(np.abs(dm) > 1e-30,
                  vx + Bx * (1.0 - v2) / np.where(np.abs(dm) > 1e-30, dm, 1.0), vx)
    ra = np.where(np.abs(dp) > 1e-30,
                  vx + Bx * (1.0 - v2) / np.where(np.abs(dp) > 1e-30, dp, 1.0), vx)
    # the pair is labelled by the sign in front of sqrt(rho h + b^2), so for
    # B_n < 0 the formula's `la` is the RIGHT-going speed: order it
    return np.where(la <= ra, la, ra), np.where(la <= ra, ra, la)


def _take(prob, idx):
    """Compact a problem dict onto a subset of lanes."""
    out = {}
    for k, v in prob.items():
        if k == "ahead":
            out[k] = [c[idx] for c in v]
        elif isinstance(v, np.ndarray):
            out[k] = v[idx]
        else:
            out[k] = v
    return out


def _tile(prob, n):
    """Repeat every lane ``n`` times, matching ``ravel`` of an ``(n, N)`` array."""
    out = {}
    for k, v in prob.items():
        if k == "ahead":
            out[k] = [np.tile(c, n) for c in v]
        elif isinstance(v, np.ndarray):
            out[k] = np.tile(v, n)
        else:
            out[k] = v
    return out


def make_solver(gamma):
    """``solve(By_t, Bz_t, ahead, switch, Bx, ...) -> (Vs, behind, err)``."""
    postshock = make_postshock_bt(gamma)
    eq416 = make_eq416(gamma)
    hug = make_hug(gamma)

    def admissible(ahead, behind, Vs, Bx, switch):
        """Per candidate: the entropy does not fall and the slow family's Lax
        conditions hold (see SLOW_ADMISSIBLE).  A root the quartic cannot
        resolve is judged on the entropy alone, as the fast shock's Lax filter
        leaves untrusted roots unjudged."""
        from . import wave_speeds_b as WB

        def entropy(s):
            rho, P, vx, vy, vz, By, Bz = s
            v2 = vx * vx + vy * vy + vz * vz
            eta = Bx * vx + By * vy + Bz * vz
            b2 = (Bx * Bx + By * By + Bz * Bz) * (1.0 - v2) + eta * eta
            return (P - 0.5 * b2) / rho ** gamma

        with np.errstate(all="ignore"):
            ok = entropy(behind) >= entropy(ahead) * (1.0 - ADM_ENTROPY_TOL)
            ea, la, ra, oka = WB.xi_all(*ahead, Bx, gamma)
            eb, _, _, okb = WB.xi_all(*behind, Bx, gamma)
            if switch == "LS":
                m = np.minimum.reduce([ea[:, 1] - Vs, Vs - eb[:, 1], Vs - la,
                                       behind[2] - Vs])
                trusted = oka[:, 1] & okb[:, 1]
            else:
                m = np.minimum.reduce([eb[:, 2] - Vs, Vs - ea[:, 2], Vs - behind[2],
                                       ra - Vs])
                trusted = oka[:, 2] & okb[:, 2]
            ok = ok & ((m >= -ADM_LAX_TOL) | ~trusted)
        return ok

    def joint(Vs, vxb, prob):
        """The joint residual ``(f_hug, f_416)``, the behind state, and validity.

        ``valid`` folds in the window test and the slow-wave causality test
        that the scalar ``_F`` expresses by returning ``None``.
        """
        ahead, Bx = prob["ahead"], prob["Bx"]
        ok = ((Vs >= prob["Vs_lo"]) & (Vs <= prob["Vs_hi"])
              & (vxb >= prob["v_lo"]) & (vxb <= prob["v_hi"]))
        vxa = ahead[2]
        if prob["switch"] == "LS":
            ok = ok & (Vs < np.minimum(vxa, vxb) - 1e-12)
        else:
            ok = ok & (Vs > np.maximum(vxa, vxb) + 1e-12)
        bh, berr = postshock(Vs, prob["By_t"], prob["Bz_t"], vxb, ahead, Bx)
        f1, bad1 = hug(Vs, bh, ahead, Bx)
        f2, bad2 = eq416(Vs, bh, ahead, Bx)
        ok = ok & ~berr & ~bad1 & ~bad2
        return f1, f2, bh, ok

    def grid(prob, NV, NX, vlo, vhi):
        """Coarse ``(NV*NX, N)`` grid; returns the cell coordinates and scores."""
        Vs_lo, Vs_hi = prob["Vs_lo"], prob["Vs_hi"]
        N = Vs_lo.shape[0]
        ci = (np.arange(NV) + 0.5) / NV
        cj = (np.arange(NX) + 0.5) / NX
        Vg = Vs_lo[None, :] + ci[:, None] * (Vs_hi - Vs_lo)[None, :]
        Xg = vlo[None, :] + cj[:, None] * (vhi - vlo)[None, :]
        Vc = np.repeat(Vg, NX, axis=0)
        Xc = np.tile(Xg, (NV, 1))
        # One joint() call on the (K*N,) grid with prob tiled K-fold instead
        # of K = NV*NX calls on (N,): identical per-lane arithmetic, so
        # bit-identical, and K-1 fewer passes of dispatch overhead.  Vc/Xc
        # are (K, N); .reshape(-1) is k-major, matching np.tile(a, K).
        K = NV * NX
        pk = {key: ([np.tile(c, K) for c in val] if key == "ahead"
                    else np.tile(val, K) if isinstance(val, np.ndarray)
                    else val)
              for key, val in prob.items()}
        f1, f2, _, ok = joint(Vc.reshape(-1), Xc.reshape(-1), pk)
        scores = np.where(ok, np.abs(f1) + np.abs(f2), np.inf).reshape(K, N)
        return Vc, Xc, scores

    def newton2d(Vs0, vxb0, prob, accuracy, max_iter=40, n_backtrack=8):
        """Damped 2-D Newton, fixed trip count, with finished lanes retired.

        Most of the ``6N`` stacked seeds die in the first iteration or two --
        a seed whose cell is nowhere near the shock family leaves the valid
        window on its first step.  Carrying them for the remaining 38
        iterations is most of the cost, so at the end of each iteration the
        stopped lanes are flushed to the output and the working arrays are
        COMPACTED onto the survivors.

        This is the same move the outer escalation ladder makes between rungs,
        applied inside the iteration.  It is safe for the same reason: the
        arithmetic on a surviving lane is bit-for-bit what it would have been
        with the retired lanes still present, because nothing here couples one
        lane to another.
        """
        M = Vs0.shape[0]
        Vs_out, X_out = Vs0.copy(), vxb0.copy()
        conv_out = np.zeros(M, dtype=bool)
        idx = np.arange(M)

        Vs, vxb, P = Vs0.copy(), vxb0.copy(), prob
        tol = accuracy * P["f_scale"]
        f1, f2, _, ok = joint(Vs, vxb, P)
        dead = ~ok
        stopped = dead | ((np.abs(f1) < tol) & (np.abs(f2) < tol) & ok)

        def flush_and_compact():
            nonlocal idx, Vs, vxb, f1, f2, dead, stopped, P, tol
            s_ = stopped
            if s_.any():
                g = idx[s_]
                Vs_out[g] = Vs[s_]
                X_out[g] = vxb[s_]
                conv_out[g] = ((np.abs(f1) < tol) & (np.abs(f2) < tol)
                               & ~dead)[s_]
            keep = ~s_
            if keep.all():
                return
            idx, Vs, vxb = idx[keep], Vs[keep], vxb[keep]
            f1, f2, dead = f1[keep], f2[keep], dead[keep]
            tol = tol[keep]
            P = _take(P, keep)
            stopped = np.zeros(idx.size, dtype=bool)

        flush_and_compact()

        for _ in range(max_iter):
            if idx.size == 0:
                break
            hV = 1e-7 * np.maximum(np.abs(Vs), 1e-3)
            hx = 1e-7 * np.maximum(np.abs(vxb), 1e-3)
            # forward difference, falling back to backward where the forward
            # probe leaves the valid window -- the scalar's `if rV is None`
            aV1, aV2, _, okV = joint(Vs + hV, vxb, P)
            bV1, bV2, _, okVb = joint(Vs - hV, vxb, P)
            hVs = np.where(okV, hV, -hV)
            rV1 = np.where(okV, aV1, bV1)
            rV2 = np.where(okV, aV2, bV2)

            ax1, ax2, _, okx = joint(Vs, vxb + hx, P)
            bx1, bx2, _, okxb = joint(Vs, vxb - hx, P)
            hxs = np.where(okx, hx, -hx)
            rx1 = np.where(okx, ax1, bx1)
            rx2 = np.where(okx, ax2, bx2)

            jac_bad = (~okV & ~okVb) | (~okx & ~okxb)

            a11 = (rV1 - f1) / hVs
            a12 = (rx1 - f1) / hxs
            a21 = (rV2 - f2) / hVs
            a22 = (rx2 - f2) / hxs
            det = a11 * a22 - a12 * a21
            sing = jac_bad | (np.abs(det) < 1e-30) | ~np.isfinite(det)
            dets = np.where(sing, 1.0, det)
            dVs = -(a22 * f1 - a12 * f2) / dets
            dvx = -(-a21 * f1 + a11 * f2) / dets
            sing = sing | ~np.isfinite(dVs) | ~np.isfinite(dvx)

            dead = dead | sing
            stopped = stopped | sing

            base = np.abs(f1) + np.abs(f2)
            lam = np.ones(idx.size)
            stepped = np.zeros(idx.size, dtype=bool)
            for _bt in range(n_backtrack):
                trial = ~stopped & ~stepped
                if not bool(trial.any()):
                    break
                Vn = Vs + lam * dVs
                Xn = vxb + lam * dvx
                g1, g2, _, okn = joint(Vn, Xn, P)
                good = trial & okn & ((np.abs(g1) + np.abs(g2)) < base)
                if bool(good.any()):
                    Vs = np.where(good, Vn, Vs)
                    vxb = np.where(good, Xn, vxb)
                    f1 = np.where(good, g1, f1)
                    f2 = np.where(good, g2, f2)
                    stepped = stepped | good
                lam = np.where(stepped, lam, lam * 0.5)

            # no admissible step: the scalar gives up on this seed entirely
            failed = ~stopped & ~stepped
            dead = dead | failed
            stopped = stopped | failed | ((np.abs(f1) < tol) & (np.abs(f2) < tol))
            flush_and_compact()

        if idx.size:
            conv_out[idx] = ((np.abs(f1) < tol) & (np.abs(f2) < tol) & ~dead)
            Vs_out[idx], X_out[idx] = Vs, vxb
        return Vs_out, X_out, conv_out

    def attempt(prob, NV, NX, vlo, vhi, accuracy):
        """One rung of the ladder: grid, N_SEEDS Newtons, RH ranking.

        The seeds go into the BATCH dimension, so the six Newton solves the
        scalar runs sequentially are a single solve here.
        """
        ahead = prob["ahead"]
        N = ahead[0].shape[0]
        Vc, Xc, scores = grid(prob, NV, NX, vlo, vhi)
        order = np.argsort(scores, axis=0)[:N_SEEDS]            # (S, N)
        cols = np.arange(N)[None, :]
        Vs0 = Vc[order, cols]
        Xs0 = Xc[order, cols]
        seed_ok = np.isfinite(scores[order, cols])

        big = _tile(prob, N_SEEDS)
        Vs, vxb, conv = newton2d(Vs0.ravel(), Xs0.ravel(), big, accuracy)
        conv = conv & seed_ok.ravel()

        bh, berr = postshock(Vs, big["By_t"], big["Bz_t"], vxb,
                             big["ahead"], big["Bx"])
        rh = rh_resid(big["ahead"], bh, Vs, big["Bx"], gamma)
        rh = np.where(conv & ~berr, rh, np.inf)
        if SLOW_ADMISSIBLE:
            rh = np.where(admissible(big["ahead"], bh, Vs, big["Bx"],
                                     prob["switch"]), rh, np.inf)

        rh = rh.reshape(N_SEEDS, N)
        best = np.argmin(rh, axis=0)
        good = np.isfinite(rh[best, cols[0]])
        pick = best * N + np.arange(N)
        Vs_b = Vs[pick]
        bh_b = [c[pick] for c in bh]

        # the best grid cell is handed back UNUSED.  It is the loose fallback,
        # and the scalar code only reaches that after every rung has failed --
        # applying it per rung would let a lane settle for a coarse grid point
        # instead of escalating, which is how it lands on a spurious root.
        kbest = np.argmin(np.where(np.isfinite(scores), scores, np.inf), axis=0)
        cell = (Vc[kbest, cols[0]], Xc[kbest, cols[0]], scores[kbest, cols[0]])
        return Vs_b, bh_b, good, cell

    def ladder(By_t, Bz_t, ahead, switch, Bx, accuracy=1e-8,
               vxb_bracket=None, vxb_hint=None):
        # RMHD_SLOWSHOCK=njit: the compiled per-lane transcription in
        # slow_shock_njit -- same contract, same return.  Measured bitwise-
        # identical on 500 problems incl. 93 failures (test_slow_shock_njit).
        from . import slow_shock_njit as _SN
        if _SN.wanted() and not SLOW_ADMISSIBLE:          # the filter is numpy-only
            return _SN.make_solver(gamma)(By_t, Bz_t, ahead, switch, Bx,
                                          accuracy=accuracy,
                                          vxb_bracket=vxb_bracket,
                                          vxb_hint=vxb_hint)
        ahead = [np.asarray(c, dtype=float) for c in ahead]
        N = ahead[0].shape[0]
        bc = lambda v: np.broadcast_to(np.asarray(v, dtype=float),
                                       (N,)).astype(float)
        By_t, Bz_t, Bx = bc(By_t), bc(Bz_t), bc(Bx)
        vxa, p_a = ahead[2], ahead[1]
        la, ra = alfven_bounds(ahead, Bx, gamma)

        if switch == "LS":
            Vs_lo = np.maximum(la - MARGIN, -0.9999)
            Vs_hi = vxa - 1e-9
        elif switch == "RS":
            Vs_lo = vxa + 1e-9
            Vs_hi = np.minimum(ra + MARGIN, 0.9999)
        else:
            raise ValueError(switch)

        if vxb_bracket is not None:
            v_lo = np.maximum(-0.9999, bc(vxb_bracket[0]))
            v_hi = np.minimum(0.9999, bc(vxb_bracket[1]))
        elif vxb_hint is not None:
            hint = bc(vxb_hint)
            half = np.maximum(0.12, np.abs(hint - vxa) * 2.0)
            v_lo = np.maximum(-0.9999, hint - half)
            v_hi = np.minimum(0.9999, hint + half)
        else:
            v_lo = np.full(N, -0.9999)
            v_hi = np.full(N, 0.9999)

        prob = {"ahead": ahead, "By_t": By_t, "Bz_t": Bz_t, "Bx": Bx,
                "Vs_lo": Vs_lo, "Vs_hi": Vs_hi, "v_lo": v_lo, "v_hi": v_hi,
                "f_scale": 1.0 + np.abs(p_a), "switch": switch}

        Vs_out = vxa.copy()
        bh_out = [c.copy() for c in ahead]
        done = np.zeros(N, dtype=bool)
        # empty windows are unsolvable, exactly as in the scalar guard
        dead = (Vs_hi <= Vs_lo) | (v_hi <= v_lo)

        # the loose-fallback cell, carried across the ladder.  The scalar
        # REASSIGNS its cell list at the second rung and keeps the fine grid's
        # cells only when the coarse grids produced none at all, so the same
        # bookkeeping is done here.
        loose_V = vxa.copy()
        loose_X = vxa.copy()
        loose_s = np.full(N, np.inf)

        # the escalation ladder, each rung run on the COMPACTED set of lanes
        # the previous rung failed
        rungs = [(12, 16, None), (24, 32, None)]
        if vxb_bracket is None:
            rungs.append((24, 40, 0.20))

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            for rung, (NV, NX, halfwidth) in enumerate(rungs):
                idx = np.flatnonzero(~done & ~dead)
                if idx.size == 0:
                    break
                sub = _take(prob, idx)
                if halfwidth is None:
                    vlo, vhi = sub["v_lo"], sub["v_hi"]
                else:
                    # highly magnetised states are finite only in a narrow
                    # vxb window around vxa that the coarse grid steps over
                    vlo = np.maximum(sub["v_lo"], sub["ahead"][2] - halfwidth)
                    vhi = np.minimum(sub["v_hi"], sub["ahead"][2] + halfwidth)
                usable = vhi > vlo
                Vs_s, bh_s, ok_s, cell = attempt(sub, NV, NX, vlo, vhi,
                                                 accuracy)
                ok_s = ok_s & usable
                sel = idx[ok_s]
                Vs_out[sel] = Vs_s[ok_s]
                for j in range(7):
                    bh_out[j][sel] = bh_s[j][ok_s]
                done[sel] = True

                cV, cX, cs = cell
                keep = usable if rung < 2 else (usable & ~np.isfinite(loose_s[idx]))
                loose_V[idx] = np.where(keep, cV, loose_V[idx])
                loose_X[idx] = np.where(keep, cX, loose_X[idx])
                loose_s[idx] = np.where(keep, cs, loose_s[idx])

            # only now, with the ladder exhausted: hand the outer Newton the
            # best grid point if it is close enough to be worth refining
            idx = np.flatnonzero(~done & ~dead & (loose_s < 0.1 * prob["f_scale"]))
            if idx.size:
                sub = _take(prob, idx)
                bl, ble = postshock(loose_V[idx], sub["By_t"], sub["Bz_t"],
                                    loose_X[idx], sub["ahead"], sub["Bx"])
                sel = idx[~ble]
                Vs_out[sel] = loose_V[sel]
                for j in range(7):
                    bh_out[j][sel] = bl[j][~ble]
                done[sel] = True

        return Vs_out, bh_out, ~done

    def retry(Vs, bh, err, By_t, Bz_t, ahead, switch, Bx, accuracy, vxb_bracket):
        """The lanes the ladder failed, once more on ``retry_window``.

        Each rung is the ladder's own ``attempt`` (grid, N_SEEDS Newtons, RH
        ranking) on the compacted lanes still failing; a root counts only if
        its full RH residual is at most RETRY_RH.  Lanes the ladder answered
        are not touched.
        """
        err = np.asarray(err, dtype=bool)
        if not err.any():
            return Vs, bh, err
        ahead = [np.asarray(c, dtype=float) for c in ahead]
        N = ahead[0].shape[0]
        bc = lambda v: np.broadcast_to(np.asarray(v, dtype=float),
                                       (N,)).astype(float)
        By_t, Bz_t, Bx = bc(By_t), bc(Bz_t), bc(Bx)
        i = np.flatnonzero(err)
        ah = [c[i] for c in ahead]
        Vs_lo, Vs_hi, v_lo, v_hi = retry_window(ah, Bx[i], gamma, switch)
        if vxb_bracket is not None:
            v_lo = np.maximum(v_lo, bc(vxb_bracket[0])[i])
            v_hi = np.minimum(v_hi, bc(vxb_bracket[1])[i])
        use = (Vs_hi > Vs_lo) & (v_hi > v_lo)
        i = i[use]
        if i.size == 0:
            return Vs, bh, err
        prob = {"ahead": [c[use] for c in ah], "By_t": By_t[i],
                "Bz_t": Bz_t[i], "Bx": Bx[i],
                "Vs_lo": Vs_lo[use], "Vs_hi": Vs_hi[use],
                "v_lo": v_lo[use], "v_hi": v_hi[use],
                "f_scale": 1.0 + np.abs(ahead[1][i]), "switch": switch}
        Vs = np.array(Vs, dtype=float, copy=True)
        bh = [np.array(c, dtype=float, copy=True) for c in bh]
        err = err.copy()
        todo = np.ones(i.size, dtype=bool)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            for NV, NX in RETRY_RUNGS:
                k = np.flatnonzero(todo)
                if k.size == 0:
                    break
                sub = _take(prob, k)
                V2, b2, ok, _ = attempt(sub, NV, NX, sub["v_lo"], sub["v_hi"],
                                        accuracy)
                rh = rh_resid(sub["ahead"], b2, V2, sub["Bx"], gamma)
                ok = ok & (rh <= RETRY_RH)
                for c in b2:
                    ok = ok & np.isfinite(c)
                kk = k[ok]
                Vs[i[kk]] = V2[ok]
                for j in range(7):
                    bh[j][i[kk]] = b2[j][ok]
                err[i[kk]] = False
                todo[kk] = False
        return Vs, bh, err

    def solve(By_t, Bz_t, ahead, switch, Bx, accuracy=1e-8,
              vxb_bracket=None, vxb_hint=None):
        out = ladder(By_t, Bz_t, ahead, switch, Bx, accuracy=accuracy,
                     vxb_bracket=vxb_bracket, vxb_hint=vxb_hint)
        if not SLOW_RETRY:
            return out
        return retry(*out, By_t, Bz_t, ahead, switch, Bx, accuracy,
                     vxb_bracket)

    return solve


def retry_window(ahead, Bx, gamma, switch):
    """The retry's ``(Vs_lo, Vs_hi, v_lo, v_hi)``, scaled to the state ahead.

    ``g`` is the gap between the fluid's normal velocity and the Alfven speed
    ahead (``vxa - la`` for a left-going wave), the room a slow shock has:
    Vs runs from the Alfven bound, less min(MARGIN, g), to vxa, and vxb over
    vxa +- 2 g -- where the ladder spans (-1, 1).  A lane with no room
    (g <= 0, or not finite) gets an empty window.
    """
    vxa = ahead[2]
    la, ra = alfven_bounds(ahead, Bx, gamma)
    if switch == "LS":
        g = vxa - la
        Vs_lo = np.maximum(la - np.minimum(MARGIN, np.maximum(g, 0.0)), -0.9999)
        Vs_hi = vxa - 1e-9
    elif switch == "RS":
        g = ra - vxa
        Vs_lo = vxa + 1e-9
        Vs_hi = np.minimum(ra + np.minimum(MARGIN, np.maximum(g, 0.0)), 0.9999)
    else:
        raise ValueError(switch)
    v_lo = np.maximum(vxa - 2.0 * np.abs(g), -0.9999)
    v_hi = np.minimum(vxa + 2.0 * np.abs(g), 0.9999)
    v_hi = np.where(g > 0, v_hi, v_lo)
    return Vs_lo, Vs_hi, v_lo, v_hi
