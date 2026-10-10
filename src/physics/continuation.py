"""The continuation rung: the seven-wave solve walked in from a tilted face.

WHY
---
On the four-quadrant problem with the field tilted 60 degrees out of the
plane (docs/what_we_solve.md section 7) every face is genuinely non-coplanar,
the seven-wave solver works alone and loses 13.8% of the attempted faces, and
nothing else in the chain can take them: the planar rungs refuse a face that
is not planar (since the planarity pre-check, without trying). The failure
ledger on 1,195 of them (calea job 39286) recovered 63% with the eps-ladder
of the coplanar-limit study (scripts/coplanar_limit.py), against a fifth for
seven seeds with eight retries; on 1,205 faces production solves the same
search answered 1,187 and landed on another root once (job 40456). It is a
continuation -- a homotopy in a tilt angle -- not a better seed.

WHAT
----
Per lane, in the planar frame of ``planar5_b.to_planar`` (the left
tangential field along +y: a rotation about the normal, so the same problem):

  rung 0   the tangential fields tilted apart by +-eps about the normal,
           eps = 0.1 (rho, gas pressure and v held; the total pressure moves
           with b^2), solved like production: the ML ensemble's seeds for
           the TILTED states -- computed by the caller in the main process,
           where the networks live -- per-lane retry keys, n_retries retries;
  rungs    eps = 3e-2 ... 1e-6, each from the previous answer; a lane that
           fails a rung walks to it again from its last converged eps in 2,
           then 4, then 8 geometric sub-steps before it is dropped;
  polish   the untilted face from the last answer, as it is, then with the
           three rotation angles snapped to {0, pi};
  accept   what production asks of an answer: the full seven-wave residual
           <= 1e-8 and the waves in order, in the planar frame (the frame the
           answer was produced in, as the five-wave rescue verifies); the
           zones rotated back to the solver frame, the ray at xi = 0
           resolved and the state there physical.

The study's diagnostics -- condition numbers, wave signatures, per-rung
records, the planar polish -- are dropped: this rung only ever sees faces
that are not planar, and production needs the answer, not the path.

Batch-independent like the rest of the batched solver: every lane's walk
sees only its own data, so it runs on the worker pool chunked by lane
(``exact_pool.map_lanes``).
"""
from __future__ import annotations

import numpy as np

EPS_LADDER = (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6)
SUBSTEPS = (2, 4, 8)
ACC = 1e-10             # the Newton target on every rung and in the polish
MAX_ITER = 60           # Newton iterations per solve, as the ledger measured
                        # it -- the rung's own budget, not production's
VERIFY = 1e-8           # what counts as solved: exact_flux's RMHD_VERIFY_TOL

# why a lane was refused: 0 accepted, 1 rung 0 did not converge, 2 lost on
# the ladder, 3 the polish did not verify, 4 the waves out of order, 5 the
# ray, 6 unphysical at xi = 0
REASONS = {0: "accepted", 1: "no start", 2: "lost on the ladder",
           3: "polish failed", 4: "self-crossing", 5: "ray", 6: "unphysical"}


def _sub(state, i):
    return [c[i] for c in state]


def _b2(s, Bn):
    """Comoving b^2, as the solver computes it."""
    _, _, vx, vy, vz, By, Bz = s
    v2 = vx * vx + vy * vy + vz * vz
    eta = Bn * vx + By * vy + Bz * vz
    return (Bn * Bn + By * By + Bz * Bz) * (1.0 - v2) + eta * eta


def tilt(s, Bn, ang):
    """Rotate the tangential field by ``ang`` about the normal; rho, gas
    pressure and v are held and the total pressure moves with b^2.
    ``ang = 0`` returns the input bit for bit."""
    out = [np.array(c, dtype=float, copy=True) for c in s]
    c, sn = np.cos(ang), np.sin(ang)
    out[5] = c * s[5] - sn * s[6]
    out[6] = sn * s[5] + c * s[6]
    out[1] = s[1] + 0.5 * (_b2(out, Bn) - _b2(s, Bn))
    return out


def tilted(L, R, Bn, eps):
    return tilt(L, Bn, eps), tilt(R, Bn, -eps)


def rung0_problem(sL, sR, Bn):
    """The first rung's problem for solver-frame states ``sL``, ``sR``: the
    planar-frame states tilted by ``EPS_LADDER[0]``. The caller seeds it."""
    from rmhd.batched import planar5_b as P5B
    L0, R0 = P5B.to_planar(sL, sR, Bn)[:2]
    return tilted(L0, R0, Bn, EPS_LADDER[0])


def _snap(a):
    w = (np.asarray(a, float) + np.pi) % (2.0 * np.pi) - np.pi
    return np.where(np.abs(w) <= 0.5 * np.pi, 0.0, np.pi)


def _solved(r):
    return np.asarray(r["nrm"]) <= VERIFY


def _walk(F, L0, R0, Bn, u_from, e_from, e_to, nsub, max_iter, n_it):
    """Step lanes from ``e_from`` (per lane) to ``e_to`` in ``nsub`` geometric
    steps, each from the last. Returns ``(reached, u, e_last)``."""
    m = Bn.size
    u = [c.copy() for c in u_from]
    e_last = e_from.copy()
    going = np.ones(m, bool)
    for j in range(1, nsub + 1):
        i = np.flatnonzero(going)
        if i.size == 0:
            break
        e = (np.full(i.size, e_to) if j == nsub
             else e_from[i] * (e_to / e_from[i]) ** (j / nsub))
        Lt, Rt = tilted(_sub(L0, i), _sub(R0, i), Bn[i], e)
        r = F["fullcontact6"](Lt, Rt, _sub(u, i), Bn[i], accuracy=ACC,
                              max_iter=max_iter)
        n_it[i] += np.asarray(r["n_iter"])
        ok = _solved(r)
        for c in range(6):
            u[c][i[ok]] = r["unk"][c][ok]
        e_last[i[ok]] = e[ok]
        going[i[~ok]] = False
    return going, u, e_last


def continuation_and_ray(P, gamma, sL, sR, Bn, *, seeds, accuracy, max_iter,
                         n_retries):
    """The rung's body on solver-frame lanes ``sL``, ``sR``, ``Bn``.

    ``seeds`` holds the first rung's ML seeds for ``rung0_problem``:
    ``seed6`` (a 6-list of per-lane arrays), ``keys`` (per-lane retry keys or
    None) and ``extra`` (a list of further 6-lists, or None). ``accuracy``
    and ``max_iter`` are production's and are not used: every solve works to
    ``ACC`` within ``MAX_ITER`` iterations and accepts at ``VERIFY``, the
    settings the ledger measured.

    Returns a dict of per-lane arrays: ``good``, ``star`` (solver frame,
    valid where good), ``region``, ``reason`` (``REASONS``), ``n_iter`` (all
    Newton iterations of the lane), ``eps`` (the smallest tilt reached).
    """
    from rmhd.batched import ml_b as MB
    from rmhd.batched import planar5_b as P5B
    from src.physics.exact_flux import _self_crossing, _WAVE_ORDER_TOL

    F = P.get("full")
    if F is None:
        from rmhd.batched import fullcontact_b as FB
        F = P["full"] = FB.make_solver(gamma)
    n = Bn.size
    reason = np.ones(n, np.int8)
    n_it = np.zeros(n, int)
    eps_reached = np.full(n, np.nan)
    out = dict(good=np.zeros(n, bool), star=None, region=None, reason=reason,
               n_iter=n_it, eps=eps_reached)
    if n == 0:
        return out
    L0, R0, alpha, _ = P5B.to_planar(sL, sR, Bn)

    # ── rung 0, seeded like production ──────────────────────────────────
    Lt, Rt = tilted(L0, R0, Bn, EPS_LADDER[0])
    r = MB.solve_with_retries(F["fullcontact6"], Lt, Rt, Bn, seeds["seed6"],
                              seeds["keys"], accuracy=ACC, max_iter=MAX_ITER,
                              n_retries=n_retries, seeds_extra=seeds["extra"])
    n_it += np.asarray(r.get("n_iter_total", r["n_iter"]))
    alive = _solved(r).copy()
    cur = [np.array(c, dtype=float, copy=True) for c in r["unk"]]
    e_last = np.where(alive, EPS_LADDER[0], np.nan)

    # ── the ladder ──────────────────────────────────────────────────────
    for eps in EPS_LADDER[1:]:
        i = np.flatnonzero(alive)
        if i.size == 0:
            break
        Lk, Rk = tilted(_sub(L0, i), _sub(R0, i), Bn[i], eps)
        r = F["fullcontact6"](Lk, Rk, _sub(cur, i), Bn[i], accuracy=ACC,
                              max_iter=MAX_ITER)
        n_it[i] += np.asarray(r["n_iter"])
        ok = _solved(r)
        for c in range(6):
            cur[c][i[ok]] = r["unk"][c][ok]
        e_last[i[ok]] = eps
        fail = i[~ok]
        for ns in SUBSTEPS:
            if fail.size == 0:
                break
            it_f = np.zeros(fail.size, int)
            reached, u, el = _walk(F, _sub(L0, fail), _sub(R0, fail), Bn[fail],
                                   _sub(cur, fail), e_last[fail], eps, ns,
                                   MAX_ITER, it_f)
            n_it[fail] += it_f
            for c in range(6):
                cur[c][fail] = u[c]
            e_last[fail] = el
            fail = fail[~reached]
        alive[fail] = False
    eps_reached[:] = e_last
    started = np.isfinite(e_last)
    reason[:] = np.where(started, 2, 1)

    # ── the polish on the untilted face ─────────────────────────────────
    # every lane that started, from the furthest point it reached (as the
    # measured study did: 29 of its 748 answers stopped above eps = 1e-6)
    route = np.zeros(n, np.int8)
    zones = [[np.full(n, np.nan) for _ in range(7)] for _ in range(6)]
    VsLv = [np.full(n, np.nan) for _ in range(3)]
    VsRv = [np.full(n, np.nan) for _ in range(3)]
    crossed = np.zeros(n, bool)
    snapped = [cur[0], cur[1], _snap(cur[2]), cur[3], _snap(cur[4]),
               _snap(cur[5])]
    for code, s in ((1, cur), (2, snapped)):
        i = np.flatnonzero(started & (route == 0))
        if i.size == 0:
            break
        r = F["fullcontact6"](_sub(L0, i), _sub(R0, i), _sub(s, i), Bn[i],
                              accuracy=ACC, max_iter=MAX_ITER)
        n_it[i] += np.asarray(r["n_iter"])
        conv = _solved(r)
        x = _self_crossing(r["zones"], r["VsLv"], r["VsRv"], _WAVE_ORDER_TOL)
        crossed[i[conv & x]] = True
        g = np.flatnonzero(conv & ~x)
        if g.size == 0:
            continue
        w = i[g]
        route[w] = code
        for k in range(6):
            for j in range(7):
                zones[k][j][w] = r["zones"][k][j][g]
        for m in range(3):
            VsLv[m][w] = r["VsLv"][m][g]
            VsRv[m][w] = r["VsRv"][m][g]
    lost = started & (route == 0)
    reason[lost] = np.where(crossed[lost], 4, np.where(alive[lost], 3, 2))
    good = route > 0
    if not good.any():
        out["good"] = good
        return out

    # ── the ray at xi = 0, in the solver frame ──────────────────────────
    w = np.flatnonzero(good)
    Zs = [P5B.from_planar([c[w] for c in z], alpha[w]) for z in zones]
    st, reg, e = P["ray"].state_at_xi(
        _sub(sL, w), _sub(sR, w), Zs, [v[w] for v in VsLv], [v[w] for v in VsRv],
        Bn[w], gamma, P["xi"], P["fan_p"], P["fan_n"])
    rho, Pt, vn, v1, v2, b1, b2 = st
    vv = vn * vn + v1 * v1 + v2 * v2
    W2 = 1.0 / np.maximum(1.0 - vv, 1e-300)
    eta = Bn[w] * vn + b1 * v1 + b2 * v2
    bb = (Bn[w] ** 2 + b1 ** 2 + b2 ** 2) / W2 + eta ** 2
    phys = (rho > 0.0) & (vv < 1.0) & (Pt - 0.5 * bb > 0.0)
    for c in st:
        phys &= np.isfinite(c)
    ok = ~e & phys
    reason[w] = np.where(e, 5, np.where(~phys, 6, 0))
    good[w] = ok
    star = [np.zeros(n) for _ in range(7)]
    region = np.full(n, -1, dtype=int)
    for j in range(7):
        star[j][w] = st[j]
    region[w] = np.asarray(reg)
    out.update(good=good, star=star, region=region)
    return out
