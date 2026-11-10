"""The Julia exact-flux chain as the exact solver of ``exact_flux_batched``
(2026-11-08, user: "port the missing pieces and couple it to a 64^2 run").

``RMHD_JULIA=1``
    Julia answers every attempted lane of a sweep: the class, the seven-wave
    Newton with its retries or (degenerate input) the three-wave solver, the
    gates (physical, wave order, admissibility, reclassified refused) and the
    rescues -- five-wave with its ladder, the late planar seven-wave rung with
    the planar network, the four-unknown solver (julia_proto's
    ``exact_chain``).  Python keeps everything around it: HLLD, the weak /
    upwind gates, the canonical orientation, the seven-wave seed networks and
    the flux assembly from the returned xi = 0 state.

``RMHD_JULIA=compare``
    Production answers the run.  Julia answers the same lanes beside it, and
    each sweep's agreement is appended to ``RMHD_JULIA_COMPARE_LOG`` (a CSV);
    the lanes whose verdicts differ go to ``<log>.lanes/<sweep>.npz``.

The Julia side is one long-lived process (julia_proto/serve_chain.jl) on its
own threads, talking over pipes: nothing is installed into Python.

    RMHD_JULIA_EXE      the julia binary
    RMHD_JULIA_PROTO    the julia_proto directory (its Project.toml, src/)
    RMHD_JULIA_THREADS  Julia threads (64)
    RMHD_JULIA_WARMUP   a capture directory: compile every rung at start
    RMHD_JULIA_DIAG     1: the per-lane diagnostics production's LAST_DIAG
                        carries (Newton iterations, each rescue's refusal code
                        and iterations, the late rung's rotations and
                        steering, the four-unknown steering, the seven-wave
                        verification residual); 0 (default): the lean answer.
                        A run with a harvest gets them anyway, plus each
                        answer's zones and wave speeds for the solved rows.

Julia implements ONE configuration -- the resolution ladder's (calea
rmhd_final-loose13 + HLLD-fix14 with the ladder's switches).  ``preflight``
refuses any other, rather than letting a switch silently mean nothing.
"""
from __future__ import annotations

import atexit
import os
import subprocess
import threading
import time

import numpy as np

MAGIC = 50004242.0
NOUT, NDIAG, NZONE = 16, 20, 48
DIAG = os.environ.get("RMHD_JULIA_DIAG", "0").strip() not in ("0", "", "off")
_PROC = None
# RMHD_MERGE_SWEEPS runs each sweep in its own thread: one request on the pipe at a time
_LOCK = threading.RLock()
_STATS = dict(calls=0, lanes=0, seconds=0.0)
_SWEEP = [0]

# rung codes of serve_chain.jl
SEVEN, FIVE, LATE, FOUR, THREE, LOST = 1, 2, 3, 4, 5, 0


def preflight(EF):
    """The switches Julia's chain hard-codes, against what this process runs."""
    env = os.environ.get
    need = [
        ("RMHD_SLOWSHOCK_LOOSE_RETRY", env("RMHD_SLOWSHOCK_LOOSE_RETRY", "0") == "1"),
        ("RMHD_SLOWSHOCK_RETRY on", env("RMHD_SLOWSHOCK_RETRY", "1") == "1"),
        ("RMHD_ALFVEN_EXACT on", env("RMHD_ALFVEN_EXACT", "1") == "1"),
        ("RMHD_FAST_EDGE_SCAN on", env("RMHD_FAST_EDGE_SCAN", "1") == "1"),
        ("RMHD_QUARTIC ferrari", env("RMHD_QUARTIC", "ferrari") == "ferrari"),
        ("RMHD_COPLANAR7 late", env("RMHD_COPLANAR7", "0") == "late"),
        ("RMHD_COPLANAR7_CALLS 4", env("RMHD_COPLANAR7_CALLS", "4") == "4"),
        ("RMHD_COPLANAR7_DS 1e-9", float(env("RMHD_COPLANAR7_DS", "1e-9")) == 1e-9),
        ("RMHD_COPLANAR7_ITER 25", env("RMHD_COPLANAR7_ITER", "25") == "25"),
        ("RMHD_COPLANAR7_NET v2a", env("RMHD_COPLANAR7_NET", "data/ml_planar_v2a.pt") == "data/ml_planar_v2a.pt"),
        ("RMHD_COPLANAR7_NET_FALLBACK all", env("RMHD_COPLANAR7_NET_FALLBACK", "all") == "all"),
        ("RMHD_COPLANAR7_P 1", env("RMHD_COPLANAR7_P", "1") == "1"),
        ("RMHD_COPLANAR7_REGULAR 1", env("RMHD_COPLANAR7_REGULAR", "1") == "1"),
        ("RMHD_SLOW_FAN_EXACT off", env("RMHD_SLOW_FAN_EXACT", "0") in ("0", "")),
        ("RMHD_SLOWSHOCK_ADMISSIBLE off", env("RMHD_SLOWSHOCK_ADMISSIBLE", "0") in ("0", "")),
        ("RMHD_LINEAR_SEED off", env("RMHD_LINEAR_SEED", "0") in ("0", "")),
        ("RMHD_PLANAR4_STAR_SET lean", env("RMHD_PLANAR4_STAR_SET", "lean") == "lean"),
        ("RMHD_PLANAR4_STAR_ITER 25", EF._PLANAR4_STAR_ITER == 25),
        ("five-wave rescue + ladder", EF._PLANAR5_FALLBACK and EF._PLANAR5_LADDER),
        ("crossed lanes offered", EF._PLANAR5_CROSSED),
        ("four-unknown rung", EF._PLANAR4),
        ("four-unknown star starts, DS 1e-4", EF._PLANAR4_STAR and EF._PLANAR4_STAR_DS == 1e-4),
        ("rotation re-solve", EF._PLANAR4_ROT),
        ("admissibility gate, DS 1e-9", EF._ADMISSIBLE_GATE and EF._ADMISSIBLE_GATE_DS == 1e-9),
        ("late rung", EF._COPLANAR7_LATE),
        ("no three-wave rescue rung", not EF._3WAVE_FALLBACK),
        ("no continuation", not EF._CONTINUATION),
        ("no compound routing", not EF._COMPOUND_SKIP and not EF._PLANAR4_SKIP_COMPOUND),
        ("planar pre-check, tol 1e-6", EF._PLANAR_PRECHECK and EF._PLANAR_TOL == 1e-6),
        ("five-wave verify 1e-8", EF._PLANAR5_VERIFY == 1e-8),
        ("wave order 1e-6", EF._WAVE_ORDER and EF._WAVE_ORDER_TOL == 1e-6),
    ]
    bad = [k for k, ok in need if not ok]
    if bad:
        raise RuntimeError("RMHD_JULIA: Julia's chain implements the ladder's configuration only; "
                           "these differ: " + ", ".join(bad))


def _read(p, n):
    need = 8 * n
    buf = bytearray()
    while len(buf) < need:
        chunk = p.stdout.read(need - len(buf))
        if not chunk:
            raise RuntimeError("the Julia chain server died (see its stderr above)")
        buf += chunk
    return np.frombuffer(bytes(buf), dtype="<f8")


def _start():
    exe = os.environ.get("RMHD_JULIA_EXE", "julia")
    proto = os.environ["RMHD_JULIA_PROTO"]
    threads = os.environ.get("RMHD_JULIA_THREADS", "64")
    cmd = [exe, "-t", threads, "--project=" + proto, "-O3", os.path.join(proto, "serve_chain.jl")]
    warm = os.environ.get("RMHD_JULIA_WARMUP", "")
    if warm:
        cmd.append(warm)
    t0 = time.perf_counter()
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)
    h = _read(p, 2)
    if h[0] != MAGIC:
        raise RuntimeError("the Julia chain server answered %r instead of its ready line" % (h,))
    print("julia chain: server up in %.1f s (%s threads)" % (time.perf_counter() - t0, threads), flush=True)
    atexit.register(_stop)
    return p


def _stop():
    global _PROC
    if _PROC is not None and _PROC.poll() is None:
        try:
            _PROC.stdin.write(np.array([MAGIC, 0, 0, 0, 0, 0, 0], dtype="<f8").tobytes())
            _PROC.stdin.flush()
            _PROC.wait(timeout=60)
        except Exception:
            _PROC.kill()
    if _STATS["calls"]:
        print("julia chain: %d calls, %d lanes, %.1f s in Julia" % (
            _STATS["calls"], _STATS["lanes"], _STATS["seconds"]), flush=True)
        _STATS.update(calls=0, lanes=0, seconds=0.0)
    _PROC = None


def solve(gamma, subL, subR, subBn, seed6, seeds_extra, p_hint, *, accuracy, max_iter,
          n_retries, tau_bt, want_zones=False):
    """One sweep's attempted lanes through Julia's chain (solver frame).
    ``want_zones``: the run harvests -- also the diagnostics and each
    answer's zones and wave speeds."""
    global _PROC
    if tau_bt != 1e-9:
        raise RuntimeError("RMHD_JULIA: Julia's chain hard-codes tau_bt = 1e-9, not %r" % tau_bt)
    extras = [] if seeds_extra is None else list(seeds_extra)
    if n_retries > len(extras):
        raise RuntimeError("RMHD_JULIA: %d retries but %d extra seeds (Julia has no jitter retries)"
                           % (n_retries, len(extras)))
    extras = extras[:n_retries]
    flags = (1 if (DIAG or want_zones) else 0) | (2 if want_zones else 0)
    with _LOCK:
        return _solve_locked(gamma, subL, subR, subBn, seed6, extras, p_hint,
                             accuracy=accuracy, max_iter=max_iter, flags=flags)


def _solve_locked(gamma, subL, subR, subBn, seed6, extras, p_hint, *, accuracy, max_iter, flags=0):
    global _PROC
    if _PROC is None:
        from . import exact_flux as EF
        preflight(EF)
        _PROC = _start()
    n = int(np.asarray(subBn).shape[0])
    lane = lambda cols: np.ascontiguousarray(np.stack([np.asarray(c, dtype=float) for c in cols], axis=1),
                                             dtype="<f8")
    ph = (np.full(n, np.nan) if p_hint is None else np.asarray(p_hint, dtype=float))
    parts = [np.array([MAGIC, n, len(extras), gamma, accuracy, max_iter, flags], dtype="<f8"),
             lane(subL), lane(subR), np.asarray(subBn, dtype="<f8"), lane(seed6)]
    parts += [lane(s) for s in extras]
    parts.append(np.asarray(ph, dtype="<f8"))
    t0 = time.perf_counter()
    _PROC.stdin.write(b"".join(np.ascontiguousarray(a).tobytes() for a in parts))
    _PROC.stdin.flush()
    h = _read(_PROC, 2)
    if h[0] != MAGIC or int(h[1]) != n:
        raise RuntimeError("the Julia chain server answered %r for %d lanes" % (h, n))
    nout = NOUT + (NDIAG if flags & 1 else 0) + (NZONE if flags & 2 else 0)
    out = _read(_PROC, nout * n).reshape(n, nout)
    dt = time.perf_counter() - t0
    _STATS["calls"] += 1
    _STATS["lanes"] += n
    _STATS["seconds"] += dt
    jr = dict(rung=out[:, 0].astype(int), region=out[:, 1].astype(int), reason7=out[:, 2].astype(np.int8),
              cls=out[:, 3].astype(int), attempts=out[:, 4].astype(int), tag=out[:, 5].astype(int),
              tried=out[:, 6].astype(int), seconds=out[:, 7].copy(),
              star=[out[:, 8 + j].copy() for j in range(7)], wall=dt, diag=None, zones=None)
    k = NOUT
    if flags & 1:
        D = out[:, k:k + NDIAG]
        k += NDIAG
        b = lambda c: D[:, c] != 0.0
        i = lambda c: D[:, c].astype(int)
        jr["diag"] = dict(n_iter=i(0), n_iter_total=i(1), bt_dead=b(2), reclass=b(3), conv7=b(4),
                          ray_ok7=b(5), reason5=i(6).astype(np.int8), n_iter5=i(7),
                          reason7p=i(8).astype(np.int8), n_iter7p=i(9),
                          rot7p=np.stack([b(10), b(11)]), steer7p=np.stack([i(12), i(13)]).astype(np.int8),
                          reason4=i(14).astype(np.int8), n_iter4=i(15),
                          kinds4=np.stack([i(16), i(17)]).astype(np.int8), verify=D[:, 18].copy())
    if flags & 2:
        Zf = out[:, k:k + NZONE]
        jr["zones"] = dict(
            zones=[[Zf[:, 7 * z + j].copy() for j in range(7)] for z in range(6)],
            VsLv=[Zf[:, 42 + j].copy() for j in range(3)], VsRv=[Zf[:, 45 + j].copy() for j in range(3)])
    return jr


def as_pipeline(jr):
    """`_run_pipeline`'s return value, from a Julia answer: the seven-wave /
    three-wave lanes are the converged ones; every exact lane carries its star.
    `d` is solve_batch's diag: the class histogram always, the Newton
    statistics with the diagnostics."""
    from rmhd.batched import classify as CL
    rung = jr["rung"]
    n = rung.size
    first = (rung == SEVEN) | (rung == THREE)
    exact = rung > 0
    star = [np.where(exact, s, 0.0) for s in jr["star"]]
    region = np.where(exact, jr["region"], -1)
    cls = jr["cls"].copy()
    res = dict(cls=cls, converged=first.copy(), attempts=jr["attempts"].copy(),
               n_iter_total=np.zeros(n, dtype=int), reclassified=np.zeros(n, dtype=bool),
               VsLv=[np.zeros(n)] * 3, VsRv=[np.zeros(n)] * 3, unk=[np.zeros(n)] * 6)
    d = dict(CL.histogram(cls), n_total=n, julia_wall=jr["wall"])
    dg = jr["diag"]
    if dg is not None:
        res["n_iter"] = dg["n_iter"].copy()
        res["n_iter_total"] = dg["n_iter_total"].copy()
        seven = np.isin(cls, (CL.FULL7, CL.COPLANAR)) & ~dg["bt_dead"]
        conv = dg["conv7"]
        cv7, nc7 = seven & conv, seven & ~conv
        it = dg["n_iter"]
        d.update(n_bt_gate=int((dg["bt_dead"] & np.isin(cls, (CL.FULL7, CL.COPLANAR))).sum()),
                 n_class_change=int(dg["reclass"].sum()), n_converged=int(conv.sum()),
                 frac_unconverged=float((~conv).mean()) if n else 0.0,
                 n_retry_rescued=int(((res["attempts"] > 1) & conv).sum()),
                 mean_newton_iters=float(it[seven].mean()) if seven.any() else 0.0,
                 iters_converged_max=int(it[cv7].max()) if cv7.any() else 0,
                 iters_converged_p50=float(np.median(it[cv7])) if cv7.any() else 0.0,
                 iters_failed_p50=float(np.median(it[nc7])) if nc7.any() else 0.0,
                 iter_hist=np.bincount(it[seven], minlength=1).tolist() if seven.any() else [])
    return res, d, star, region, first.copy(), 0


def rescue_masks(jr, n):
    """The rescue rungs' masks and per-lane diagnostics, from a Julia answer:
    production's codes and iterations with the diagnostics, else coarse ones
    (0 answered, 1 lost, 7/8 not offered) and no iterations."""
    rung, tag, tried = jr["rung"], jr["tag"], jr["tried"]
    take5, take7p, take4 = rung == FIVE, rung == LATE, rung == FOUR
    o5, o7, o4 = (tried & 1) > 0, (tried & 2) > 0, (tried & 4) > 0
    rung5 = np.where(take5, tag, -1).astype(np.int8)
    stage7p = np.where(take7p, tag, 0).astype(np.int8)
    reason7 = jr["reason7"].copy()
    out = dict(take5=take5, take7p=take7p, take4=take4, rung5=rung5, stage7p=stage7p,
               n5_att=int(o5.sum()), n7p_att=int(o7.sum()), n4_att=int(o4.sum()), reason7=reason7,
               # the seven-wave gates' counts, from Julia's reason codes
               n_unphys=int((reason7 == 3).sum()), n_crossed=int(np.isin(reason7, (4, 6)).sum()),
               n_reclass_refused=int((reason7 == 5).sum()), n_ray_failed=int((reason7 == 2).sum()))
    dg = jr["diag"]
    if dg is None:
        out.update(reason5=np.where(o5, np.where(take5, 0, 1), 7).astype(np.int8),
                   reason7p=np.where(o7, np.where(take7p, 0, 1), 8).astype(np.int8),
                   reason4=np.where(o4, np.where(take4, 0, 1), 8).astype(np.int8),
                   iters5=None, iters7p=None, iters4=None, kinds4=None, rot7p=None)
    else:
        # production counts a non-planar lane's skip (reason 2) as offered to neither
        out.update(reason5=dg["reason5"], reason7p=dg["reason7p"], reason4=dg["reason4"],
                   iters5=dg["n_iter5"], iters7p=dg["n_iter7p"], iters4=dg["n_iter4"],
                   kinds4=dg["kinds4"], rot7p=dg["rot7p"])
    return out


def verified(jr, take, cls, take5, take7p, take4, tol):
    """exact_flux's `verified` over the attempted lanes: with the diagnostics,
    a seven- or three-wave answer passes production's own check (the
    seven-wave residual at the answer, RK4 fans, <= tol) as there; without,
    a seven-wave answer counts as verified (Julia accepted it on that residual
    at that tolerance) and a three-wave one does not.  The rescues certify
    inside their solvers, as in production."""
    from rmhd.batched import classify as CL
    seven = np.isin(cls, (CL.FULL7, CL.COPLANAR))
    dg = jr["diag"]
    if dg is None:
        v = take & seven
    else:
        v = take & (dg["verify"] <= tol)
    return v | take5 | take7p | take4


def harvest(harvester, jr, subL, subR, subBn, *, planar5_rows, rescue_rows):
    """The harvest rows production writes, from a Julia answer, in its order:
    the seven-wave call (solved rows for its answers, unsolved rows with their
    reasons for the rest), the reclassified lanes (RMHD_HARVEST_RESCUES), the
    five-wave answers (source 1), the late rung's and the four-unknown
    solver's (sources 2, 3; RMHD_HARVEST_RESCUES).  Needs the zones."""
    from rmhd.batched import classify as CL
    from src.physics.harvest import R_RECLASSIFIED
    dg, zz = jr["diag"], jr["zones"]
    if dg is None or zz is None:
        raise RuntimeError("julia_chain.harvest: the answer carries no zones (want_zones)")
    rung, reason7, cls = jr["rung"], jr["reason7"], jr["cls"]
    n = rung.size
    seven_wave = np.isin(cls, (CL.FULL7, CL.COPLANAR))
    # accepted at the harvest = every gate passed, before the reclassified refusal
    accepted = (rung == SEVEN) | (rung == THREE) | (reason7 == 5)
    zf = [[np.where(rung == SEVEN, c, 0.0) for c in z] for z in zz["zones"]]
    vl = [np.where(rung == SEVEN, c, 0.0) for c in zz["VsLv"]]
    vr = [np.where(rung == SEVEN, c, 0.0) for c in zz["VsRv"]]
    harvester.record(subL, subR, subBn, cls=cls.astype(np.int8), converged=dg["conv7"], zones=zf,
                     VsLv=vl, VsRv=vr, attempts=jr["attempts"], accepted=accepted,
                     ray_ok=dg["ray_ok7"], crossed=np.isin(reason7, (4, 6)), source=0,
                     seven_wave=seven_wave)
    if rescue_rows and (reason7 == 5).any():
        harvester.record_unsolved(subL, subR, subBn, np.flatnonzero(reason7 == 5), reason=R_RECLASSIFIED,
                                  cls=cls, attempts=jr["attempts"])

    def rows(w, source, extras=None):
        k = int(w.size)
        if k == 0:
            return
        harvester.record(
            [np.asarray(c)[w] for c in subL], [np.asarray(c)[w] for c in subR], np.asarray(subBn)[w],
            cls=np.full(k, CL.COPLANAR, dtype=np.int8), converged=np.ones(k, dtype=bool),
            zones=[[c[w] for c in z] for z in zz["zones"]],
            VsLv=[v[w] for v in zz["VsLv"]], VsRv=[v[w] for v in zz["VsRv"]],
            attempts=np.full(k, 2, dtype=int), accepted=np.ones(k, dtype=bool),
            seven_wave=np.ones(k, dtype=bool), source=source, extras=extras)

    if planar5_rows:
        rows(np.flatnonzero(rung == FIVE), 1)
    if rescue_rows:
        w = np.flatnonzero(rung == LATE)
        rows(w, 2, dict(steer=dg["steer7p"][:, w], rot=dg["rot7p"][:, w],
                        stage=jr["tag"][w].astype(np.int8)))
        w = np.flatnonzero(rung == FOUR)
        if w.size:
            Z = zz["zones"]
            # a rotation answer's R3 / R6 differ from R2 / R7 (exact_flux._planar4_take)
            rot4 = np.stack([(Z[1][5][w] != Z[0][5][w]) | (Z[1][6][w] != Z[0][6][w]),
                             (Z[4][5][w] != Z[5][5][w]) | (Z[4][6][w] != Z[5][6][w])])
            rows(w, 3, dict(steer=dg["kinds4"][:, w], rot=rot4, stage=np.zeros(w.size, dtype=np.int8)))


def prod_rungs(take, take3, take5, take7p, take4, takec, cls):
    """Production's verdict per lane, in Julia's rung codes."""
    seven = np.isin(cls, (0, 4))                       # FULL7, COPLANAR
    r = np.zeros(take.size, dtype=int)
    r[take & seven] = SEVEN
    r[take & ~seven] = THREE
    r[take5] = FIVE
    r[take7p] = LATE
    r[take4] = FOUR
    r[take3 | takec] = 9                               # rungs Julia does not run
    return r


def compare(idir, gamma, subL, subR, subBn, seed6, seeds_extra, p_hint, *, accuracy, max_iter,
            n_retries, tau_bt, prod):
    """RMHD_JULIA=compare: Julia on the sweep production just answered."""
    with _LOCK:
        return _compare_locked(idir, gamma, subL, subR, subBn, seed6, seeds_extra, p_hint,
                               accuracy=accuracy, max_iter=max_iter, n_retries=n_retries,
                               tau_bt=tau_bt, prod=prod)


def _compare_locked(idir, gamma, subL, subR, subBn, seed6, seeds_extra, p_hint, *, accuracy,
                    max_iter, n_retries, tau_bt, prod):
    path = os.environ.get("RMHD_JULIA_COMPARE_LOG", "julia_compare.csv")
    t0 = time.perf_counter()
    jr = solve(gamma, subL, subR, subBn, seed6, seeds_extra, p_hint, accuracy=accuracy,
               max_iter=max_iter, n_retries=n_retries, tau_bt=tau_bt)
    pr = prod_rungs(prod["take"], prod["take3"], prod["take5"], prod["take7p"], prod["take4"],
                    prod["takec"], prod["cls"])
    jrung = jr["rung"]
    k = _SWEEP[0]
    _SWEEP[0] += 1
    both = (pr > 0) & (pr != 9) & (jrung > 0)
    ps = np.stack([np.asarray(c, dtype=float) for c in prod["star"]], axis=1)
    js = np.stack(jr["star"], axis=1)
    rel = np.zeros(pr.size)
    if both.any():
        d = np.abs(js[both] - ps[both]) / np.maximum(np.abs(ps[both]), 1.0)
        rel[both] = d.max(axis=1)
    same = both & (pr == jrung)
    other = both & (pr != jrung)
    only_p = (pr > 0) & (jrung == 0)
    only_j = (pr == 0) & (jrung > 0)
    q = lambda m: (float(rel[m].max()) if m.any() else 0.0)
    counts = np.zeros((6, 6), dtype=int)                # production rows, julia columns: 0 lost, 1..5 rungs
    for a in range(6):
        for b in range(6):
            counts[a, b] = int(((pr == a) & (jrung == b)).sum())
    new = not os.path.exists(path)
    with open(path, "a") as f:
        if new:
            f.write("sweep,idir,n,exact_prod,exact_julia,same_rung,other_rung,only_prod,only_julia,"
                    "max_rel_same,max_rel_other,n_rel_gt_1e6,julia_wall,counts\n")
        f.write("%d,%d,%d,%d,%d,%d,%d,%d,%d,%.3e,%.3e,%d,%.3f,%s\n" % (
            k, idir, pr.size, int((pr > 0).sum()), int((jrung > 0).sum()), int(same.sum()),
            int(other.sum()), int(only_p.sum()), int(only_j.sum()), q(same), q(other),
            int((rel > 1e-6).sum()), jr["wall"], " ".join(str(x) for x in counts.ravel())))
    odd = only_p | only_j | (rel > 1e-6) | (pr == 9)
    if odd.any():
        dd = path + ".lanes"
        os.makedirs(dd, exist_ok=True)
        np.savez_compressed(
            os.path.join(dd, "%06d.npz" % k), sweep=k, idir=idir, gamma=gamma,
            sL=np.stack(subL, axis=1)[odd], sR=np.stack(subR, axis=1)[odd],
            Bn=np.asarray(subBn)[odd], seed6=np.stack(seed6, axis=1)[odd],
            extra=(np.stack([np.stack(s, axis=1)[odd] for s in seeds_extra]) if seeds_extra else np.zeros(0)),
            p_hint=(np.asarray(p_hint)[odd] if p_hint is not None else np.zeros(0)),
            prod_rung=pr[odd], julia_rung=jrung[odd], cls_prod=np.asarray(prod["cls"])[odd],
            cls_julia=jr["cls"][odd], prod_star=ps[odd], julia_star=js[odd], rel=rel[odd],
            reason7_prod=np.asarray(prod["reason7"])[odd], reason7_julia=jr["reason7"][odd])
    return time.perf_counter() - t0


_RUNG_FILE = [None]


def log_rungs(path, idir, n_faces, n_att, seven, degen, take5, take7p, take4, lin_fb, wall):
    """RMHD_RUNG_LOG: one line per sweep -- which solver answered, and the
    wall of the exact part (seeds, solve, rescues, flux assembly)."""
    new = not os.path.exists(path)
    with open(path, "a") as f:
        if new:
            f.write("idir,faces,attempted,seven,three,five,late,four,lost,linear_fallback,wall\n")
        n_ex = int(seven.sum() + degen.sum() + take5.sum() + take7p.sum() + take4.sum())
        f.write("%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%.4f\n" % (
            idir, n_faces, n_att, int(seven.sum()), int(degen.sum()), int(take5.sum()),
            int(take7p.sum()), int(take4.sum()), n_att - n_ex, int(lin_fb.sum()), wall))
