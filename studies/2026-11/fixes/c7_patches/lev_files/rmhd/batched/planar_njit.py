"""Compiled per-lane twins of the planar solvers' numpy root-finders
(``RMHD_PLANAR_NJIT=1``; off by default, the numpy code is the reference).

Why: profiled 2026-11-06 on hard faces of the 64^2 runs (exact_flux_batched
on 3-6 faces that reach the planar rungs, the late runs' switches), the
planar solves are not dear for their Python glue but for three numpy
root-finders that each evaluate a residual dozens of times on a handful of
lanes:

* four-unknown solver, a pressure-steered slow shock
  (`planar4_b.slow_shock_p`): a 501-node scan, then up to three bisections
  of 60 trips -- 2.3 s of 6.0 s on three 4Q faces, 6.0 s of 14.0 s on three
  rotor faces;
* the fast shock's second search (`shock_b.fast_edge_scan`): 3.1 s of the
  rotor's 14.0 s;
* the slow shock's retry (`slow_shock_b.retry`): 3.0 s of 13.3 s on six 4Q
  faces.

Each is transcribed here line for line from its numpy source and run per
lane inside one prange loop, with the residual, post-shock state, bisection
and wave speed the shock port (`shock_njit`) already compiles and checks
bit for bit against numpy.  What stays in numpy is what each function does
once per call on whole arrays (the window bounds, the node fractions), so
the kernels see the very same numbers.

Every kernel uses ``error_model="numpy"`` and its outputs are pre-filled as
failed: an exception raised inside a prange body is swallowed and the lane's
outputs are simply never written (numba-kernel-pitfalls).
"""
from __future__ import annotations

import math
import os

import numpy as np

try:
    from numba import njit, prange
    HAVE_NUMBA = True
except ImportError:                       # pragma: no cover
    HAVE_NUMBA = False

_NAN = float("nan")


def mode() -> str:
    """``RMHD_PLANAR_NJIT``: unset/0 numpy (the reference), 1/njit the
    compiled kernels, check both -- numpy's answer is returned and every
    lane on which the two differ is counted in `CHECK` (and dumped to
    ``RMHD_PLANAR_NJIT_DUMP``/<name>_<k>.npz if that directory is set)."""
    m = os.environ.get("RMHD_PLANAR_NJIT", "0").lower()
    if m in ("0", "", "off", "numpy"):
        return "numpy"
    return "check" if m == "check" else "njit"


def wanted() -> bool:
    return mode() == "njit"


CHECK = {}


def _same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return (a == b) | (np.isnan(a) & np.isnan(b)) if a.dtype.kind == "f" else (a == b)


def compare(name, ref, got, **inputs):
    """Lane by lane, bit for bit: ``(Vs, behind7, err)`` of numpy (``ref``)
    against the kernel (``got``)."""
    ok = _same(ref[0], got[0]) & _same(ref[2], got[2])
    for a, b in zip(ref[1], got[1]):
        ok &= _same(a, b)
    c = CHECK.setdefault(name, [0, 0, 0])
    c[0] += 1
    c[1] += ok.size
    c[2] += int((~ok).sum())
    d = os.environ.get("RMHD_PLANAR_NJIT_DUMP")
    if d and not ok.all():
        os.makedirs(d, exist_ok=True)
        np.savez(os.path.join(d, "%s_%d.npz" % (name, c[0])), bad=~ok,
                 Vs_np=ref[0], Vs_nj=got[0], err_np=ref[2], err_nj=got[2],
                 bh_np=np.stack(ref[1]) if len(ref[1]) else np.zeros(0),
                 bh_nj=np.stack(got[1]) if len(got[1]) else np.zeros(0),
                 **{k: (np.stack(v) if isinstance(v, (list, tuple)) else np.asarray(v))
                    for k, v in inputs.items()})


def _pieces():
    """shock_njit's compiled residual, post-shock state, bisection and the
    fan port's compiled wave speed."""
    from . import shock_njit as SN
    pp_c, xi_c = SN._compiled()
    if SN._PIECES is None:
        SN._LANES = SN._build(pp_c, xi_c)
    p = SN._PIECES
    return p["shockfunc"], p["behind"], p["bisect"], xi_c


# ── planar4_b.slow_shock_p ─────────────────────────────────────────────────

_SLOW_P = None


def _build_slow_p():
    sf, behind, bisect, xi_c = _pieces()

    @njit(parallel=True, cache=False, error_model="numpy")
    def lanes(Pt, X, Bn, xs, la, ra, left, t, gamma, accuracy, max_roots, bis_iter,
              out_Vs, out_bh, err):
        N = Pt.shape[0]
        K = t.shape[0]
        kroot = 1 if left else 2
        for n in prange(N):
            ah = (X[0, n], X[1, n], X[2, n], X[3, n], X[4, n], X[5, n], X[6, n])
            if left:
                lo, hi = la[n], xs[n]
            else:
                lo, hi = xs[n], ra[n]
            span = hi - lo
            V = np.empty(K)
            f = np.empty(K)
            for k in range(K):
                V[k] = lo + t[k] * span
                f[k] = sf(V[k], Pt[n], ah, Bn[n], gamma)[0]
            # the brackets, weakest (nearest the slow end) first; ties by
            # position, as numpy's stable argsort orders them
            nb = 0
            idx = np.empty(K - 1, dtype=np.int64)
            key = np.empty(K - 1)
            for j in range(K - 1):
                a, b = f[j], f[j + 1]
                if math.isfinite(a) and math.isfinite(b):
                    sa = 1.0 if a > 0.0 else (-1.0 if a < 0.0 else 0.0)
                    sb = 1.0 if b > 0.0 else (-1.0 if b < 0.0 else 0.0)
                    if sa * sb < 0.0:
                        idx[nb] = j
                        key[nb] = abs(V[j] - xs[n])
                        nb += 1
            order = np.argsort(key[:nb], kind="mergesort")
            tol = 1e-6 * (1.0 + abs(xs[n]))
            found = False
            for m in range(min(nb, max_roots)):
                j = idx[order[m]]
                Vs = bisect(Pt[n], ah, Bn[n], gamma, V[j], f[j], V[j + 1], f[j + 1],
                            accuracy, bis_iter)
                bh, e = behind(Vs, Pt[n], ah, Bn[n], gamma)
                if e or not math.isfinite(Vs):
                    continue
                xb = xi_c(bh, kroot, Bn[n], gamma)
                if left:
                    evo = (Vs >= la[n] - tol) and (Vs <= xs[n] + tol)
                    if math.isfinite(xb):
                        evo = evo and (Vs >= xb - tol)
                else:
                    evo = (Vs <= ra[n] + tol) and (Vs >= xs[n] - tol)
                    if math.isfinite(xb):
                        evo = evo and (Vs <= xb + tol)
                if evo:
                    out_Vs[n] = Vs
                    for q in range(7):
                        out_bh[q, n] = bh[q]
                    found = True
                    break
            if not found:
                out_Vs[n] = ah[2]
                for q in range(7):
                    out_bh[q, n] = ah[q]
            err[n] = not found

    return lanes


def _lanes_f(a, N):
    return np.ascontiguousarray(np.broadcast_to(np.asarray(a, dtype=float), (N,)))


def slow_shock_p(gamma, Pt, X, Bn, left, xs, la, ra, t, accuracy, max_roots,
                 bis_iter=60):
    """Drop-in for the scan-and-bisect body of ``planar4_b.slow_shock_p``:
    ``(Vs, behind7, err)``.  ``xs``, ``la``, ``ra`` and the node fractions
    ``t`` are numpy's own, computed by the caller."""
    global _SLOW_P
    if _SLOW_P is None:
        _SLOW_P = _build_slow_p()
    Xa = np.ascontiguousarray(np.stack(np.broadcast_arrays(
        *[np.asarray(c, dtype=float) for c in X]), axis=0))
    N = Xa.shape[1]
    out_Vs = np.full(N, _NAN)
    out_bh = np.empty((7, N))
    err = np.ones(N, dtype=np.bool_)
    _SLOW_P(_lanes_f(Pt, N), Xa, _lanes_f(Bn, N), _lanes_f(xs, N), _lanes_f(la, N),
            _lanes_f(ra, N), bool(left), np.ascontiguousarray(t, dtype=float),
            float(gamma), float(accuracy), int(max_roots), int(bis_iter),
            out_Vs, out_bh, err)
    return out_Vs, [out_bh[j] for j in range(7)], err


# ── shock_b.fast_edge_scan ─────────────────────────────────────────────────

_EDGE = None


def _build_edge():
    sf, behind, bisect, xi_c = _pieces()

    @njit(cache=False, error_model="numpy")
    def alfven(st, Bn, gamma):
        """shock_b.alfven_speeds for one lane, ordered (la, ra)."""
        rho, P, vx, vy, vz, By, Bz = st
        v2 = vx * vx + vy * vy + vz * vz
        B2 = Bn * Bn + By * By + Bz * Bz
        eta = Bn * vx + By * vy + Bz * vz
        b2 = B2 * (1.0 - v2) + eta * eta
        h = 1.0 + gamma * (P - 0.5 * b2) / ((gamma - 1.0) * rho)
        sw = math.sqrt(max(rho * h + b2, 1e-30))
        dm = eta - sw
        dp = eta + sw
        la = vx + Bn * (1.0 - v2) / dm if abs(dm) > 1e-30 else vx
        ra = vx + Bn * (1.0 - v2) / dp if abs(dp) > 1e-30 else vx
        if la <= ra:
            return la, ra
        return ra, la

    @njit(cache=False, error_model="numpy")
    def h_of(Vs, pb, ah, Bn, gamma, right):
        bh, e = behind(Vs, pb, ah, Bn, gamma)
        la, ra = alfven(bh, Bn, gamma)
        bad = sf(Vs, pb, ah, Bn, gamma)[1]
        if bad or e:
            return _NAN
        return Vs - (ra if right else la)

    @njit(cache=False, error_model="numpy")
    def bisect_h(pb, ah, Bn, gamma, right, xlo, flo, xhi, fhi, tol, max_iter):
        """shock_b.bisect on h_of for one lane (freeze = stop, as bisect_s)."""
        if flo < 0.0:
            lo, hi, f_lo = xlo, xhi, flo
        else:
            lo, hi, f_lo = xhi, xlo, fhi
        for _ in range(max_iter):
            if abs(hi - lo) <= tol:
                break
            mid = 0.5 * (lo + hi)
            fm = h_of(mid, pb, ah, Bn, gamma, right)
            if not math.isfinite(fm):
                fm = 0.0
            if (f_lo * fm) <= 0.0:
                hi = mid
            else:
                lo = mid
                f_lo = fm
        return 0.5 * (lo + hi)

    @njit(cache=False, error_model="numpy")
    def sgn_(x):
        return 1.0 if x > 0.0 else (-1.0 if x < 0.0 else 0.0)

    @njit(parallel=True, cache=False, error_model="numpy")
    def lanes(pb, A, Bn, right, kroot, sgn, xa, trusted, xmin, xmax, alf, cone, gap,
              lin, lin_out, gamma, accuracy, max_roots, out_Vs, out_bh, err):
        N = pb.shape[0]
        K = lin.shape[0]
        KO = lin_out.shape[0]
        for n in prange(N):
            ah = (A[0, n], A[1, n], A[2, n], A[3, n], A[4, n], A[5, n], A[6, n])
            # ── the split points: zeros of h, nearest the Alfven edge first
            span = xmax[n] - xmin[n]
            V = np.empty(K)
            h = np.empty(K)
            for k in range(K):
                V[k] = xmin[n] + span * lin[k]
                h[k] = h_of(V[k], pb[n], ah, Bn[n], gamma, right)
            nb = 0
            idx = np.empty(K - 1, dtype=np.int64)
            key = np.empty(K - 1)
            for j in range(K - 1):
                if (math.isfinite(h[j]) and math.isfinite(h[j + 1])
                        and sgn_(h[j]) * sgn_(h[j + 1]) < 0.0):
                    idx[nb] = j
                    key[nb] = abs(V[j] - alf[n])
                    nb += 1
            order = np.argsort(key[:nb], kind="mergesort")
            ns = min(nb, max_roots)
            splits = np.empty(ns + 1)
            for m in range(ns):
                j = idx[order[m]]
                splits[m] = bisect_h(pb[n], ah, Bn[n], gamma, right, V[j], h[j],
                                     V[j + 1], h[j + 1], 1e-14, 60)
            splits[ns] = xa[n]
            # ── outward from each, the first admissible root
            found = False
            if trusted[n]:
                slack = 1e-6 * (1.0 + abs(xa[n]))
                W = np.empty(KO + 1)
                f = np.empty(KO + 1)
                for m in range(ns + 1):
                    s = splits[m]
                    far = max(abs(cone[n] - s), 1e-300)
                    lo = min(1e-3 * gap[n], far)
                    W[0] = s + sgn * 0.0
                    for i in range(KO):
                        W[i + 1] = s + sgn * (lo * (far / lo) ** lin_out[i])
                    for i in range(KO + 1):
                        f[i] = sf(W[i], pb[n], ah, Bn[n], gamma)[0]
                    j = -1
                    if math.isfinite(f[0]):
                        s0 = sgn_(f[0])
                        for i in range(KO):
                            if math.isfinite(f[i + 1]) and sgn_(f[i + 1]) * s0 < 0.0:
                                j = i
                                break
                    if j < 0:
                        continue
                    fa, fb = f[j], f[j + 1]
                    if not (math.isfinite(fa) and sgn_(fa) * sgn_(fb) < 0.0):
                        continue
                    Vs = bisect(pb[n], ah, Bn[n], gamma, W[j], fa, W[j + 1], fb,
                                accuracy, 60)
                    bh, e = behind(Vs, pb[n], ah, Bn[n], gamma)
                    if e or not math.isfinite(Vs):
                        continue
                    xb = xi_c(bh, kroot, Bn[n], gamma)
                    if not math.isfinite(xb):
                        continue
                    la_b, ra_b = alfven(bh, Bn[n], gamma)
                    if right:
                        lax = (Vs >= xa[n] - slack) and (Vs <= xb + slack)
                        fastb = Vs >= ra_b - slack
                    else:
                        lax = (Vs <= xa[n] + slack) and (Vs >= xb - slack)
                        fastb = Vs <= la_b + slack
                    if lax and fastb:
                        out_Vs[n] = Vs
                        for q in range(7):
                            out_bh[q, n] = bh[q]
                        found = True
                        break
            if not found:
                out_Vs[n] = ah[2]
                for q in range(7):
                    out_bh[q, n] = ah[q]
            err[n] = not found

    return lanes


def fast_edge_scan(gamma, pb, ahead, Bn, right, xa, trusted, xmin, xmax, alf, cone,
                   gap, lin, lin_out, accuracy, max_roots):
    """Drop-in for the body of ``shock_b.fast_edge_scan`` after its numpy
    preamble (the window, the trusted fast speed, the gap): ``(Vs, behind7,
    err)``."""
    global _EDGE
    if _EDGE is None:
        _EDGE = _build_edge()
    A = np.ascontiguousarray(np.stack(np.broadcast_arrays(
        *[np.asarray(c, dtype=float) for c in ahead]), axis=0))
    N = A.shape[1]
    out_Vs = np.full(N, _NAN)
    out_bh = np.empty((7, N))
    err = np.ones(N, dtype=np.bool_)
    _EDGE(_lanes_f(pb, N), A, _lanes_f(Bn, N), bool(right), 3 if right else 0,
          1.0 if right else -1.0, _lanes_f(xa, N),
          np.ascontiguousarray(np.broadcast_to(np.asarray(trusted, dtype=np.bool_), (N,))),
          _lanes_f(xmin, N), _lanes_f(xmax, N), _lanes_f(alf, N), _lanes_f(cone, N),
          _lanes_f(gap, N), np.ascontiguousarray(lin, dtype=float),
          np.ascontiguousarray(lin_out, dtype=float), float(gamma), float(accuracy),
          int(max_roots), out_Vs, out_bh, err)
    return out_Vs, [out_bh[j] for j in range(7)], err


# ── slow_shock_b.make_solver.retry ─────────────────────────────────────────

_RETRY = None


def _build_retry():
    from . import slow_shock_njit as SSN
    from .alfven_njit import rh_vec_s
    if SSN._PIECES is None:
        SSN.make_solver(5.0 / 3.0)            # builds the pieces; gamma is an argument
    attempt = SSN._PIECES["attempt"]

    @njit(parallel=True, cache=False, error_model="numpy")
    def lanes(A, Byt, Bzt, Bx, Vs_lo, Vs_hi, v_lo, v_hi, is_LS, gamma, accuracy, rh_max,
              rungs, out_Vs, out_bh, ok_out):
        N = Bx.shape[0]
        for n in prange(N):
            ah = (A[0, n], A[1, n], A[2, n], A[3, n], A[4, n], A[5, n], A[6, n])
            tol = accuracy * (1.0 + abs(ah[1]))
            for r in range(rungs.shape[0]):
                Vs_b, bh_b, good, _, _, _ = attempt(
                    ah, Byt[n], Bzt[n], Bx[n], Vs_lo[n], Vs_hi[n], v_lo[n], v_hi[n],
                    is_LS, gamma, rungs[r, 0], rungs[r, 1], v_lo[n], v_hi[n], tol, 40, 8)
                if not good:
                    continue
                res, _ = rh_vec_s(ah, bh_b, Vs_b, Bx[n], gamma)
                rh = 0.0
                for q in range(7):
                    a = abs(res[q])
                    if a > rh or not math.isfinite(a):
                        rh = a
                ok = rh <= rh_max
                for q in range(7):
                    if not math.isfinite(bh_b[q]):
                        ok = False
                if ok:
                    out_Vs[n] = Vs_b
                    for q in range(7):
                        out_bh[q, n] = bh_b[q]
                    ok_out[n] = True
                    break

    return lanes


def slow_retry(gamma, ahead, By_t, Bz_t, Bx, Vs_lo, Vs_hi, v_lo, v_hi, is_LS,
               accuracy, rh_max, rungs):
    """The retry's rungs on the lanes given (their windows from numpy's
    ``retry_window``): ``(Vs, behind7, ok)``, ok where a root with RH residual
    at most ``rh_max`` was found -- slow_shock_b's ``attempt`` per lane
    (slow_shock_njit) plus the retry's own RH and finiteness tests."""
    global _RETRY
    if _RETRY is None:
        _RETRY = _build_retry()
    A = np.ascontiguousarray(np.stack(np.broadcast_arrays(
        *[np.asarray(c, dtype=float) for c in ahead]), axis=0))
    N = A.shape[1]
    out_Vs = np.full(N, _NAN)
    out_bh = np.full((7, N), _NAN)
    ok = np.zeros(N, dtype=np.bool_)
    _RETRY(A, _lanes_f(By_t, N), _lanes_f(Bz_t, N), _lanes_f(Bx, N), _lanes_f(Vs_lo, N),
           _lanes_f(Vs_hi, N), _lanes_f(v_lo, N), _lanes_f(v_hi, N), bool(is_LS),
           float(gamma), float(accuracy), float(rh_max),
           np.ascontiguousarray(np.asarray(rungs, dtype=np.int64).reshape(-1, 2)),
           out_Vs, out_bh, ok)
    return out_Vs, [out_bh[j] for j in range(7)], ok


# fullcontact_b.make_xi through fan_njit's compiled xi was tried (2026-11-07)
# and NOT kept: its quartic differs from numpy's xi_all by a few ulp (median
# 6e-16, max 1e-14 relative; never in which roots exist), so it would move
# every answer at round-off.  The kernels here use that xi only inside
# toleranced admissibility tests (slack 1e-6), where they agreed bit for bit.


def compare_arrays(name, ref, got, **inputs):
    """`compare` for a single per-lane array."""
    compare(name, (ref, [], np.zeros(np.shape(ref), bool)),
            (got, [], np.zeros(np.shape(got), bool)), **inputs)
