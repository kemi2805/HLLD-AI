"""Worker-process pool for the exact flux.

The batched solver's orchestration is single-threaded Python; only its
compiled kernels parallelise, and only over the lanes of ONE call.  On a
many-core node that leaves most cores idle.  Measured 2026-09-06: a 64^2
rotor step is 75 s on a laptop with one process, while the node meant for
the production run has 64 cores.

So the per-lane numerical pipeline -- the seven-/three-wave solve with its
retry ladder, then the xi = 0 ray state -- runs in P worker processes, each
on an interleaved slice of the sweep's lanes with its own numba thread
count.  Everything the main process keeps is either torch (the HLLD
fallback, the frame transforms, the flux assembly) or shared (the ML seeds:
one forward pass for the whole sweep, then sliced per worker).

Why a lane's answer does not depend on which worker solved it: the batched
solver is batch-independent by construction (per-lane keys for the retry
jitter, per-lane seeds for the ensemble; `batched/test_*` assert it), so a
slice of a batch solves to the same numbers as the whole.

Env:  RMHD_POOL=<workers>      off unless >= 2
      RMHD_POOL_THREADS=<n>    numba threads per worker (default cores // P)
      RMHD_POOL_CHUNKS=<n>     interleaved slices per sweep (default P; more
                               balances uneven lane costs at the price of
                               the per-call fixed cost, replicated per slice)
      RMHD_POOL_MIN=<n>        below this many lanes, solve in-process
                               (default 8)
"""
from __future__ import annotations

import atexit
import os

import numpy as np

_EXEC = None
_EXEC_KEY = None
_PARTS = None            # worker side: the compiled solver for one gamma


def config():
    """The pool configuration from the environment, or None when off."""
    w = int(os.environ.get("RMHD_POOL", "0") or 0)
    if w < 2:
        return None
    ncpu = os.cpu_count() or w
    thr = int(os.environ.get("RMHD_POOL_THREADS", "0") or 0) or max(1, ncpu // w)
    chunks = int(os.environ.get("RMHD_POOL_CHUNKS", "0") or 0) or w
    min_lanes = int(os.environ.get("RMHD_POOL_MIN", "8") or 8)
    return dict(workers=w, threads=thr, chunks=chunks, min_lanes=min_lanes)


# ── worker side ──────────────────────────────────────────────────────────
def _worker_init(gamma, threads):
    """Runs once per worker, in the fresh (spawned) interpreter.

    The thread count goes into the environment BEFORE numba is imported
    (its pool size is fixed at import); `set_num_threads` afterwards covers
    the case where the parent's `__main__` already imported it during the
    spawn bootstrap.
    """
    os.environ["NUMBA_NUM_THREADS"] = str(int(threads))
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"
    import numba
    numba.set_num_threads(max(1, min(int(threads), numba.config.NUMBA_NUM_THREADS)))
    import torch
    torch.set_num_threads(1)
    np.seterr(all="ignore")
    global _PARTS
    from src.physics import exact_flux as EF
    _PARTS = EF._batched_parts(gamma)


def _run_chunk(payload):
    subL, subR, subBn, seed6, keys, seeds_extra, kw = payload
    from src.physics import exact_flux as EF
    gamma = kw.pop("gamma")
    return EF._solve_and_ray(_PARTS, gamma, subL, subR, subBn, seed6, keys,
                             seeds_extra, **kw)


def _run_rescue(payload):
    """One chunk of a rescue rung: the named `exact_flux` function on the
    worker's compiled parts.  The rungs take (P, gamma, sL, sR, Bn, **kw)
    and return a dict of per-lane arrays (or None where nothing was
    accepted), which `map_lanes` merges."""
    name, sL, sR, Bn, kw = payload
    from src.physics import exact_flux as EF
    gamma = kw.pop("gamma")
    return getattr(EF, name)(_PARTS, gamma, sL, sR, Bn, **kw)


# ── main-process side ────────────────────────────────────────────────────
def _executor(gamma, cfg):
    global _EXEC, _EXEC_KEY
    key = (float(gamma), cfg["workers"], cfg["threads"])
    if _EXEC is None or _EXEC_KEY != key:
        if _EXEC is not None:
            _EXEC.shutdown(wait=False, cancel_futures=True)
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        _EXEC = ProcessPoolExecutor(
            max_workers=cfg["workers"], mp_context=mp.get_context("spawn"),
            initializer=_worker_init, initargs=(float(gamma), cfg["threads"]))
        _EXEC_KEY = key
    return _EXEC


def shutdown():
    global _EXEC, _EXEC_KEY
    if _EXEC is not None:
        _EXEC.shutdown(wait=True, cancel_futures=True)
        _EXEC, _EXEC_KEY = None, None


atexit.register(shutdown)


def _slice(x, i):
    """Slice a per-lane array, or a (nested) list of them; None stays None."""
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return [_slice(e, i) for e in x]
    return np.asarray(x)[i]


def _alloc_like(x, n):
    if isinstance(x, (list, tuple)):
        return [_alloc_like(e, n) for e in x]
    x = np.asarray(x)
    return np.empty((n,) + x.shape[1:], dtype=x.dtype)


def _fill(dst, src, i):
    if isinstance(dst, list):
        for d, s in zip(dst, src):
            _fill(d, s, i)
    else:
        dst[i] = src


def _merge_diag(diags, weights):
    """Counts add; the summary statistics are weighted by lane count; the
    iteration histogram adds elementwise; a max is a max."""
    out = {}
    wsum = float(sum(weights)) or 1.0
    for k in diags[0]:
        vals = [d.get(k) for d in diags]
        v0 = vals[0]
        if isinstance(v0, bool):
            out[k] = all(bool(v) for v in vals)
        elif isinstance(v0, (int, np.integer)):
            if k.endswith("_max"):
                out[k] = int(max(int(v) for v in vals))
            else:
                out[k] = int(sum(int(v) for v in vals))
        elif isinstance(v0, (float, np.floating)):
            out[k] = float(sum(float(v) * w for v, w in zip(vals, weights)) / wsum)
        elif isinstance(v0, list):
            L = max(len(v) for v in vals)
            acc = np.zeros(L)
            for v in vals:
                acc[:len(v)] += np.asarray(v, dtype=float)
            out[k] = acc.astype(int).tolist() if all(
                float(e).is_integer() for v in vals for e in v) else acc.tolist()
        else:
            out[k] = v0
    if "n_total" in out and out["n_total"]:
        out["frac_unconverged"] = 1.0 - out.get("n_converged", 0) / out["n_total"]
    return out


# Rescue-dict entries whose lane axis is not the first: (key -> axis).
# `planar4_b` returns the two slow-wave kinds as (2, n).
_LANE_AXIS = {"kinds": 1}


def _slice_rescue(x, i, key=None):
    """`_slice` for a rescue input: a (2, n) array is cut along its lane
    axis."""
    ax = _LANE_AXIS.get(key, 0)
    if x is None or isinstance(x, (list, tuple)) or ax == 0:
        return _slice(x, i)
    return np.take(np.asarray(x), i, axis=ax)


def _merge_lane_dicts(parts, idxs, n):
    """Scatter the chunks' per-lane results back into full-length arrays.

    A key whose value is None in EVERY chunk stays None.  A key that is None
    in some chunks (a rung that accepted nothing there returns None for its
    ``star``, ``region``, ``VsLv``, ``VsRv``) is filled from a chunk that
    has it, with zeros where it was None -- `exact_flux` reads those entries
    only under the ``good`` mask, which is False there.  Entries whose lane
    axis is not the first (`_LANE_AXIS`) are scattered along that axis.
    """
    out = {}
    for k in parts[0].keys():
        vals = [p[k] for p in parts]
        have = [v for v in vals if v is not None]
        if not have:
            out[k] = None
            continue
        ax = _LANE_AXIS.get(k, 0)
        if ax == 0:
            dst = _zeros_like(have[0], n)
            for v, i in zip(vals, idxs):
                if v is not None:
                    _fill(dst, v, i)
        else:
            v0 = np.asarray(have[0])
            shape = list(v0.shape); shape[ax] = n
            dst = np.zeros(shape, dtype=v0.dtype)
            for v, i in zip(vals, idxs):
                if v is not None:
                    idx = [slice(None)] * v0.ndim; idx[ax] = i
                    dst[tuple(idx)] = v
        out[k] = dst
    return out


def _zeros_like(x, n):
    if isinstance(x, (list, tuple)):
        return [_zeros_like(e, n) for e in x]
    x = np.asarray(x)
    return np.zeros((n,) + x.shape[1:], dtype=x.dtype)


def map_lanes(name, gamma, sL, sR, Bn, **kw):
    """Run the rescue rung `exact_flux.<name>` over the pool, chunked by
    lane, and merge; the in-process call is the fallback below the pool's
    lane minimum.  The rungs are batch-independent (each lane's Newton sees
    only its own data), so a chunk solves to the same numbers as the whole:
    `tests/test_exact_pool.py` asserts it to the bit."""
    cfg = config()
    n = int(np.asarray(Bn).shape[0])
    if cfg is None or n < cfg["min_lanes"]:
        from src.physics import exact_flux as EF
        return getattr(EF, name)(EF._batched_parts(gamma), gamma, sL, sR, Bn, **kw)
    nch = max(1, min(cfg["chunks"], n))
    ex = _executor(gamma, cfg)
    owner = np.arange(n) % nch
    kw = dict(kw, gamma=float(gamma))
    futs, idxs = [], []
    for c in range(nch):
        i = np.flatnonzero(owner == c)
        if i.size == 0:
            continue
        kwc = {}
        for k, v in kw.items():
            if isinstance(v, dict):                 # a rescue dict (the ladder's r5)
                kwc[k] = {kk: _slice_rescue(vv, i, kk) for kk, vv in v.items()}
            else:
                kwc[k] = v
        futs.append(ex.submit(_run_rescue, (name, _slice(sL, i), _slice(sR, i),
                                            _slice(Bn, i), kwc)))
        idxs.append(i)
    parts = [f.result() for f in futs]
    return _merge_lane_dicts(parts, idxs, n)


def solve_and_ray(gamma, subL, subR, subBn, seed6, keys, seeds_extra, **kw):
    """The pooled twin of `exact_flux._solve_and_ray`, same return."""
    cfg = config()
    n = int(np.asarray(subBn).shape[0])
    nch = max(1, min(cfg["chunks"], n))
    ex = _executor(gamma, cfg)
    owner = np.arange(n) % nch
    kw = dict(kw, gamma=float(gamma))
    futs, idxs = [], []
    for c in range(nch):
        i = np.flatnonzero(owner == c)
        if i.size == 0:
            continue
        payload = (_slice(subL, i), _slice(subR, i), _slice(subBn, i),
                   _slice(seed6, i), _slice(keys, i), _slice(seeds_extra, i),
                   dict(kw))
        futs.append(ex.submit(_run_chunk, payload))
        idxs.append(i)
    parts = [f.result() for f in futs]         # raises the worker's error

    res0, d0, star0, region0, ray_ok0, _ = parts[0]
    res = {k: _alloc_like(v, n) for k, v in res0.items()}
    star = _alloc_like(star0, n)
    region = _alloc_like(region0, n)
    ray_ok = _alloc_like(ray_ok0, n)
    n_fan = 0
    for i, (r, d, st, reg, ok, nf) in zip(idxs, parts):
        for k in res:
            _fill(res[k], r[k], i)
        _fill(star, st, i)
        _fill(region, reg, i)
        _fill(ray_ok, ok, i)
        n_fan += int(nf)
    diag = _merge_diag([p[1] for p in parts], [len(i) for i in idxs])
    diag["pool_workers"] = cfg["workers"]
    diag["pool_chunks"] = len(parts)
    return res, diag, star, region, ray_ok, n_fan
