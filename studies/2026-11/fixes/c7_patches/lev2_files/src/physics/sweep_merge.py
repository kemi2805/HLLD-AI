"""One seven-wave pipeline call for the x and y sweeps of an RK stage
(``RMHD_MERGE_SWEEPS=1``; off by default).

Why: the exact flux's per-lane pipeline (the seven-wave Newton with its
retries, then the ray) costs a sweep about the same whatever its lane count
-- its wall is the slowest lane's iterations times the per-call overhead
(rmhd cost floor).  Measured on the 64^2 timing arms (2026-11-06), the
pipeline took ~7 s a call for ~470 lanes (4Q) and ~9.3 s for ~1,660 (OT).
Two sweeps a stage pay that twice; one call on both sweeps' lanes pays it
about once.

What is merged, and what is not.  Only the pipeline -- pure numpy/numba on
per-lane inputs, the part `exact_pool` already chunks freely because every
lane's answer is independent of what shares its batch.  Everything that runs
on a whole sweep stays per sweep and in its order: the reconstruction, the
HLLD fallback flux and star pressure (torch, over the full face array), the
gates, the network seeds (a matrix product over the sweep's lanes), the
rescue rungs, the flux assembly, the harvest records and coverage, and
``LAST_DIAG``.  So the run computes the same numbers in the same order; only
the pipeline's lanes travel together.

How: `ct_stage` runs each sweep in its own thread, ONE AT A TIME.  The x
sweep runs until its pipeline call (`pipeline`) or until it returns without
one (`skip`); then the y sweep runs to the same point; the last to arrive
runs the pipeline once on all lanes; then the x sweep finishes, the driver
marks it `done`, and the y sweep finishes.  No two threads ever compute at
the same time, so nothing here needs to be thread-safe beyond the hand-over.
``LAST_DIAG`` is saved when a sweep arrives and restored when it resumes, so
each sweep's diagnostics read as they would without the merge.  The
pipeline's own diagnostics (``d``) cover both sweeps; ``n_fan`` is recounted
per sweep from its lanes.
"""
from __future__ import annotations

import os
import threading

import numpy as np


def wanted() -> bool:
    return os.environ.get("RMHD_MERGE_SWEEPS", "0") not in ("0", "", "off")


def _cat(parts):
    """Concatenate per-lane values -- arrays, or (nested) lists of them."""
    p0 = parts[0]
    if p0 is None:
        if any(p is not None for p in parts):
            raise ValueError("sweep merge: a per-lane value is None in one sweep only")
        return None
    if isinstance(p0, (list, tuple)):
        return [_cat([p[i] for p in parts]) for i in range(len(p0))]
    return np.concatenate([np.asarray(p) for p in parts])


class Rendezvous:
    """The hand-over between the sweeps of one stage (module doc)."""

    def __init__(self, order):
        self.order = list(order)
        self.cv = threading.Condition()
        self.requests = {}
        self.arrived = set()
        self.results = None
        self.turn = 0
        self.saved = {}
        self.error = None

    # ── called from the sweeps' threads ─────────────────────────────────
    def pipeline(self, key, P, gamma, subL, subR, subBn, seed6, keys, seeds_extra, **kw):
        from .hlld import LAST_DIAG
        with self.cv:
            self.requests[key] = (P, gamma, subL, subR, subBn, seed6, keys, seeds_extra, kw)
            self._arrive(key, LAST_DIAG)
            self._resume(key, LAST_DIAG)
            return self.results.pop(key)

    def skip(self, key):
        """A sweep with no pipeline to run (no attempted lane) -- it still
        waits for its turn before writing its diagnostics and coverage."""
        from .hlld import LAST_DIAG
        with self.cv:
            self._arrive(key, LAST_DIAG)
            self._resume(key, LAST_DIAG)

    # ── called from the driver ───────────────────────────────────────────
    def arrived_or_failed(self, key):
        with self.cv:
            self._wait(lambda: key in self.arrived or self.turn > self.order.index(key),
                       allow_error=True)

    def done(self, key):
        with self.cv:
            if self.order[self.turn] != key:
                raise RuntimeError("sweep merge: %r finished out of turn" % (key,))
            self.turn += 1
            self.cv.notify_all()

    def abort(self, exc):
        with self.cv:
            if self.error is None:
                self.error = exc
            self.cv.notify_all()

    # ── internals (under the lock) ──────────────────────────────────────
    def _wait(self, pred, allow_error=False):
        while not pred():
            if self.error is not None:
                if allow_error:
                    return
                raise RuntimeError("sweep merge aborted") from self.error
            self.cv.wait()

    def _arrive(self, key, LAST_DIAG):
        self.saved[key] = dict(LAST_DIAG)
        self.arrived.add(key)
        self.cv.notify_all()
        self._wait(lambda: len(self.arrived) == len(self.order))
        if self.results is None:
            self._run()
            self.cv.notify_all()

    def _resume(self, key, LAST_DIAG):
        self._wait(lambda: self.order[self.turn] == key)
        LAST_DIAG.clear()
        LAST_DIAG.update(self.saved.pop(key))

    def _run(self):
        from rmhd.batched import classify as C
        from . import exact_flux as EF
        from .exact_pool import _slice
        ks = [k for k in self.order if k in self.requests]
        self.results = {}
        if not ks:
            return
        reqs = [self.requests[k] for k in ks]
        P, gamma, kw = reqs[0][0], reqs[0][1], reqs[0][8]
        for r in reqs[1:]:
            if r[1] != gamma or r[8] != kw:
                raise ValueError("sweep merge: the sweeps' solver settings differ")
        n = [int(np.asarray(r[4]).shape[0]) for r in reqs]
        args = [_cat([r[j] for r in reqs]) for j in range(2, 8)]
        res, d, star, region, ray_ok, _ = EF._run_pipeline(P, gamma, *args, **kw)
        RAY = P["ray"]
        off = 0
        for k, nk in zip(ks, n):
            i = np.arange(off, off + nk)
            off += nk
            rk = {name: _slice(v, i) for name, v in res.items()}
            regk = _slice(region, i)
            conv = np.asarray(rk["converged"], dtype=bool)
            seven = conv & np.isin(rk["cls"], (C.FULL7, C.COPLANAR))
            n_fan = int((seven & np.isin(regk, RAY.FAN_REGIONS)).sum()
                        + (conv & ~seven & np.isin(regk, RAY.FAN_REGIONS_3)).sum())
            self.results[k] = (rk, dict(d), _slice(star, i), regk, _slice(ray_ok, i), n_fan)
