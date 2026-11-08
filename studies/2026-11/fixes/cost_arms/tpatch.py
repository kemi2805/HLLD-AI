"""Wall per solver stage of a 2D run, from the main process: the seven-wave
pipeline (exact_flux._run_pipeline) and each rescue rung
(exact_pool.map_lanes, by rung name), with calls and lanes.  Written to
$RUNG_TIMES every 60 s and at exit; imported before scripts/run_2d.py."""
import atexit
import collections
import json
import os
import time

import numpy as np
from src.physics import exact_flux as EF
from src.physics import exact_pool as XP

T = collections.defaultdict(float)
C = collections.defaultdict(int)
L = collections.defaultdict(int)
OUT = os.environ["RUNG_TIMES"]
_last = [time.time()]


def _dump():
    with open(OUT + ".tmp", "w") as f:
        json.dump({k: dict(wall=round(T[k], 3), calls=C[k], lanes=L[k]) for k in sorted(T)}, f, indent=1)
    os.replace(OUT + ".tmp", OUT)


def _tick(name, t0, n):
    T[name] += time.perf_counter() - t0
    C[name] += 1
    L[name] += n
    if time.time() - _last[0] > 60:
        _last[0] = time.time()
        _dump()


_map_lanes = XP.map_lanes


def map_lanes(name, gamma, sL, sR, Bn, **kw):
    t0 = time.perf_counter()
    try:
        return _map_lanes(name, gamma, sL, sR, Bn, **kw)
    finally:
        _tick(name, t0, int(np.asarray(Bn).shape[0]))


_pipeline = EF._run_pipeline


def _run_pipeline(P, gamma, subL, subR, subBn, *a, **kw):
    t0 = time.perf_counter()
    try:
        return _pipeline(P, gamma, subL, subR, subBn, *a, **kw)
    finally:
        _tick("_run_pipeline", t0, int(np.asarray(subBn).shape[0]))


XP.map_lanes = map_lanes
EF._run_pipeline = _run_pipeline
atexit.register(_dump)
