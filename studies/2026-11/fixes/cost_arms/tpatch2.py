"""tpatch.py plus per-lane statistics of the planar seven-wave rung and of
every in-process four-unknown solve, for one timing arm:

* the wall per stage (tpatch.py) -> $RUNG_TIMES;
* per lane offered the late rung (LAST_DIAG of the inner, canonical call):
  its verdict, stage, iterations and rotations, and which rung answered it;
* per call of a planar4 solve made in THIS process (the rungs run in process
  below the pool's lane minimum, 8): the caller, lanes, wall, iteration cap,
  iterations of the slowest lane, lanes converged.
Both -> $RUNG_STATS (npz), every 60 s and at exit."""
import collections
import os
import sys
import time

import numpy as np
import tpatch  # noqa: F401  (the stage timers)
from rmhd.batched import planar4_b as P4
from src.physics import exact_flux as EF
from src.physics.hlld import LAST_DIAG

OUT = os.environ["RUNG_STATS"]
LANE = collections.defaultdict(list)
CALL = collections.defaultdict(list)
_last = [time.time()]


def _dump():
    d = {}
    for k, v in LANE.items():
        d["lane_" + k] = np.concatenate(v) if v else np.zeros(0)
    for k, v in CALL.items():
        d["call_" + k] = np.array(v)
    np.savez_compressed(OUT + ".tmp.npz", **d)
    os.replace(OUT + ".tmp.npz", OUT)


def _maybe_dump():
    if time.time() - _last[0] > 60:
        _last[0] = time.time()
        _dump()


_make_solver = P4.make_solver


def make_solver(gamma):
    S = _make_solver(gamma)
    inner = S["solve"]

    def solve(*a, **kw):
        f = sys._getframe(1)
        who = "%s.%s" % (f.f_globals.get("__name__", "?").rsplit(".", 1)[-1], f.f_code.co_name)
        t0 = time.perf_counter()
        r = inner(*a, **kw)
        n = int(np.asarray(a[2]).shape[0]) if len(a) > 2 else -1
        CALL["who"].append(who)
        CALL["lanes"].append(n)
        CALL["wall"].append(time.perf_counter() - t0)
        CALL["max_iter"].append(int(kw.get("max_iter", 40)))
        it = np.asarray(r.get("n_iter", [0]))
        CALL["iter_max"].append(int(it.max()) if it.size else 0)
        CALL["converged"].append(int(np.asarray(r["converged"]).sum()))
        _maybe_dump()
        return r

    S = dict(S, solve=solve)
    return S


P4.make_solver = make_solver
_efb = EF.exact_flux_batched


def exact_flux_batched(*a, **kw):
    out = _efb(*a, **kw)
    if kw.get("_record_idir") is not None or not EF._CANONICAL:
        d = LAST_DIAG
        if "coplanar7_late_reason" in d:
            rs = np.asarray(d["coplanar7_late_reason"])
            off = rs != 8
            if off.any():
                LANE["c7_reason"].append(rs[off])
                LANE["c7_stage"].append(np.asarray(d["coplanar7_late_stage"])[off])
                LANE["c7_n_iter"].append(np.asarray(d["coplanar7_late_n_iter"])[off])
                rot = np.asarray(d["coplanar7_late_rot"])
                LANE["c7_rot"].append(rot[0][off].astype(np.int8) + 2 * rot[1][off].astype(np.int8))
                m = lambda k: np.asarray(d[k], bool)[off] if k in d else np.zeros(off.sum(), bool)
                ex = d["exact_mask"]
                ex = np.asarray(ex.numpy() if hasattr(ex, "numpy") else ex)
                sel = np.asarray(d["sel"])
                exs = (ex[sel] if ex.size != sel.size else ex)[off]
                LANE["answered_by"].append(np.where(m("coplanar7_late_mask"), 4,
                                           np.where(m("planar4_mask"), 3, np.where(exs, 9, 0))).astype(np.int8))
                LANE["n_offered_in_call"].append(np.full(off.sum(), off.sum()))
        _maybe_dump()
    return out


EF.exact_flux_batched = exact_flux_batched
import atexit  # noqa: E402
atexit.register(_dump)
