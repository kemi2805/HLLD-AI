"""RMHD_PLANAR_NJIT: the compiled twins of the planar solvers' numpy
root-finders answer every lane bit for bit as numpy does.

``check`` runs both on every call and counts the lanes that differ
(planar_njit.CHECK).  The lanes here are the planar seven-wave fixtures of
test_coplanar7 (ledger lanes 1013 and 214 of the 4Q 128^2 run), solved with
every branch of every stage at once: the pressure-steered branches build
slow shocks to a total pressure (planar_njit.slow_shock_p), 760 lanes in 195
calls.  The other two kernels (the slow shock's retry, the fast shock's edge
scan) are not reached by these lanes; they were checked on hard faces of the
three 64^2 runs (2026-11-07: retry 6,485 lanes, edge scan 282, slow_shock_p
7,667 -- none differed; fixes/levers/profile_rung.py with check).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
np.seterr(all="ignore")

from rmhd.batched import coplanar7_b as CP7      # noqa: E402
from rmhd.batched import planar_njit as PNJ      # noqa: E402
from test_coplanar7 import GAMMA, LANES, _problem  # noqa: E402

pytest.importorskip("numba")


@pytest.fixture(autouse=True)
def _production_seed(monkeypatch):
    # every branch of every stage from production's seed, as documented above
    # (the planar network is on by default since 2026-11-08)
    monkeypatch.setattr(CP7, "NET", "")


@pytest.fixture
def check_mode(monkeypatch):
    monkeypatch.setenv("RMHD_PLANAR_NJIT", "check")
    PNJ.CHECK.clear()
    yield
    PNJ.CHECK.clear()


def _solve_both_lanes():
    S = CP7.make_solver(GAMMA)
    cat = lambda a, b: [np.concatenate([x, y]) for x, y in zip(a, b)]
    (L1, R1, B1, s1), (L2, R2, B2, s2) = _problem(1013), _problem(214)
    return S["solve"](cat(L1, L2), cat(R1, R2), np.concatenate([B1, B2]), cat(s1, s2),
                      accuracy=1e-8)


def test_kernels_agree_with_numpy_bit_for_bit(check_mode, monkeypatch):
    # every branch of every stage in one solve (RMHD_COPLANAR7_CALLS=1, the
    # same answers): the staged solve answers both lanes at stage 2 and never
    # reaches the pressure-steered stage
    monkeypatch.setattr(CP7, "CALLS", 1)
    r = _solve_both_lanes()
    assert r["converged"].all()
    assert "slow_shock_p" in PNJ.CHECK
    for name, (calls, lanes, differ) in PNJ.CHECK.items():
        assert calls > 0 and lanes > 0, name
        assert differ == 0, "%s: %d of %d lanes differ" % (name, differ, lanes)


def test_njit_mode_gives_the_numpy_answers(monkeypatch):
    monkeypatch.setenv("RMHD_PLANAR_NJIT", "0")
    ref = _solve_both_lanes()
    monkeypatch.setenv("RMHD_PLANAR_NJIT", "1")
    got = _solve_both_lanes()
    for key in ("converged", "stage", "n_iter"):
        assert np.array_equal(got[key], ref[key]), key
    for a, b in zip(got["zones"], ref["zones"]):
        for x, y in zip(a, b):
            assert np.array_equal(x, y)
    for key in ("VsLv", "VsRv", "unk6"):
        for x, y in zip(got[key], ref[key]):
            assert np.array_equal(x, y), key


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("RMHD_PLANAR_NJIT", raising=False)
    assert PNJ.mode() == "numpy" and not PNJ.wanted()
