"""The planar warm-start network in the late rung (``RMHD_COPLANAR7_NET``,
coplanar7_b + planar_net).

* Off by default.
* Its features do not depend on the frame: the two states turned together
  about the normal give the same features and the same seed in the late
  rung's frame.
* Where the network's output is unusable, every lane falls back to
  production's seed: the answers, stages and branches are production's, to the
  bit.
* With the trained network (data/ml_planar_v2a.pt; skipped without it) the two
  ledger lanes of test_coplanar7 are still answered with the regular answer
  (both rotations for 1013, the left one for 214). That is the same root as
  production's, to the Newton's accuracy.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
np.seterr(all="ignore")

from rmhd import paths                           # noqa: E402
from rmhd.batched import coplanar7_b as CP7      # noqa: E402
from rmhd.batched import planar_net as PN        # noqa: E402
from rmhd.batched.planar5_b import to_planar     # noqa: E402
from test_coplanar7 import GAMMA, LANES, _problem  # noqa: E402

pytest.importorskip("torch")
CKPT = "data/ml_planar_v2a.pt"
HAVE_CKPT = os.path.exists(paths.resolve(CKPT))


def _both():
    cat = lambda a, b: [np.concatenate([x, y]) for x, y in zip(a, b)]
    (L1, R1, B1, s1), (L2, R2, B2, s2) = _problem(1013), _problem(214)
    return cat(L1, L2), cat(R1, R2), np.concatenate([B1, B2]), cat(s1, s2)


def _turn(U, th):
    """Both tangential vectors (v, B) of a state turned by th about the normal."""
    V = [c.copy() for c in U]
    c, s = np.cos(th), np.sin(th)
    V[3], V[4] = c * U[3] - s * U[4], s * U[3] + c * U[4]
    V[5], V[6] = c * U[5] - s * U[6], s * U[5] + c * U[6]
    return V


def test_off_by_default():
    env = {k: v for k, v in os.environ.items() if k != "RMHD_COPLANAR7_NET"}
    out = subprocess.check_output(
        [sys.executable, "-c", "import rmhd.batched.coplanar7_b as C; print(repr(C.NET), C.NET_FALLBACK)"],
        env=env, text=True).strip()
    assert out == "'' all"


def test_features_are_frame_free():
    L, R, Bn, _ = _both()
    X0 = PN.features(L, R, Bn, GAMMA)[0]
    for th in (0.3, 2.0, -1.1):
        X1 = PN.features(_turn(L, th), _turn(R, th), Bn, GAMMA)[0]
        assert np.allclose(X1, X0, rtol=0, atol=1e-12), th


@pytest.mark.skipif(not HAVE_CKPT, reason="no planar network checkpoint")
def test_seed_turns_with_the_frame():
    L, R, Bn, _ = _both()
    p0 = PN.predict(paths.resolve(CKPT), L, R, Bn, GAMMA, to_planar(L, R, Bn)[2])
    for th in (0.3, 2.0):
        Lt, Rt = _turn(L, th), _turn(R, th)
        p1 = PN.predict(paths.resolve(CKPT), Lt, Rt, Bn, GAMMA, to_planar(Lt, Rt, Bn)[2])
        for key in ("u_S", "u_P"):
            for a, b in zip(p0[key], p1[key]):
                assert np.allclose(a, b, rtol=1e-5, atol=1e-6), (key, th)
        assert np.array_equal(p0["rot"], p1["rot"]) and np.array_equal(p0["pressure"], p1["pressure"])


def test_unusable_network_output_is_production(monkeypatch):
    L, R, Bn, s6 = _both()
    S = CP7.make_solver(GAMMA)
    monkeypatch.setattr(CP7, "NET", "")
    ref = S["solve"](L, R, Bn, s6, accuracy=1e-8)

    def junk(path, left, right, Bn_, gamma, alpha):
        n = Bn_.shape[0]
        nan = lambda: np.full(n, np.nan)
        return dict(u_S=[nan() for _ in range(4)], u_P=[nan() for _ in range(4)],
                    rot=np.ones((2, n), bool), pressure=np.ones(n, bool), ok=np.zeros(n, bool))

    monkeypatch.setattr(CP7, "NET", "unused.pt")
    monkeypatch.setattr(PN, "predict", junk)
    got = S["solve"](L, R, Bn, s6, accuracy=1e-8)
    assert ref["converged"].all() and np.array_equal(got["converged"], ref["converged"])
    assert np.array_equal(got["stage"], ref["stage"])
    for key in ("kinds", "rot"):
        for a, b in zip(got[key], ref[key]):
            assert np.array_equal(a, b), key
    for a, b in zip(got["zones"], ref["zones"]):
        for x, y in zip(a, b):
            assert np.array_equal(x, y)


@pytest.mark.skipif(not HAVE_CKPT, reason="no planar network checkpoint")
@pytest.mark.parametrize("fallback", ["all", "first"])
def test_network_answers_the_ledger_lanes(monkeypatch, fallback):
    L, R, Bn, s6 = _both()
    S = CP7.make_solver(GAMMA)
    monkeypatch.setattr(CP7, "NET", "")
    ref = S["solve"](L, R, Bn, s6, accuracy=1e-8)
    monkeypatch.setattr(CP7, "NET", CKPT)
    monkeypatch.setattr(CP7, "NET_FALLBACK", fallback)
    before = dict(CP7.STATS)
    got = S["solve"](L, R, Bn, s6, accuracy=1e-8)
    assert CP7.STATS["calls"] == before["calls"] + 1 and CP7.STATS["solves"] > before["solves"]
    assert got["converged"].all()
    for i, lane in enumerate((1013, 214)):
        assert (bool(got["rot"][0][i]), bool(got["rot"][1][i])) == LANES[lane]["ROT"]
    for a, b in zip(got["zones"], ref["zones"]):
        for x, y in zip(a, b):
            assert np.allclose(x, y, rtol=1e-6, atol=1e-9)
    print("stages with the network:", got["stage"].tolist(), "production:", ref["stage"].tolist(),
          "iterations:", got["n_iter"].tolist(), ref["n_iter"].tolist())
