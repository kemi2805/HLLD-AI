"""The planar warm-start network in the late rung (``RMHD_COPLANAR7_NET``,
coplanar7_b + planar_net).

* On by default since 2026-11-08 (RMHD_COPLANAR7_NET=0 switches it off), and
  so is the preference for regular answers (RMHD_COPLANAR7_REGULAR).
* Its features do not depend on the frame: the two states turned together
  about the normal give the same features and the same seed in the late
  rung's frame.
* Where the network's output is unusable, every lane falls back to
  production's seed: the answers, stages and branches are production's, to the
  bit.
* An answer with an intermediate slow shock gives way to a regular one from a
  later branch (two rotor lanes), and is kept where no branch is regular.
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


# rotor 64^2 harvest run, held-out sweeps 1623 / 1627 (replay faces 47483 /
# 47634, 2026-11-08), with production's seed (rotor_ft candidate 0, clamped):
# production's late rung answers both at stage 2 with no rotation and an
# INTERMEDIATE left slow shock (entropy rising) -- stage 2's first branch.  A
# left Alfven rotation and a regular slow shock answers them too: the next
# branch of the same stage (and the network's stage 3)
ROTOR = {
    47483: dict(
        UL=[1.0246832897616178, 0.07900824059538968, 0.2188358465534758, -0.24585207578620863,
            0.0, -0.12315781878292631, 0.0, 0.3082093331372757],
        UR=[1.6998616776752962, 0.18279124678075587, 0.04468138531752245, -0.13412842905729722,
            0.0, 0.0005728870439151296, 0.0, 0.3082093331372757],
        S6=[-1.9813844814381398, -5.158661348346938, 3.189751310191836, -1.935863341339568,
            0.020234238355367112, -1.7989216841023186]),
    47634: dict(
        UL=[1.0261049825308084, 0.0792209373012045, 0.21808340514678579, -0.24512460571865055,
            0.0, -0.1231127386263875, 0.0, 0.30827118388023117],
        UR=[1.6988462075187445, 0.1827835421210664, 0.04447415565428476, -0.1340226774519641,
            0.0, 0.0005575894895203659, 0.0, 0.30827118388023117],
        S6=[-1.9807080075439614, -5.150013985499581, 3.189498852681392, -1.935346250265519,
            0.021558021521287576, -1.7889013716826496]),
}


def _rotor():
    f = sorted(ROTOR)
    L = [np.array([ROTOR[q]["UL"][j] for q in f]) for j in range(7)]
    R = [np.array([ROTOR[q]["UR"][j] for q in f]) for j in range(7)]
    s6 = [np.array([ROTOR[q]["S6"][j] for q in f]) for j in range(6)]
    return L, R, np.array([ROTOR[q]["UL"][7] for q in f]), s6


def test_defaults():
    env = {k: v for k, v in os.environ.items() if not k.startswith("RMHD_COPLANAR7_")}
    code = "import rmhd.batched.coplanar7_b as C; print(repr(C.NET), C.NET_FALLBACK, C.REGULAR)"
    run = lambda e: subprocess.check_output([sys.executable, "-c", code], env=e, text=True).strip()
    assert run(env) == "'data/ml_planar_v2a.pt' all True"
    assert run(dict(env, RMHD_COPLANAR7_NET="0", RMHD_COPLANAR7_REGULAR="0")) == "'' all False"


def test_regular_answer_preferred(monkeypatch):
    L, R, Bn, s6 = _rotor()
    S = CP7.make_solver(GAMMA)
    monkeypatch.setattr(CP7, "NET", "")
    monkeypatch.setattr(CP7, "REGULAR", False)
    old = S["solve"](L, R, Bn, s6, accuracy=1e-8)
    assert old["converged"].all() and old["intermediate"].all()
    assert (old["stage"] == 2).all()
    monkeypatch.setattr(CP7, "REGULAR", True)
    before = dict(CP7.STATS)
    new = S["solve"](L, R, Bn, s6, accuracy=1e-8)
    assert new["converged"].all() and not new["intermediate"].any() and new["replaced"].all()
    assert (new["stage"] == 2).all()                # the same stage, its next branch
    assert CP7.STATS["intermediate_replaced"] - before["intermediate_replaced"] == 2
    assert CP7.STATS["intermediate_kept"] == before["intermediate_kept"]
    # the regular answer: a rotation in front of the left slow wave, nothing on the right
    assert new["rot"][0].all() and not new["rot"][1].any()
    # and its left slow shock is outside the Alfven speeds of both of its sides
    al = CP7.P4.make_solver(GAMMA)["alfven_speed"]
    R2, R3, R4, R5, R6, R7 = new["zones"]
    V = new["VsLv"][2]
    assert ((V - al(R3, Bn, "L")) * (V - al(R4, Bn, "L")) > 0).all()
    print("regular answers at stages", new["stage"].tolist(), "rotations",
          [x.tolist() for x in new["rot"]], "kinds", [x.tolist() for x in new["kinds"]])


def test_intermediate_only_lanes_keep_their_first_answer(monkeypatch):
    """With every accepted branch called intermediate, the preference finds
    no regular answer and each lane keeps its first accepted one: the answers
    without the preference, to the bit."""
    L, R, Bn, s6 = _both()
    S = CP7.make_solver(GAMMA)
    monkeypatch.setattr(CP7, "NET", "")
    monkeypatch.setattr(CP7, "REGULAR", False)
    ref = S["solve"](L, R, Bn, s6, accuracy=1e-8)
    monkeypatch.setattr(CP7, "REGULAR", True)
    monkeypatch.setattr(CP7, "intermediate_shocks", lambda r, Bn_, al: np.ones(Bn_.shape[0], bool))
    before = dict(CP7.STATS)
    got = S["solve"](L, R, Bn, s6, accuracy=1e-8)
    assert ref["converged"].all() and np.array_equal(got["converged"], ref["converged"])
    assert got["intermediate"].all() and np.array_equal(got["stage"], ref["stage"])
    assert not got["replaced"].any()
    assert CP7.STATS["intermediate_kept"] - before["intermediate_kept"] == 2
    for key in ("kinds", "rot"):
        for a, b in zip(got[key], ref[key]):
            assert np.array_equal(a, b), key
    for a, b in zip(got["zones"], ref["zones"]):
        for x, y in zip(a, b):
            assert np.array_equal(x, y)


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


@pytest.mark.skipif(not HAVE_CKPT, reason="no planar network checkpoint")
def test_prediction_does_not_depend_on_the_batch():
    """A face's prediction, alone or among others in any order, to the bit
    (a face and its mirror image sit in different sweeps: a batch-dependent
    seed broke the rotor's symmetry at 128^2, 2026-11-08)."""
    L, R, Bn, _ = _both()
    Lr, Rr, Br, _ = _rotor()
    cat = lambda a, b: [np.concatenate([x, y]) for x, y in zip(a, b)]
    path = paths.resolve(CKPT)
    LL, RR, BB = cat(cat(L, Lr), L), cat(cat(R, Rr), R), np.concatenate([Bn, Br, Bn])
    many = PN.predict(path, LL, RR, BB, GAMMA, to_planar(LL, RR, BB)[2])
    for i in range(BB.size):
        one = PN.predict(path, [c[i:i + 1] for c in LL], [c[i:i + 1] for c in RR], BB[i:i + 1], GAMMA,
                         to_planar([c[i:i + 1] for c in LL], [c[i:i + 1] for c in RR], BB[i:i + 1])[2])
        for key in ("u_S", "u_P"):
            for a, b in zip(one[key], many[key]):
                assert a[0] == b[i], (key, i)
        assert np.array_equal(one["rot"][:, 0], many["rot"][:, i]) and one["pressure"][0] == many["pressure"][i]


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
