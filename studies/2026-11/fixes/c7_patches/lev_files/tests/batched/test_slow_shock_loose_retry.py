"""The loose-fallback retry (slow_shock_b.LOOSE_RETRY, RMHD_SLOWSHOCK_LOOSE_RETRY).

Found 2026-11-04 on the failure ledger of the four-quadrant 128^2 run (ft3):
in a band of contact fields around the true root the left slow-shock ladder
converges from no seed and hands back its loose fallback -- the best grid
cell, here Vs = -0.0238 against an Alfven bound of 0.0055 and P_tot 14% low
-- as an ANSWER.  The planar residual then has a hole exactly at its zero,
and the retry, which runs on failed lanes only, never sees the lane.  The
fixture is ledger lane 62's state behind the left fast wave at its regular
root (two slow shocks, the answer the tube converges to).  With the switch
on the retry must build the root; off, the old answer must stand; a lane the
ladder answers with a real root is untouched either way.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
np.seterr(all="ignore")

from rmhd.batched import slow_shock_b as SB       # noqa: E402
from rmhd.batched import slow_shock_njit as SN    # noqa: E402
from rmhd.batched.jump import rh_resid            # noqa: E402

GAMMA = 1.6666666666666667
BN = 0.35355339059408397
AHEAD = [0.10808246616636694, 1.353872087582769, 0.010694229635442446,
         0.9888757594660211, 1.2110313324626748e-16, 0.36440804257436593,
         5.458369761026578e-20]
BT = 0.3188758
BEHIND = [0.10887063652547653, 1.3538823824088637, 0.010787854654000489,
          0.9882972136818902, 1.2110233846168708e-16, 0.3188758, 0.0]
VS = 0.005546194681897403

KERNELS = ["numpy"] + (["njit"] if SN.HAVE_NUMBA else [])


def _fixture():
    ahead = [np.array([c]) for c in AHEAD]
    return np.array([BT]), np.zeros(1), ahead, np.array([BN])


@pytest.mark.parametrize("kernel", KERNELS)
def test_loose_retry_builds_the_root(monkeypatch, kernel):
    monkeypatch.setenv("RMHD_SLOWSHOCK", kernel)
    monkeypatch.setattr(SB, "SLOW_RETRY", True)
    Byt, Bzt, ahead, Bn = _fixture()
    solve = SB.make_solver(GAMMA)
    monkeypatch.setattr(SB, "LOOSE_RETRY", False)
    V0, b0, e0 = solve(Byt, Bzt, ahead, "LS", Bn, accuracy=1e-10)
    assert not e0[0]
    assert rh_resid(ahead, b0, V0, Bn, GAMMA)[0] > 1e-3, \
        "the ladder now finds this root: the fixture no longer tests the loose retry"
    monkeypatch.setattr(SB, "LOOSE_RETRY", True)
    V, b, e = solve(Byt, Bzt, ahead, "LS", Bn, accuracy=1e-10)
    assert not e[0]
    assert abs(V[0] - VS) < 1e-9
    scale = np.abs(BEHIND).max()
    assert max(abs(b[j][0] - BEHIND[j]) for j in range(7)) < 1e-9 * scale
    assert rh_resid(ahead, b, V, Bn, GAMMA)[0] <= SB.RETRY_RH


def _problems(seed, N):
    """The hard set of test_slow_shock_njit (targets 0.02x..1.6x, B_n to 0.05)."""
    rng = np.random.default_rng(seed)
    rho = 10 ** rng.uniform(-0.5, 1.2, N); p = 10 ** rng.uniform(-2.5, 0.9, N)
    v = rng.uniform(-0.7, 0.7, (3, N)); v *= 0.9 / np.maximum(1, np.linalg.norm(v, axis=0) / 0.9)
    By, Bz, Bn = rng.uniform(-2, 2, N), rng.uniform(-2, 2, N), rng.uniform(0.05, 0.6, N)
    fac = rng.uniform(0.02, 1.6, N)
    B2 = Bn ** 2 + By ** 2 + Bz ** 2
    v2 = (v ** 2).sum(0)
    eta = Bn * v[0] + By * v[1] + Bz * v[2]
    P = p + 0.5 * (B2 * (1 - v2) + eta ** 2)
    return [rho, P, v[0], v[1], v[2], By, Bz], By * fac, Bz * fac, Bn


@pytest.mark.parametrize("switch", ["LS", "RS"])
def test_loose_retry_touches_only_loose_answers(monkeypatch, switch):
    """On or off, a lane answered with a root or refused keeps its bits; what
    the switch changes was not a root before and is one after."""
    monkeypatch.setattr(SB, "SLOW_RETRY", True)
    ahead, Byt, Bzt, Bn = _problems(99, 300)
    if switch == "LS":                         # the fixture rides along
        Byt = np.append(Byt, BT)
        Bzt = np.append(Bzt, 0.0)
        ahead = [np.append(c, a) for c, a in zip(ahead, AHEAD)]
        Bn = np.append(Bn, BN)
    solve = SB.make_solver(GAMMA)
    monkeypatch.setattr(SB, "LOOSE_RETRY", False)
    V0, b0, e0 = solve(Byt, Bzt, ahead, switch, Bn, accuracy=1e-8)
    monkeypatch.setattr(SB, "LOOSE_RETRY", True)
    V1, b1, e1 = solve(Byt, Bzt, ahead, switch, Bn, accuracy=1e-8)
    assert np.array_equal(e0, e1)
    moved = (V1 != V0) | np.any([b1[j] != b0[j] for j in range(7)], axis=0)
    rh0 = rh_resid(ahead, b0, V0, Bn, GAMMA)
    rh1 = rh_resid(ahead, b1, V1, Bn, GAMMA)
    assert (rh0[moved] > SB.RETRY_RH).all(), "a root was replaced"
    assert (rh1[moved] <= SB.RETRY_RH).all(), "a loose cell was replaced by a non-root"
    if switch == "LS":
        assert moved[-1], "the fixture was not retried"


@pytest.mark.skipif(not SN.HAVE_NUMBA, reason="numba not installed")
@pytest.mark.parametrize("switch", ["LS", "RS"])
def test_loose_retry_njit_matches_numpy(monkeypatch, switch):
    monkeypatch.setattr(SB, "SLOW_RETRY", True)
    monkeypatch.setattr(SB, "LOOSE_RETRY", True)
    ahead, Byt, Bzt, Bn = _problems(99, 300)
    out = {}
    for kernel in ("numpy", "njit"):
        monkeypatch.setenv("RMHD_SLOWSHOCK", kernel)
        out[kernel] = SB.make_solver(GAMMA)(Byt, Bzt, ahead, switch, Bn, accuracy=1e-8)
    (V1, b1, e1), (V2, b2, e2) = out["numpy"], out["njit"]
    assert np.array_equal(e1, e2)
    ok = ~e1
    assert np.array_equal(V1[ok], V2[ok])
    for j in range(7):
        assert np.array_equal(b1[j][ok], b2[j][ok])
