"""The four-unknown planar solver from a caller's start (planar4_b.solve
``unk_init``) and with a central-difference Jacobian (``central``).

Found 2026-11-04 on the failure ledger of the four-quadrant 128^2 run (ft3).
On faces with the field nearly normal the solution can carry B_t through zero
inside a slow RAREFACTION on each side: no pi-rotation exists for these
relativistic states, and every reversing slow shock from them lowers the
entropy.  A field-steered fan cannot cross zero, a pressure-steered one can,
and the default start (the mean total pressure) is 15% off where both fast
waves are rarefactions -- HLLD's star pressure is not.  Lane 215 is that face.
Lane 1099 (a reversing fan on the left, a slow shock on the right) converges
from the star pressure only with central differences: the forward ones
zig-zag at 0.9 per iteration on its Jacobian of condition ~2e4.  Lane 900
(beta ~ 300, field normal to 0.7%) needs the signed-field slow wave (KIND_S):
there a total-pressure change of 1e-5 swings B_t by 0.1, so pressure is an
ill-conditioned coordinate, and |B_t| cannot cross zero.  Lane 858 (strong
field, its right fast wave 6e-4 from the Alfven speed) converges once that
weak fast wave starts at its own ahead pressure instead of at p*.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
np.seterr(all="ignore")

from rmhd.batched import fan_njit as FN       # noqa: E402
from rmhd.batched import planar4_b as P4      # noqa: E402

GAMMA = 1.6666666666666667
LANES = {
    215: dict(
        UL=[0.15180878997495367, 3.999093849758541, -0.57339491582865,
            -0.3301313917886574, 0.0, -0.0217955723084498, 0.0, 0.6509288913300132],
        UR=[0.18162720099582397, 4.003782844145203, -0.4820903568953471,
            -0.2999256739967405, 0.0, -0.021795572308449803, 0.0, 0.6509288913300132],
        PSTAR=3.46132318166412),
    1099: dict(
        UL=[0.49842557625513184, 29.257798784863947, -0.5771556809886513,
            -0.4120358114376433, 0.0, -0.057603453899823964, 0.0, 1.2246227291664036],
        UR=[0.5751463954056417, 32.92138736036745, -0.5623228832071362,
            -0.39789299774139547, 0.0, 0.13258566332768162, 0.0, 1.2246227291664036],
        PSTAR=30.218592830969076),
    900: dict(
        UL=[1.3841344608035706, 106.99391980953312, -0.056164017695629884,
            -0.29184282390617183, 0.0, -0.006077328559210518, 0.0, 0.8740886066802181],
        UR=[1.3852836031360922, 107.00793794052991, -0.06403796159947266,
            -0.2814586000263408, 0.0, -0.006077328559210518, 0.0, 0.8740886066802181],
        PSTAR=107.93410877537106),
    858: dict(
        UL=[0.49232275285131594, 2.4635799831970693, -0.15265937260975632,
            -0.6846553486164344, 0.0, 0.8215029605151833, -0.0, 2.2503926554507077],
        UR=[0.5151179169063762, 2.7270494120956243, -0.1611033727386318,
            -0.649154131255671, 0.0, 0.8850096255783002, -0.0, 2.2503926554507077],
        PSTAR=2.683857608959755),
}


@pytest.fixture(scope="module")
def S():
    return P4.make_solver(GAMMA)


def _problem(lane):
    d = LANES[lane]
    L = [np.array([c]) for c in d["UL"][:7]]
    R = [np.array([c]) for c in d["UR"][:7]]
    return L, R, np.array([d["UL"][7]]), d["PSTAR"]


def _star_start(S, L, R, Bn, p, kinds, delta=1e-4, pr=None):
    """Both fast waves to p (the right one to ``pr`` if given), the slow
    waves weak fans steered as ``kinds`` (0 field, 1 pressure, 2 signed
    field)."""
    pr = p if pr is None else pr
    L0, R0 = S["to_planar"](L, R, Bn)[:2]
    W = S["waves"]
    A, _, _ = W["fast_wave"](np.array([p]), L0, "LF", Bn)
    D, _, _ = W["fast_wave"](np.array([pr]), R0, "RF", Bn)
    sl = np.log(A[1] * (1 - delta)) if kinds[0] == 1 else A[5] * (1 + delta)
    sr = np.log(D[1] * (1 - delta)) if kinds[1] == 1 else D[5] * (1 + delta)
    return [np.log(np.array([p])), sl, sr, np.log(np.array([pr]))]


def test_default_start_loses_the_reversing_fans(S):
    L, R, Bn, _ = _problem(215)
    r = S["solve"](L, R, Bn)
    assert not r["converged"][0], "the default start now solves lane 215: the fixture no longer tests the start"


def test_star_start_solves_the_reversing_fans(S):
    L, R, Bn, p = _problem(215)
    kinds = np.array([[P4.KIND_P], [P4.KIND_P]], np.int8)
    r = S["solve"](L, R, Bn, kinds=kinds, retry=False,
                   unk_init=_star_start(S, L, R, Bn, p, (1, 1)))
    assert r["converged"][0] and r["reason"][0] == 0
    assert not r["shocks"][:, 0].any(), "four fans expected"
    A, B, C, D = r["zones"]
    # the field reverses inside both slow fans (lab frame: By < 0 outside)
    assert A[5][0] < 0 and D[5][0] < 0 and B[5][0] > 0 and C[5][0] > 0


def test_central_differences_converge_where_forward_zigzag(S):
    L, R, Bn, p = _problem(1099)
    kinds = np.array([[P4.KIND_P], [P4.KIND_B]], np.int8)
    u0 = _star_start(S, L, R, Bn, p, (1, 0))
    fwd = S["solve"](L, R, Bn, kinds=kinds, retry=False, unk_init=u0)
    cen = S["solve"](L, R, Bn, kinds=kinds, retry=False, unk_init=u0, central=True)
    assert not fwd["converged"][0], "forward differences now converge: the fixture no longer tests central ones"
    assert cen["converged"][0] and cen["reason"][0] == 0
    assert cen["cert_resid"][0] <= 1e-8 and cen["ordered"][0]


def test_unk_init_and_central_off_change_nothing(S):
    """Without the new arguments the solve is the old one, bit for bit."""
    for lane in LANES:
        L, R, Bn, _ = _problem(lane)
        a = S["solve"](L, R, Bn)
        b = S["solve"](L, R, Bn, unk_init=None, central=False)
        for k in ("nrm", "reason", "n_iter"):
            assert np.array_equal(np.asarray(a[k]), np.asarray(b[k]))
        for c in range(4):
            assert np.array_equal(a["unk"][c], b["unk"][c])


KERNELS = ["numpy"] + (["njit"] if FN.HAVE_NUMBA else [])


@pytest.mark.parametrize("kernel", KERNELS)
def test_signed_fan_carries_the_field_through_zero(S, monkeypatch, kernel):
    """The signed-field fan from lane 900's post-fast state to a field of the
    opposite sign: isentropic, the pressure falls, B_t lands on the target."""
    monkeypatch.setenv("RMHD_FAN", kernel)
    L, R, Bn, p = _problem(900)
    L0 = S["to_planar"](L, R, Bn)[0]
    A, _, _ = S["waves"]["fast_wave"](np.array([p]), L0, "LF", Bn)
    target = -10.0 * A[5]
    y, V, e = S["fan_signed"](target, A, "LS", Bn)
    assert not e[0]
    assert y[5][0] == target[0] and y[1][0] < A[1][0]

    def K(X):
        rho, Pt, vx, vy, vz, By, Bz = X
        W2 = 1.0 / (1.0 - vx * vx - vy * vy - vz * vz)
        vB = vx * Bn + vy * By + vz * Bz
        return (Pt - 0.5 * ((Bn ** 2 + By ** 2 + Bz ** 2) / W2 + vB ** 2)) / X[0] ** GAMMA

    assert abs(K(y)[0] / K(A)[0] - 1.0) < 1e-9


@pytest.mark.skipif(not FN.HAVE_NUMBA, reason="numba not installed")
def test_signed_fan_njit_matches_numpy(S, monkeypatch):
    L, R, Bn, p = _problem(215)
    L0 = S["to_planar"](L, R, Bn)[0]
    A, _, _ = S["waves"]["fast_wave"](np.array([p]), L0, "LF", Bn)
    out = {}
    for kernel in ("numpy", "njit"):
        monkeypatch.setenv("RMHD_FAN", kernel)
        out[kernel] = S["fan_signed"](np.array([-3.0 * A[5][0]]), A, "LS", Bn)
    (y1, V1, e1), (y2, V2, e2) = out["numpy"], out["njit"]
    assert not e1[0] and not e2[0]
    for j in range(7):
        assert abs(y1[j][0] - y2[j][0]) <= 1e-12 * max(1.0, abs(y1[j][0]))


def test_signed_field_steering_solves_lane_900(S):
    L, R, Bn, p = _problem(900)
    kinds = np.array([[P4.KIND_S], [P4.KIND_S]], np.int8)
    r = S["solve"](L, R, Bn, kinds=kinds, retry=False, central=True,
                   unk_init=_star_start(S, L, R, Bn, p, (2, 2)))
    assert r["converged"][0] and r["reason"][0] == 0
    for kd in ((1, 1), (0, 0)):
        k2 = np.array(kd, np.int8).reshape(2, 1)
        r2 = S["solve"](L, R, Bn, kinds=k2, retry=False, central=True,
                        unk_init=_star_start(S, L, R, Bn, p, kd))
        assert not r2["converged"][0], "steering %s now solves lane 900" % (kd,)


def test_weak_fast_wave_start_solves_lane_858(S):
    L, R, Bn, p = _problem(858)
    kinds = np.array([[P4.KIND_P], [P4.KIND_P]], np.int8)
    r = S["solve"](L, R, Bn, kinds=kinds, retry=False, central=True,
                   unk_init=_star_start(S, L, R, Bn, p, (1, 1)))
    assert not r["converged"][0], "p* now solves lane 858: the fixture no longer tests the start"
    pr = float(R[1][0]) * (1.0 + 1e-3)          # p* is within 2% of P_R
    r = S["solve"](L, R, Bn, kinds=kinds, retry=False, central=True,
                   unk_init=_star_start(S, L, R, Bn, p, (1, 1), pr=pr))
    assert r["converged"][0] and r["reason"][0] == 0
    assert r["shocks"][0, 0] and r["shocks"][3, 0], "two fast shocks expected"

