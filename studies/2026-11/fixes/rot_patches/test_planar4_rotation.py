"""The coplanar rotation (fullcontact_b.coplanar_rotation) and the
four-unknown planar solver with it in front of the slow waves (planar4_b
``rot``).

Found 2026-11-05 on the failure ledger of the four-quadrant 128^2 run (ft3):
the four-unknown answers the admissibility gate refuses carry a slow jump that
reverses B_t, runs between the Alfven speeds of its two sides and lowers the
entropy by 1e-9..7e-5 -- a ROTATION fused into the slow jump, because planar4
has no rotations.  A coplanar rotation turns the field through pi in the fluid
frame, and in the lab its B_t lands on either sign: on lane 1013 the left one
reverses B_t (+0.0552 -> -0.0449), the right one keeps its sign (+0.0569 ->
+0.0220).  Production's reversal (the lab angle + pi) cannot build the second.
With the rotation in place every refused lane tested (112) has a regular,
admissible seven-wave answer whose flux differs from the refused one by 1e-8.
The SEED of each lane is its refused answer: [ln P behind LF, B_t behind LS,
B_t behind RS, ln P behind RF], planar frame.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
np.seterr(all="ignore")

from rmhd.batched import fullcontact_b as FB   # noqa: E402
from rmhd.batched import planar4_b as P4       # noqa: E402

GAMMA = 1.6666666666666667
LANES = {
    1013: dict(
        UL=[1.378386925848998, 98.66974128503908, 0.07190488089954004, -0.33832185247875757,
            0.0, 0.05340731061629455, -0.0, 0.8856402602503213],
        UR=[1.4028585297569935, 102.42968253162465, 0.07717537567358936, -0.3863115697967262,
            0.0, 0.05340731061629455, -0.0, 0.8856402602503213],
        SEED=[4.604161204797077, -0.4295265047587535, -0.4295265047587535, 4.604161356986647],
        ROT=(True, True)),
    166: dict(
        UL=[0.5694732694256432, 30.963080295607135, 0.35533107947466447, -0.046420604876458126,
            0.0, 0.3255922518436347, 0.0, 1.1852902206951548],
        UR=[0.633771467458453, 39.367777055453985, 0.5478325752265099, -0.2132316720391396,
            0.0, 0.3255922518436347, 0.0, 1.1852902206951548],
        SEED=[3.300327729698877, -0.7852852157048479, -0.7852852157048479, 3.3003470988161983],
        ROT=(True, True)),
    214: dict(
        UL=[0.14804848609027457, 2.9796639888835417, 0.5731287934351738, -0.3047573744857859,
            0.0, -0.12018022546425375, -0.0, 1.6184622983565142],
        UR=[0.23970883491428194, 3.913007906673923, 0.5251090991935994, -0.11440750533910755,
            0.0, -0.12018022546425375, -0.0, 1.6184622983565142],
        SEED=[1.287247181737806, -0.22205663202099474, -0.22205663202099474, 1.2817850997131406],
        ROT=(True, False)),
}


@pytest.fixture(scope="module")
def S():
    return P4.make_solver(GAMMA)


def _problem(S, lane):
    d = LANES[lane]
    L = [np.array([c]) for c in d["UL"][:7]]
    R = [np.array([c]) for c in d["UR"][:7]]
    Bn = np.array([d["UL"][7]])
    L0, R0 = S["to_planar"](L, R, Bn)[:2]
    return L, R, L0, R0, Bn


def _fast_states(S, lane):
    L, R, L0, R0, Bn = _problem(S, lane)
    W = S["waves"]
    seed = LANES[lane]["SEED"]
    A, _, _ = W["fast_wave"](np.exp(np.array([seed[0]])), L0, "LF", Bn)
    D, _, _ = W["fast_wave"](np.exp(np.array([seed[3]])), R0, "RF", Bn)
    return A, D, Bn


def _K(X, Bn):
    rho, Pt, vx, vy, vz, By, Bz = X
    W2 = 1.0 / (1.0 - (vx * vx + vy * vy + vz * vz))
    vB = vx * Bn + vy * By + vz * Bz
    return (Pt - 0.5 * ((Bn ** 2 + By ** 2 + Bz ** 2) / W2 + vB ** 2)) / rho ** GAMMA


def test_rotation_lands_on_either_lab_sign(S):
    A, D, Bn = _fast_states(S, 1013)
    for X, side, flips in ((A, "L", True), (D, "R", False)):
        Vs = S["alfven_speed"](X, Bn, side)
        Y, err = FB.coplanar_rotation(X, Bn, Vs, GAMMA)
        assert not err[0]
        assert (np.sign(Y[5][0]) != np.sign(X[5][0])) == flips
        assert abs(Y[5][0] - X[5][0]) > 1e-2                  # not the identity
        assert S["rh_residual"](X, Y, Vs, Bn)[0] < 1e-12
        assert abs(Y[0][0] / X[0][0] - 1.0) < 1e-12           # rho continuous
        assert abs(Y[1][0] / X[1][0] - 1.0) < 1e-12           # and the total pressure
        assert abs(_K(Y, Bn)[0] / _K(X, Bn)[0] - 1.0) < 1e-12


def test_reversal_alone_misses_the_same_sign_rotation(S):
    """production's reversal: the closed form at the lab angle + pi"""
    _, D, Bn = _fast_states(S, 1013)
    Vs = S["alfven_speed"](D, Bn, "R")
    _, err = FB.exact_rotation(D, Bn, Vs, np.arctan2(D[6], D[5]) + np.pi, GAMMA)
    assert err[0]
    _, err2 = FB.coplanar_rotation(D, Bn, Vs, GAMMA)
    assert not err2[0]


def test_no_rotation_is_the_old_structure(S):
    """rot all False: the same residual, bit for bit, as without ``rot``"""
    L, R, L0, R0, Bn = _problem(S, 1013)
    u = [np.array([c]) for c in LANES[1013]["SEED"]]
    kinds = np.full((2, 1), P4.KIND_S, dtype=np.int8)
    f0 = S["structure"](L0, R0, u, Bn, kinds)
    f1 = S["structure"](L0, R0, u, Bn, kinds, rot=np.zeros((2, 1), bool))
    assert len(f0) == 5 and len(f1) == 6
    assert np.array_equal(f0[0], f1[0])
    for z0, z1 in zip(f0[1], f1[1]):
        for a, b in zip(z0, z1):
            assert np.array_equal(a, b)


@pytest.mark.parametrize("lane", sorted(LANES))
def test_rotation_answer_is_admissible(S, lane):
    L, R, L0, R0, Bn = _problem(S, lane)
    d = LANES[lane]
    rot = np.array(d["ROT"], dtype=bool).reshape(2, 1)
    r = S["solve"](L, R, Bn, kinds=np.full((2, 1), P4.KIND_S, dtype=np.int8), retry=False,
                   unk_init=[np.array([c]) for c in d["SEED"]], central=True, rot=rot,
                   max_iter=25)
    assert r["converged"][0] and r["ordered"][0]
    assert r["cert_resid"][0] < 1e-8
    A, B, C, D = r["zones"]
    A2, D2 = r["zones_rot"]
    # every shock raises the entropy; the rotations keep it
    for k, (ahead, behind) in enumerate(((L, A), (A2, B), (D2, C), (R, D))):
        if r["shocks"][k][0]:
            assert _K(behind, Bn)[0] / _K(ahead, Bn)[0] >= 1.0 - 1e-12
    for k, (ahead, behind) in enumerate(((A, A2), (D, D2))):
        if rot[k][0]:
            assert abs(_K(behind, Bn)[0] / _K(ahead, Bn)[0] - 1.0) < 1e-12
