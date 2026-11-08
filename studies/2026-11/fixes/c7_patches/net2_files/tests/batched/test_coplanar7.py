"""The planar form of the seven-wave family on COPLANAR interfaces
(coplanar7_b, ``RMHD_COPLANAR7``).

Ledger lanes of the four-quadrant 128^2 run (ft3) whose only answer before
2026-11-05 carried a slow jump across the Alfven speed that lowers the
entropy: seeded as the seven-wave solver would be (no rotation suggested), the
planar form finds the regular answer -- rotation on both sides for 1013, on the
left for 214 -- in its second stage, every shock raising the entropy.  With
the switch off, `api.solve_batch` never builds it and returns no new keys.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
np.seterr(all="ignore")

from rmhd.batched import coplanar7_b as CP7    # noqa: E402

GAMMA = 1.6666666666666667
LANES = {
    1013: dict(
        UL=[1.378386925848998, 98.66974128503908, 0.07190488089954004, -0.33832185247875757,
            0.0, 0.05340731061629455, -0.0, 0.8856402602503213],
        UR=[1.4028585297569935, 102.42968253162465, 0.07717537567358936, -0.3863115697967262,
            0.0, 0.05340731061629455, -0.0, 0.8856402602503213],
        # [ln p_LF, ln|Bt_CD|, psi_CD, ln p_RF, phi_L, phi_R]: the refused answer, no rotation
        S6=[4.604161204797077, np.log(0.4295265047587535), np.pi, 4.604161356986647, 0.0, 0.0],
        ROT=(True, True)),
    214: dict(
        UL=[0.14804848609027457, 2.9796639888835417, 0.5731287934351738, -0.3047573744857859,
            0.0, -0.12018022546425375, -0.0, 1.6184622983565142],
        UR=[0.23970883491428194, 3.913007906673923, 0.5251090991935994, -0.11440750533910755,
            0.0, -0.12018022546425375, -0.0, 1.6184622983565142],
        S6=[1.287247181737806, np.log(0.22205663202099474), 0.0, 1.2817850997131406, 0.0, 0.0],
        ROT=(True, False)),
}


@pytest.fixture(scope="module")
def S():
    return CP7.make_solver(GAMMA)


@pytest.fixture(autouse=True)
def _production_seed(monkeypatch):
    """These tests are about the staged solve from production's seed; the
    planar network (on by default since 2026-11-08) is test_planar_net.py's."""
    monkeypatch.setattr(CP7, "NET", "")


def _problem(lane):
    d = LANES[lane]
    L = [np.array([c]) for c in d["UL"][:7]]
    R = [np.array([c]) for c in d["UR"][:7]]
    s6 = [np.array([c]) for c in d["S6"]]
    return L, R, np.array([d["UL"][7]]), s6


@pytest.mark.parametrize("lane", sorted(LANES))
def test_regular_answer_from_an_unrotated_seed(S, lane):
    L, R, Bn, s6 = _problem(lane)
    r = S["solve"](L, R, Bn, s6, accuracy=1e-8)
    assert r["converged"][0]
    assert r["stage"][0] == 2                       # the seed suggested no rotation
    assert (bool(r["rot"][0][0]), bool(r["rot"][1][0])) == LANES[lane]["ROT"]
    # seven-wave layout: the rotations are Alfven waves, rho and p continuous
    R2, R3, R4, R5, R6, R7 = r["zones"]
    for k, (a, b) in enumerate(((R2, R3), (R7, R6))):
        if LANES[lane]["ROT"][k]:
            assert abs(b[0][0] / a[0][0] - 1.0) < 1e-12
            assert abs(b[5][0] - a[5][0]) > 1e-3
    # the ordering of the seven waves
    VL, VR = r["VsLv"], r["VsRv"]
    assert VL[0][0] <= VL[1][0] + 1e-9 and VR[1][0] <= VR[0][0] + 1e-9


@pytest.mark.parametrize("calls", [2, 1])
def test_merged_calls_give_the_staged_answers(S, monkeypatch, calls):
    """RMHD_COPLANAR7_CALLS: the stages merged into fewer planar4 solves
    answer every lane with the same branch, to the bit, and count stage and
    iterations as the staged solve does (both lanes answer at stage 2, so the
    merged solve also ran branches the staged one never started)."""
    L1, R1, B1, s1 = _problem(1013)
    L2, R2, B2, s2 = _problem(214)
    cat = lambda a, b: [np.concatenate([x, y]) for x, y in zip(a, b)]
    L, R, Bn, s6 = cat(L1, L2), cat(R1, R2), np.concatenate([B1, B2]), cat(s1, s2)
    monkeypatch.setattr(CP7, "CALLS", 4)
    ref = S["solve"](L, R, Bn, s6, accuracy=1e-8)
    monkeypatch.setattr(CP7, "CALLS", calls)
    got = S["solve"](L, R, Bn, s6, accuracy=1e-8)
    assert ref["converged"].all()
    for key in ("converged", "stage", "n_iter"):
        assert np.array_equal(got[key], ref[key]), key
    for key in ("zones",):
        for a, b in zip(got[key], ref[key]):
            for x, y in zip(a, b):
                assert np.array_equal(x, y), key
    for key in ("VsLv", "VsRv", "unk6", "shocks", "kinds", "rot"):
        for x, y in zip(got[key], ref[key]):
            assert np.array_equal(x, y), key


@pytest.mark.skipif("RMHD_COPLANAR7" in os.environ, reason="the switch is set")
def test_off_by_default():
    assert not CP7.wanted()


def test_solve_batch_without_the_switch_has_no_new_keys(monkeypatch):
    from rmhd.batched import api
    monkeypatch.setattr(CP7, "MODE", "0")
    L, R, Bn, s6 = _problem(1013)
    res, diag = api.make_solver(GAMMA)(L, R, Bn, seed6=s6, accuracy=1e-8, max_iter=40)
    assert "coplanar7" not in res and "n_coplanar7" not in diag


def test_solve_batch_with_the_switch_answers_through_the_planar_form(monkeypatch):
    from rmhd.batched import api
    monkeypatch.setattr(CP7, "MODE", "1")
    L, R, Bn, s6 = _problem(1013)
    res, diag = api.make_solver(GAMMA)(L, R, Bn, seed6=s6, accuracy=1e-8, max_iter=40)
    assert res["converged"][0] and res["coplanar7"][0]
    assert diag["n_coplanar7"] == 1 and diag["n_coplanar7_rotations"] == 1


def test_late_mode_stays_out_of_solve_batch(monkeypatch):
    """RMHD_COPLANAR7=late: the exact flux offers the planar form as a rung
    after the five-wave solver; `api.solve_batch` builds no stage of it."""
    from rmhd.batched import api
    monkeypatch.setattr(CP7, "MODE", "late")
    assert CP7.late() and not CP7.wanted() and not CP7.after()
    L, R, Bn, s6 = _problem(1013)
    res, diag = api.make_solver(GAMMA)(L, R, Bn, seed6=s6, accuracy=1e-8, max_iter=40)
    assert "coplanar7" not in res and "n_coplanar7" not in diag


def test_after_mode_answers_what_the_newton_leaves(monkeypatch):
    """RMHD_COPLANAR7=after: fullcontact6 first; the lane it cannot answer
    admissibly (from the refused answer's seed) goes through the planar form."""
    from rmhd.batched import api
    monkeypatch.setattr(CP7, "MODE", "after")
    L, R, Bn, s6 = _problem(1013)
    res, diag = api.make_solver(GAMMA)(L, R, Bn, seed6=s6, accuracy=1e-8, max_iter=40)
    assert res["converged"][0] and res["coplanar7"][0]
    R2, R3, R4, R5, R6, R7 = res["zones"]
    assert CP7.entropy_min7(L, R, res["zones"], Bn, GAMMA)[0] >= 1.0 - 1e-9
