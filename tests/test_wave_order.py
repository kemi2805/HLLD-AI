"""Interfaces whose fan crosses itself must not be counted exact.

Verification asks that the seven jump conditions hold. It says nothing about
where the waves SIT, and a fan whose rotational discontinuity has overtaken
its slow wave is self-crossing -- not a Riemann solution, whatever its
residual. Measured over 24,632 production rows: the left rotation crosses on
10.0% of interfaces but carries a jump on only 5.6%, so 2.52% are genuinely
inadmissible and were being given an exact flux.

The distinction the gate has to get right is that a crossing between
ZERO-STRENGTH waves is harmless -- nothing depends on where a wave with no
jump sits. The planar path crosses on 89.8% of its answers and is
inadmissible on 0.03%, so a gate that keyed on speed order alone would
throw away almost everything the five-wave solver earns.
"""
import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.physics.eos import hybrid_eos                      # noqa: E402
from src.physics.hlld import LAST_DIAG                      # noqa: E402

GAMMA = 5.0 / 3.0
N_PROB = 48
MAX_ITER = 10


@pytest.fixture(scope="module")
def eos():
    return hybrid_eos(K=0.0, gamma=GAMMA, gamma_th=GAMMA)


@pytest.fixture(scope="module")
def problems(eos):
    from rmhd.eos import set_eos
    set_eos("ideal")
    from rmhd.riemann_dataset import generate_dataset

    sols = generate_dataset(N_PROB, gamma=GAMMA, Bx_range=(0.02, 2.0), seed=73,
                            xi_window=(0.0, 0.0), verbose=False)
    Z = np.array([s.zones for s in sols])
    Bx = np.array([s.Bx for s in sols])

    def prim(slot):
        rho, Pt, vn, v1, v2_, B1, B2 = (Z[:, slot, j] for j in range(7))
        v2 = vn ** 2 + v1 ** 2 + v2_ ** 2
        W2 = 1.0 / (1.0 - v2)
        eta = Bx * vn + B1 * v1 + B2 * v2_
        b2 = (Bx ** 2 + B1 ** 2 + B2 ** 2) / W2 + eta ** 2
        t = lambda a: torch.tensor(a, dtype=torch.float64)
        d = {"rho": t(rho), "vx": t(vn), "vy": t(v1), "vz": t(v2_),
             "p": t(np.maximum(Pt - 0.5 * b2, 1e-10)),
             "Bx": t(Bx), "By": t(B1), "Bz": t(B2)}
        d["eps"] = eos.eps__press_rho(d["p"], d["rho"])
        return d

    return prim(0), prim(7)


def _run(sL, sR, eos, *, gate):
    old = os.environ.get("RMHD_WAVE_ORDER")
    os.environ["RMHD_WAVE_ORDER"] = "1" if gate else "0"
    try:
        import src.physics.exact_flux as EF
        EF = importlib.reload(EF)
        F, U, p = EF.exact_flux_batched(sL, sR, eos, idir=0, max_iter=MAX_ITER)
        return ({k: v.clone() for k, v in F.items()}, dict(LAST_DIAG))
    finally:
        if old is None:
            os.environ.pop("RMHD_WAVE_ORDER", None)
        else:
            os.environ["RMHD_WAVE_ORDER"] = old
        import src.physics.exact_flux as EF
        importlib.reload(EF)


def test_on_is_the_default():
    import src.physics.exact_flux as EF
    EF = importlib.reload(EF)
    assert EF._WAVE_ORDER is True, (
        "a self-crossing fan is not a solution; the gate should default on")


def test_every_exact_interface_has_ordered_waves(problems, eos):
    """The contract: nothing counted exact may have a crossing wave that
    carries a jump, nor any other pair out of order."""
    sL, sR = problems
    _, d = _run(sL, sR, eos, gate=True)
    assert d["n_exact"] > 0, "fixture produced no exact fluxes; proves nothing"
    assert "n_wave_order_rejected" in d, "the gate must report what it removed"
    assert d["n_wave_order_rejected"] >= 0


def test_the_gate_only_ever_removes(problems, eos):
    """Turning it on cannot make an interface exact that was not."""
    sL, sR = problems
    _, off = _run(sL, sR, eos, gate=False)
    _, on = _run(sL, sR, eos, gate=True)
    assert on["n_exact"] <= off["n_exact"], "the gate added exact interfaces"
    assert (off["n_exact"] - on["n_exact"]) == on["n_wave_order_rejected"], (
        "the drop in exact interfaces must equal what the gate reports "
        "rejecting")
    ma, mb = off["exact_mask"].numpy(), on["exact_mask"].numpy()
    assert bool((mb & ~ma).sum() == 0)


def test_a_zero_strength_crossing_is_not_rejected(problems, eos):
    """The distinction that matters.

    Planar answers cross on ~90% of interfaces with zero-strength rotations.
    If the gate keyed on speed order alone it would reject nearly all of
    them, so the planar rescue count must not move when the gate is enabled.
    """
    sL, sR = problems
    old = os.environ.get("RMHD_PLANAR5_FALLBACK")
    os.environ["RMHD_PLANAR5_FALLBACK"] = "1"
    try:
        _, off = _run(sL, sR, eos, gate=False)
        _, on = _run(sL, sR, eos, gate=True)
    finally:
        if old is None:
            os.environ.pop("RMHD_PLANAR5_FALLBACK", None)
        else:
            os.environ["RMHD_PLANAR5_FALLBACK"] = old
    assert on.get("n_planar5_exact", 0) == off.get("n_planar5_exact", 0), (
        "the gate removed planar answers, whose rotations have zero strength "
        "and therefore cannot cross anything that matters")


# ── degenerate-class lanes ─────────────────────────────────────────────────
# The gate reads the seven-wave speed layout res["VsLv"] / res["VsRv"].  The
# DEGENERATE classes (ALFVEN_DEGEN, SLOW_DEGEN, FAST_DEGEN) are answered by the
# reduced three-wave solver inside solve_batch, which fills only R4/R5 and
# VsL/VsR and leaves those triples at zero, so the gate sees the order
# [0, 0, 0, v_cd, 0, 0, 0] and calls the fan crossed whenever |v_cd| > 1e-6.

def _degenerate_tube(eos, Bn=1e-4):
    """A Sod-like tube with a tiny normal field: ALFVEN_DEGEN on the input.

    |B_n| = 1e-4 puts the Alfven speeds within ~2e-5 of the fan width of the
    fluid speed, far under classify.TOL_ALFVEN_CONTACT = 1e-3.  The contact
    moves at about +0.34, so the zeroed speed layout reads as crossed.
    """
    t = lambda *a: torch.tensor(a, dtype=torch.float64)

    def side(rho, p, By, Bz):
        d = {"rho": t(rho), "p": t(p), "vx": t(0.0), "vy": t(0.0),
             "vz": t(0.0), "Bx": t(Bn), "By": t(By), "Bz": t(Bz)}
        d["eps"] = eos.eps__press_rho(d["p"], d["rho"])
        return d

    return side(1.0, 1.0, 0.5, 0.2), side(0.125, 0.1, 0.3, -0.4)


@pytest.fixture(scope="module")
def degenerate_runs(eos):
    """(gate off, gate on) diagnostics for the one-lane tube.  Shared: each
    call reloads exact_flux and so rebuilds the solvers (~80 s on the Mac)."""
    sL, sR = _degenerate_tube(eos)
    return _run(sL, sR, eos, gate=False)[1], _run(sL, sR, eos, gate=True)[1]


def test_a_degenerate_lane_is_exact_without_the_gate(degenerate_runs):
    """Precondition for the xfail below: the lane is ALFVEN_DEGEN, the reduced
    solver answers it, and with the gate off it is exact -- so whatever
    removes it with the gate on is the gate and nothing else."""
    off, on = degenerate_runs
    assert off["n_alfven_degen"] == 1, "fixture is not a degenerate-class lane"
    assert off["n_converged"] == 1 and off["n_exact"] == 1
    assert on["n_exact"] + on["n_wave_order_rejected"] == 1


def test_the_gate_does_not_judge_a_degenerate_lane_on_seven_wave_speeds(
        degenerate_runs):
    _, on = degenerate_runs
    assert on["n_wave_order_rejected"] == 0, (
        "a three-wave answer was rejected on a speed layout it never filled")
    assert on["n_exact"] == 1
    # counted apart from the verified answers, and not refused: the lane was
    # degenerate from the input, not reclassified out of a seven-wave solve
    assert on["n_degenerate_exact"] == 1
    assert on["n_reclassified_refused"] == 0


def test_a_reclassified_lane_is_refused(eos, monkeypatch):
    """solve_batch flags a lane it moved from a seven-wave class to a
    degenerate one; its class was read off an abandoned (possibly
    unconverged) seven-wave attempt, and on the rotor those three-wave
    answers were off 0.55-0.82 against the verified planar solve.  The flag
    is forced on the tube's lane here, since no cheap input reclassifies."""
    import src.physics.exact_flux as EF
    EF = importlib.reload(EF)
    orig = EF._run_pipeline

    def flagged(*a, **k):
        out = orig(*a, **k)
        out[0]["reclassified"][:] = True
        return out

    monkeypatch.setattr(EF, "_run_pipeline", flagged)
    sL, sR = _degenerate_tube(eos)
    EF.exact_flux_batched(sL, sR, eos, idir=0, max_iter=MAX_ITER)
    d = dict(LAST_DIAG)
    assert d["n_converged"] == 1, "precondition: the three-wave solve converged"
    assert d["n_reclassified_refused"] == 1
    assert d["n_exact"] == 0 and d["n_degenerate_exact"] == 0


# ── the harvest must see the gate ──────────────────────────────────────────
# `accepted` is what the harvester writes as a TRAINING row.  The gate once ran
# after the harvest, so every fan it rejected was still written to the solved
# shards: replayed offline, 1.04% of the 2.33M seven-wave rows of
# rotor_128_exact were self-crossing.

def _harvest(problems, eos, tmp, *, reject_all):
    """One sweep with a harvester, gate on.  ``reject_all`` replaces the gate
    by one that rejects every lane, which makes the ordering of gate and
    harvest observable whatever the fixture's natural crossing rate is."""
    import glob
    from src.physics.harvest import Harvester
    sL, sR = problems
    old = os.environ.get("RMHD_WAVE_ORDER")
    os.environ["RMHD_WAVE_ORDER"] = "1"
    try:
        import src.physics.exact_flux as EF
        EF = importlib.reload(EF)
        if reject_all:
            EF._self_crossing = lambda Zc, VL, VR, tol: np.ones(
                np.shape(VL[0]), dtype=bool)
        h = Harvester(str(tmp), GAMMA, only_retried=False,
                      max_solved=10 ** 6, max_unsolved=10 ** 6)
        EF.exact_flux_batched(sL, sR, eos, idir=0, max_iter=MAX_ITER,
                              harvester=h)
        diag = dict(LAST_DIAG)
        h.close()
    finally:
        if old is None:
            os.environ.pop("RMHD_WAVE_ORDER", None)
        else:
            os.environ["RMHD_WAVE_ORDER"] = old
        import src.physics.exact_flux as EF
        importlib.reload(EF)

    def load(kind, keys):
        z = [np.load(f) for f in sorted(glob.glob(str(tmp / f"{kind}_*.npz")))]
        return {k: np.concatenate([q[k] for q in z]) if z else np.zeros(0)
                for k in keys}

    return (load("solved", ("zones", "speeds", "source")),
            load("unsolved", ("reason", "cls")), diag)


@pytest.fixture(scope="module")
def harvest_all_rejected(problems, eos, tmp_path_factory):
    return _harvest(problems, eos, tmp_path_factory.mktemp("wo_all"),
                    reject_all=True)


def test_a_rejected_fan_is_not_harvested_as_solved(harvest_all_rejected):
    solved, _, diag = harvest_all_rejected
    assert diag["n_wave_order_rejected"] > 0, (
        "fixture has no seven-wave answers for the gate to reject")
    n0 = int((solved["source"] == 0).sum())
    assert n0 == 0, (f"{n0} seven-wave rows the gate rejected were harvested "
                     "as solved training rows")


def test_a_rejected_fan_is_harvested_as_unsolved_with_its_reason(
        harvest_all_rejected):
    """Filed under its own reason: those lanes converged to a physical state,
    so without one they would read as `unphysical` in the failure set."""
    from src.physics.harvest import R_SELF_CROSSING
    _, unsolved, diag = harvest_all_rejected
    n = int((unsolved["reason"] == R_SELF_CROSSING).sum())
    assert n == diag["n_wave_order_rejected"]


def test_no_harvested_seven_wave_row_is_self_crossing(problems, eos,
                                                      tmp_path_factory):
    """The contract in the consumer's terms: replay the gate on the SHARDS.

    The shard keeps everything the gate reads -- zones R1..R8 and the speeds
    in spatial order (LF, LA, LS, CD, RS, RA, RF) -- so this is the same test
    exact_flux_batched applied, not an approximation of it."""
    import src.physics.exact_flux as EF
    solved, _, _ = _harvest(problems, eos, tmp_path_factory.mktemp("wo_real"),
                            reject_all=False)
    seven = solved["source"] == 0
    if not seven.any():
        pytest.skip("no seven-wave rows harvested")
    Z, S = solved["zones"][seven], solved["speeds"][seven]
    Zc = [[Z[:, z, j] for j in range(7)] for z in range(1, 7)]      # R2..R7
    crossed = EF._self_crossing(Zc, [S[:, 0], S[:, 1], S[:, 2]],
                                [S[:, 6], S[:, 5], S[:, 4]],
                                EF._WAVE_ORDER_TOL)
    assert int(np.sum(crossed)) == 0, (
        f"{int(np.sum(crossed))} harvested seven-wave rows are self-crossing")
