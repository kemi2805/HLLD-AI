"""The five-wave planar rescue in the flux path: it must ADD lanes only.

Same contract as ``test_three_wave_fallback.py``, and the same reasoning for
the reload idiom -- the knobs are read at import so a compiled kernel can
never capture a stale value.

The difference from the three-wave rescue is what justifies this one being
usable at all: ``planar5_b.solve`` verifies every answer against the FULL
seven-wave residual before reporting it converged, so an accepted lane
carries an exact solution rather than a close one.  These tests check the
plumbing around that; ``rmhd_final/batched/test_planar5.py`` checks the
claim itself.
"""
import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.physics.eos import hybrid_eos                       # noqa: E402
from src.physics.hlld import LAST_DIAG                       # noqa: E402

GAMMA = 5.0 / 3.0
N_PROB = 24
MAX_ITER = 10


@pytest.fixture(scope="module")
def eos():
    return hybrid_eos(K=0.0, gamma=GAMMA, gamma_th=GAMMA)


@pytest.fixture(scope="module")
def problems(eos):
    """COPLANAR problems -- what a 2D run presents, and the only population
    the planar solver claims."""
    from rmhd.eos import set_eos
    set_eos("ideal")
    from rmhd.riemann_dataset import generate_dataset

    sols = generate_dataset(N_PROB, gamma=GAMMA, Bx_range=(0.01, 1.5), seed=77,
                            xi_window=(0.0, 0.0), coplanar_frac=1.0,
                            verbose=False)
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
             "p": t(Pt - 0.5 * b2), "Bx": t(Bx), "By": t(B1), "Bz": t(B2)}
        d["eps"] = eos.eps__press_rho(d["p"], d["rho"])
        return d

    return prim(0), prim(7), len(sols)


from conftest import PINNED                                   # noqa: E402


def _run(sL, sR, eos, *, planar5, tol=None, extra=None):
    extra = dict(PINNED, **(extra or {}))
    old = {k: os.environ.get(k) for k in
           ("RMHD_PLANAR5_FALLBACK", "RMHD_PLANAR_TOL") + tuple(extra)}
    os.environ["RMHD_PLANAR5_FALLBACK"] = "1" if planar5 else "0"
    os.environ.update(extra)
    if tol is not None:
        os.environ["RMHD_PLANAR_TOL"] = tol
    try:
        import src.physics.exact_flux as EF
        EF = importlib.reload(EF)
        F, U, p = EF.exact_flux_batched(sL, sR, eos, idir=0, max_iter=MAX_ITER)
        return ({k: v.clone() for k, v in F.items()}, p.clone(), dict(LAST_DIAG))
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        import src.physics.exact_flux as EF
        importlib.reload(EF)


def test_off_is_the_default():
    import src.physics.exact_flux as EF
    EF = importlib.reload(EF)
    assert EF._PLANAR5_FALLBACK is False


def test_seven_wave_lanes_are_untouched(eos, problems):
    sL, sR, n = problems
    F0, p0, d0 = _run(sL, sR, eos, planar5=False)
    F1, p1, d1 = _run(sL, sR, eos, planar5=True)
    m0 = d0["exact_mask"]
    assert torch.equal(p1[m0], p0[m0]), "the rescue moved a seven-wave lane's p*"
    for k in F0:
        assert torch.equal(F1[k][m0], F0[k][m0]), f"the rescue moved {k}"


def test_it_adds_lanes_and_loses_none(eos, problems):
    sL, sR, n = problems
    F0, p0, d0 = _run(sL, sR, eos, planar5=False)
    F1, p1, d1 = _run(sL, sR, eos, planar5=True)
    assert not bool((d0["exact_mask"] & ~d1["exact_mask"]).any()), (
        "the rescue LOST a lane that was exact without it")
    new = int((d1["exact_mask"] & ~d0["exact_mask"]).sum())
    assert d1["n_exact"] == d0["n_exact"] + new
    assert new == d1["n_planar5_exact"]


def test_the_planarity_gate_is_respected(eos, problems):
    """With the gate closed nothing is accepted and the flux is unchanged."""
    sL, sR, n = problems
    F0, p0, d0 = _run(sL, sR, eos, planar5=False)
    F1, p1, d1 = _run(sL, sR, eos, planar5=True, tol="0.0")
    assert d1["n_planar5_exact"] == 0
    assert d1["n_exact"] == d0["n_exact"]
    assert torch.equal(p1, p0)


def test_diagnostics_account_for_every_interface(eos, problems):
    sL, sR, n = problems
    _, _, d = _run(sL, sR, eos, planar5=True)
    assert d["n_interfaces"] == n
    assert d["n_exact"] + d["n_hlld_fallback"] == n
    assert d["n_planar5_exact"] <= d["n_planar5_attempted"]
    assert int(d["planar5_mask"].sum()) == d["n_planar5_exact"]


def test_batch_independence_with_the_rescue(eos, problems):
    """A lane's answer must not depend on which lanes share its batch."""
    sL, sR, n = problems
    F, p, _ = _run(sL, sR, eos, planar5=True)
    half = n // 2
    for sl in (slice(0, half), slice(half, None)):
        a = {k: v[sl] for k, v in sL.items()}
        b = {k: v[sl] for k, v in sR.items()}
        Fp, pp, d = _run(a, b, eos, planar5=True)
        ex = d["exact_mask"]
        assert torch.equal(pp[ex], p[sl][ex])
        for k in F:
            assert torch.equal(Fp[k], F[k][sl]), f"splitting moved {k}"


def test_verified_is_a_subset_of_exact_and_covers_the_planar_lanes(eos, problems):
    """The rule the paper states: exact means verified, per interface.

    ``n_verified`` counts interfaces whose structure satisfies the COMPLETE
    seven-wave jump conditions to 1e-8, whichever solver produced it.  Two
    things must hold: it is never larger than ``n_exact``, and every lane the
    planar solver won is in it, since that solver verifies before reporting
    convergence.

    It is deliberately NOT asserted equal.  Measured on eighteen recorded
    sweeps, 138 of 763 exact fluxes fail the check, and they are the
    degenerate-class interfaces that `solve_batch` routes to the reduced
    three-wave solver -- which drops the slow waves and therefore does not
    produce a root of the full system.  Those lanes have always been counted
    exact; this test pins the discrepancy rather than hiding it, so that
    closing it is a deliberate decision and not a silent one.
    """
    sL, sR, n = problems
    for planar5 in (False, True):
        _, _, d = _run(sL, sR, eos, planar5=planar5)
        assert d["n_verified"] <= d["n_exact"]
        v, ex = d["verified_mask"], d["exact_mask"]
        assert int((~v[np.asarray(d["planar5_mask"], dtype=bool)]).sum()) == 0, (
            "a planar lane was not verified, though `solve` verifies before "
            "reporting convergence")


def test_verification_can_be_switched_off_but_defaults_on(eos, problems):
    import src.physics.exact_flux as EF
    EF = importlib.reload(EF)
    assert EF._VERIFY_EXACT is True
    assert EF._VERIFY_TOL == 1e-8


# ── why a lane was lost (2026-09-29) ─────────────────────────────────────

def test_the_failure_reasons_account_for_every_attempted_lane(eos, problems):
    """`reason7` and `reason5` say why each solver lost a lane.  They are
    diagnostics -- nothing reads them to decide anything -- so what is pinned
    is that they are CONSISTENT with the decisions actually taken: a lane
    answered by the seven-wave path has reason7 == 0, a lane the planar
    rescue answered has reason5 == 0, a lane that fell back has neither, and
    the planar rescue was offered exactly the lanes the seven-wave path lost.
    """
    sL, sR, n = problems
    _run(sL, sR, eos, planar5=True)
    d = dict(LAST_DIAG)
    sel = np.asarray(d["sel"])
    r7, r5 = np.asarray(d["reason7"]), np.asarray(d["reason5"])
    assert sel.size == d["n_attempted"] == r7.size == r5.size
    exact = d["exact_mask"].numpy()[sel]
    p5 = np.asarray(d["planar5_mask"], bool)
    assert set(np.unique(r7)) <= {0, 1, 2, 3, 4, 5}
    assert set(np.unique(r5)) <= {0, 1, 2, 3, 4, 5, 6, 7}
    assert np.array_equal(r5 == 0, p5)                 # planar answered
    assert np.array_equal((r7 == 0) | p5, exact)       # someone answered
    assert np.all(r5[r7 == 0] == 7)                    # solved lanes: not offered
    lost = (r7 != 0) & (r7 != 4)                       # crossing fans are not
    assert np.all(r5[lost] != 7)                       #   offered, the rest are
    it = np.asarray(d["planar_n_iter"])
    assert np.all(it[r5 == 7] == -1) and np.all(it[r5 != 7] >= 0)
    tot = np.asarray(d["seven_n_iter_total"])
    assert tot.size == sel.size and np.all(tot >= 0)


def test_the_planar_helper_is_what_the_rescue_runs(eos, problems):
    """`_planar_and_ray` IS the rescue's body: offered the lanes the rescue
    took, it accepts them all and returns the rescue's own star pressure."""
    sL, sR, n = problems
    F, ps, d = _run(sL, sR, eos, planar5=True)
    import src.physics.exact_flux as EF
    sel = np.asarray(d["sel"])
    took = sel[np.asarray(d["planar5_mask"], bool)]
    if took.size == 0:
        pytest.skip("the planar rescue answered no lane of this fixture")
    (L7, Bn), (R7, _) = (EF.to_solver_frame(sL, eos, 0),
                         EF.to_solver_frame(sR, eos, 0))
    g = lambda t: t.detach().cpu().numpy().astype(float)[took]
    P = EF._batched_parts(GAMMA)
    r = EF._planar_and_ray(P, GAMMA, [g(c) for c in L7], [g(c) for c in R7],
                           g(Bn), accuracy=1e-8, max_iter=MAX_ITER)
    assert r["good"].all() and np.all(r["reason"] == 0)
    assert np.array_equal(r["star"][1], ps.numpy().reshape(-1)[took])


def test_the_ladder_and_the_crossed_offer_are_on_by_default():
    import src.physics.exact_flux as EF
    saved = {k: os.environ.pop(k, None) for k in PINNED}
    try:
        EF = importlib.reload(EF)
        assert EF._PLANAR5_LADDER is True and EF._PLANAR5_CROSSED is True
    finally:
        os.environ.update({k: v for k, v in saved.items() if v is not None})
        importlib.reload(EF)


def test_the_ladder_only_adds(eos, problems):
    """The rungs see only lanes the rescue left unsolved: every flux that was
    exact stays exact and bitwise the same, and whatever is gained names the
    rung that gained it."""
    sL, sR, n = problems
    F0, p0, d0 = _run(sL, sR, eos, planar5=True)
    F1, p1, d1 = _run(sL, sR, eos, planar5=True,
                      extra={"RMHD_PLANAR5_LADDER": "1",
                             "RMHD_PLANAR5_CROSSED": "1"})
    m0 = np.asarray(d0["exact_mask"], bool).reshape(-1)
    m1 = np.asarray(d1["exact_mask"], bool).reshape(-1)
    assert not (m0 & ~m1).any()
    for k in F0:
        a, b = F0[k].reshape(-1).numpy(), F1[k].reshape(-1).numpy()
        assert np.array_equal(a[m0], b[m0])
    assert np.array_equal(p0.reshape(-1).numpy()[m0], p1.reshape(-1).numpy()[m0])
    r0, r1 = np.asarray(d0["planar_rung"]), np.asarray(d1["planar_rung"])
    assert set(np.unique(r0)) <= {-1, 0}
    assert r1.min() >= -1 and r1.max() <= 6
    took = np.zeros(m1.size, bool)
    took[np.asarray(d1["sel"])[r1 >= 0]] = True
    assert np.array_equal(took, np.asarray(d1["planar5_mask"], bool).reshape(-1))
    gained = m1 & ~m0
    assert not (gained & ~took).any()
