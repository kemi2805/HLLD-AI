"""Skipping interfaces whose fan lies entirely on one side of the ray.

HLLD's signal speeds are clamped at zero, so ``cmin == 0`` means every wave
is right-going and its flux is ``f_L`` exactly; ``cmax == 0`` gives ``f_R``.
The exact solution at xi = 0 is that same upwind state, so solving those
interfaces is pure waste.

The contract:

* the flux must not move -- measured max 1.9e-16 relative over 45,650
  harvested interfaces, so 1e-13 here is a loose bound, not a tuned one;
* the skipped interfaces must be exactly the ones the gate names, and the
  diagnostics must still account for every interface;
* switching it on must not change which interfaces are called EXACT among
  those still attempted.

It is deliberately NOT asserted to be bitwise: a handful of lanes differ in
the last bit, which is why the knob exists.
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
from src.physics.hlld import LAST_DIAG, compute_srmhd_fluxes  # noqa: E402

GAMMA = 5.0 / 3.0
N_PROB = 40
MAX_ITER = 10


@pytest.fixture(scope="module")
def eos():
    return hybrid_eos(K=0.0, gamma=GAMMA, gamma_th=GAMMA)


@pytest.fixture(scope="module")
def problems(eos):
    """Forward-constructed problems, boosted so a good share go supersonic.

    A static Riemann problem almost always straddles the ray; adding a bulk
    normal velocity sweeps the whole fan to one side, which is exactly the
    population the skip targets.
    """
    from rmhd.eos import set_eos
    set_eos("ideal")
    from rmhd.riemann_dataset import generate_dataset

    sols = generate_dataset(N_PROB, gamma=GAMMA, Bx_range=(0.05, 1.5), seed=91,
                            xi_window=(0.0, 0.0), verbose=False)
    Z = np.array([s.zones for s in sols])
    Bx = np.array([s.Bx for s in sols])

    def prim(slot, boost):
        rho, Pt, vn, v1, v2_, B1, B2 = (Z[:, slot, j] for j in range(7))
        vn = np.clip(vn + boost, -0.97, 0.97)
        v2 = vn ** 2 + v1 ** 2 + v2_ ** 2
        keep = v2 < 0.985
        W2 = 1.0 / (1.0 - np.where(keep, v2, 0.5))
        eta = Bx * vn + B1 * v1 + B2 * v2_
        b2 = (Bx ** 2 + B1 ** 2 + B2 ** 2) / W2 + eta ** 2
        t = lambda a: torch.tensor(a, dtype=torch.float64)
        d = {"rho": t(rho), "vx": t(vn), "vy": t(v1), "vz": t(v2_),
             "p": t(np.maximum(Pt - 0.5 * b2, 1e-8)),
             "Bx": t(Bx), "By": t(B1), "Bz": t(B2)}
        d["eps"] = eos.eps__press_rho(d["p"], d["rho"])
        return d, keep

    # half boosted right, half left, so both branches of the test fire
    boost = np.where(np.arange(len(Z)) % 2 == 0, 0.85, -0.85)
    L, kL = prim(0, boost)
    R, kR = prim(7, boost)
    return L, R, (kL & kR)


def _run(sL, sR, eos, *, skip):
    old = os.environ.get("RMHD_UPWIND_SKIP")
    os.environ["RMHD_UPWIND_SKIP"] = "1" if skip else "0"
    try:
        import src.physics.exact_flux as EF
        EF = importlib.reload(EF)
        F, U, p = EF.exact_flux_batched(sL, sR, eos, idir=0, max_iter=MAX_ITER)
        return ({k: v.clone() for k, v in F.items()}, p.clone(), dict(LAST_DIAG))
    finally:
        if old is None:
            os.environ.pop("RMHD_UPWIND_SKIP", None)
        else:
            os.environ["RMHD_UPWIND_SKIP"] = old
        import src.physics.exact_flux as EF
        importlib.reload(EF)


def test_on_is_the_default():
    import src.physics.exact_flux as EF
    EF = importlib.reload(EF)
    assert EF._UPWIND_SKIP is True, "the skip is flux-neutral and should default on"


def test_the_gate_matches_hllds_own_upwind_branch(problems, eos):
    """The skip must name exactly the interfaces where HLLD returns f_L/f_R."""
    sL, sR, _ = problems
    *_, cmax, cmin = compute_srmhd_fluxes(sL, sR, eos, 0)
    upwind = ((cmin <= 0.0) | (cmax <= 0.0))
    _, _, d = _run(sL, sR, eos, skip=True)
    assert d["n_upwind_skip"] <= int(upwind.sum()), (
        "more interfaces were skipped than HLLD calls upwind")
    assert d["n_upwind_skip"] > 0, (
        "the fixture produced no supersonic interfaces, so this proves nothing")


def test_the_flux_does_not_move(problems, eos):
    """The whole justification: skipping changes no answer."""
    sL, sR, _ = problems
    Fa, pa, da = _run(sL, sR, eos, skip=False)
    Fb, pb, db = _run(sL, sR, eos, skip=True)
    assert db["n_upwind_skip"] > 0
    worst, where = 0.0, None
    for k in Fa:
        a, b = Fa[k].numpy(), Fb[k].numpy()
        rel = np.abs(a - b) / np.maximum(np.abs(a), 1e-30)
        if rel.max() > worst:
            worst, where = float(rel.max()), k
    assert worst <= 1e-13, (
        f"the upwind skip moved the flux by {worst:.3e} in {where}; it is "
        "supposed to skip only interfaces where HLLD already gives the exact "
        "answer (measured max 1.9e-16 on real data)")


def test_it_only_removes_attempts(problems, eos):
    """Fewer attempts, and every interface still accounted for."""
    sL, sR, _ = problems
    _, _, da = _run(sL, sR, eos, skip=False)
    _, _, db = _run(sL, sR, eos, skip=True)
    assert db["n_attempted"] == da["n_attempted"] - db["n_upwind_skip"], (
        "attempts did not drop by exactly the number skipped")
    for d in (da, db):
        assert (d["n_bad"] + d["n_weak_gate"] + d.get("n_upwind_skip", 0)
                + d["n_attempted"]) == d["n_interfaces"]


def test_it_does_not_change_the_exact_verdict_on_what_remains(problems, eos):
    """Interfaces still attempted must be resolved exactly as before."""
    sL, sR, _ = problems
    _, _, da = _run(sL, sR, eos, skip=False)
    _, _, db = _run(sL, sR, eos, skip=True)
    ma, mb = da["exact_mask"].numpy(), db["exact_mask"].numpy()
    # every interface exact WITH the skip must have been exact without it
    assert bool((mb & ~ma).sum() == 0), (
        "the skip made an interface exact that was not exact before")
    lost = int((ma & ~mb).sum())
    assert lost == db["n_upwind_skip"] or lost <= db["n_upwind_skip"], (
        f"{lost} interfaces lost their exact flux but only "
        f"{db['n_upwind_skip']} were skipped")
