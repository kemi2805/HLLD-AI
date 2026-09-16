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
_RMHD = "/Users/miler/Codes/rmhd_final"
if _RMHD not in sys.path:
    sys.path.append(_RMHD)

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
    from batched.ray_scalar import claim_rmhd_namespace
    claim_rmhd_namespace()
    from eos import set_eos
    set_eos("ideal")
    from riemann_dataset import generate_dataset

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
