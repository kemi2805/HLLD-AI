"""The reduced three-wave rescue: it must ADD lanes and change nothing else.

The fallback is opt-in (``RMHD_3WAVE_FALLBACK``) because it answers an
interface with a solver that neglects the slow waves.  The contract these
tests hold it to is therefore narrow and strict:

* every interface the seven-wave solver owns keeps its flux BIT-FOR-BIT;
* the only interfaces that change are ones that took the fallback flux
  before AND sit inside the ``|B_n|`` gate;
* the diagnostics still account for every interface;
* switching the fallback on does not break batch independence.

Why the gate is on ``|B_n|``: the reduced solver's error is set by the slow
waves it drops, measured against 4000 interfaces with known exact answers as
9.8e-05 (median) below ``|B_n| = 0.03`` and 5.9e-04 below 0.1, rising to
3.8e-03 above 0.3.  See the block comment in ``exact_flux``.
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
    """Forward-constructed problems with a SMALL normal field, so the gate
    admits them; that is the regime the fallback exists for."""
    from batched.ray_scalar import claim_rmhd_namespace
    claim_rmhd_namespace()
    from eos import set_eos
    set_eos("ideal")
    from riemann_dataset import generate_dataset

    sols = generate_dataset(N_PROB, gamma=GAMMA, Bx_range=(0.005, 0.08), seed=17,
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
             "p": t(Pt - 0.5 * b2), "Bx": t(Bx), "By": t(B1), "Bz": t(B2)}
        d["eps"] = eos.eps__press_rho(d["p"], d["rho"])
        return d

    return prim(0), prim(7), len(sols)


def _run(sL, sR, eos, *, fallback, bn_max="0.1"):
    """Reload exact_flux with the switch in the requested state.

    The knobs are read at import so a compiled kernel can never capture a
    stale value; the test therefore reloads rather than poking the module.
    """
    old = (os.environ.get("RMHD_3WAVE_FALLBACK"), os.environ.get("RMHD_BN_3WX_MAX"))
    os.environ["RMHD_3WAVE_FALLBACK"] = "1" if fallback else "0"
    os.environ["RMHD_BN_3WX_MAX"] = bn_max
    try:
        import src.physics.exact_flux as EF
        EF = importlib.reload(EF)
        F, U, p = EF.exact_flux_batched(sL, sR, eos, idir=0, max_iter=MAX_ITER)
        return ({k: v.clone() for k, v in F.items()}, p.clone(), dict(LAST_DIAG))
    finally:
        for k, v in zip(("RMHD_3WAVE_FALLBACK", "RMHD_BN_3WX_MAX"), old):
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        import src.physics.exact_flux as EF
        importlib.reload(EF)


def test_off_is_the_default():
    import src.physics.exact_flux as EF
    EF = importlib.reload(EF)
    assert EF._3WAVE_FALLBACK is False, (
        "the reduced fallback answers with a solver that drops the slow "
        "waves; it must never be on unless asked for")


def test_seven_wave_lanes_are_untouched(eos, problems):
    """A lane the seven-wave solver owns must not move by one bit."""
    sL, sR, n = problems
    F0, p0, d0 = _run(sL, sR, eos, fallback=False)
    F1, p1, d1 = _run(sL, sR, eos, fallback=True)

    m0 = d0["exact_mask"]
    assert torch.equal(p1[m0], p0[m0]), "the rescue moved a seven-wave lane's p*"
    for k in F0:
        assert torch.equal(F1[k][m0], F0[k][m0]), (
            f"the rescue moved a seven-wave lane's flux in {k}")


def test_only_gated_fallback_lanes_change(eos, problems):
    """New exact lanes were fallback before and sit inside the gate."""
    sL, sR, n = problems
    F0, p0, d0 = _run(sL, sR, eos, fallback=False)
    F1, p1, d1 = _run(sL, sR, eos, fallback=True)

    new = d1["exact_mask"] & ~d0["exact_mask"]
    assert not bool((d0["exact_mask"] & ~d1["exact_mask"]).any()), (
        "the rescue LOST a lane that was exact without it")
    assert d1["n_exact"] == d0["n_exact"] + int(new.sum())
    if bool(new.any()):
        bn = sL["Bx"][new].abs()
        assert float(bn.max()) <= 0.1 + 1e-12, (
            "a lane outside the |B_n| gate was rescued")


def test_gate_is_respected(eos, problems):
    """With the gate closed, nothing is attempted and nothing changes."""
    sL, sR, n = problems
    F0, p0, d0 = _run(sL, sR, eos, fallback=False)
    F1, p1, d1 = _run(sL, sR, eos, fallback=True, bn_max="0.0")
    assert d1["n_3wave_attempted"] == 0
    assert d1["n_exact"] == d0["n_exact"]
    assert torch.equal(p1, p0)


def test_diagnostics_account_for_every_interface(eos, problems):
    sL, sR, n = problems
    _, _, d = _run(sL, sR, eos, fallback=True)
    assert d["n_interfaces"] == n
    assert d["n_exact"] + d["n_hlld_fallback"] == n
    assert d["n_3wave_exact"] <= d["n_3wave_attempted"]
    assert int(d["three_wave_mask"].sum()) == d["n_3wave_exact"]


def test_batch_independence_with_the_rescue(eos, problems):
    """Splitting the batch must not move a lane, rescue enabled."""
    sL, sR, n = problems
    F, p, _ = _run(sL, sR, eos, fallback=True)
    half = n // 2
    for sl in (slice(0, half), slice(half, None)):
        a = {k: v[sl] for k, v in sL.items()}
        b = {k: v[sl] for k, v in sR.items()}
        Fp, pp, d = _run(a, b, eos, fallback=True)
        ex = d["exact_mask"]
        assert torch.equal(pp[ex], p[sl][ex])
        for k in F:
            assert torch.equal(Fp[k], F[k][sl]), f"splitting moved {k}"
