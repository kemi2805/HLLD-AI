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
    """COPLANAR problems -- what a 2D run presents, and the only population
    the planar solver claims."""
    from batched.ray_scalar import claim_rmhd_namespace
    claim_rmhd_namespace()
    from eos import set_eos
    set_eos("ideal")
    from riemann_dataset import generate_dataset

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


def _run(sL, sR, eos, *, planar5, tol=None):
    old = {k: os.environ.get(k) for k in
           ("RMHD_PLANAR5_FALLBACK", "RMHD_PLANAR_TOL")}
    os.environ["RMHD_PLANAR5_FALLBACK"] = "1" if planar5 else "0"
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
