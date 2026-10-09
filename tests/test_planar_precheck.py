"""The planarity pre-check (RMHD_PLANAR_PRECHECK, exact_flux) skips the
planar rungs on lanes whose input is not planar. None of those lanes could
have been accepted -- every planar rung refuses planar_resid > _PLANAR_TOL,
with the same `planar5_b.to_planar` -- so the check must change no flux, no
state, no star pressure and no exact-lane mask, only the work done and the
recorded refusal reason ("not planar").

The batch mixes non-coplanar interfaces (where the check skips) with
coplanar ones (where it must not), and runs the seven-wave solve on a short
Newton budget so that it loses lanes and the rescues are offered.
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.physics.eos import hybrid_eos                       # noqa: E402
from src.physics.hlld import LAST_DIAG                       # noqa: E402

GAMMA = 5.0 / 3.0
MAX_ITER = 4


def _states(seed, n, coplanar_frac):
    from rmhd.eos import set_eos
    set_eos("ideal")
    from rmhd.riemann_dataset import generate_dataset
    sols = generate_dataset(n, gamma=GAMMA, Bx_range=(0.05, 2.0), seed=seed,
                            xi_window=(0.0, 0.0), coplanar_frac=coplanar_frac,
                            verbose=False)
    return np.array([s.zones for s in sols]), np.array([s.Bx for s in sols])


@pytest.fixture(scope="module")
def interfaces():
    eos = hybrid_eos(K=0.0, gamma=GAMMA, gamma_th=GAMMA)
    Z1, B1 = _states(91, 16, 0.0)          # non-coplanar
    Z2, B2 = _states(92, 16, 1.0)          # coplanar
    Z, Bx = np.concatenate([Z1, Z2]), np.concatenate([B1, B2])

    def prim(slot):
        rho, Pt, vn, v1, v2_, Bt1, Bt2 = (Z[:, slot, j] for j in range(7))
        W2 = 1.0 / (1.0 - (vn ** 2 + v1 ** 2 + v2_ ** 2))
        eta = Bx * vn + Bt1 * v1 + Bt2 * v2_
        b2 = (Bx ** 2 + Bt1 ** 2 + Bt2 ** 2) / W2 + eta ** 2
        t = lambda a: torch.tensor(a, dtype=torch.float64)
        d = {"rho": t(rho), "vx": t(vn), "vy": t(v1), "vz": t(v2_),
             "p": t(Pt - 0.5 * b2), "Bx": t(Bx), "By": t(Bt1), "Bz": t(Bt2)}
        d["eps"] = eos.eps__press_rho(d["p"], d["rho"])
        return d

    return prim(0), prim(7), eos


def _run(monkeypatch, interfaces, precheck):
    for k in ("RMHD_PLANAR5_FALLBACK", "RMHD_PLANAR5_LADDER",
              "RMHD_PLANAR5_CROSSED", "RMHD_PLANAR4"):
        monkeypatch.setenv(k, "1")
    monkeypatch.setenv("RMHD_POOL", "0")
    monkeypatch.setenv("RMHD_PLANAR_PRECHECK", "1" if precheck else "0")
    import src.physics.exact_flux as EF
    EF = importlib.reload(EF)
    sL, sR, eos = interfaces
    F, U, p = EF.exact_flux_batched(sL, sR, eos, idir=0, max_iter=MAX_ITER)
    return F, U, p, dict(LAST_DIAG)


def test_the_precheck_changes_no_answer(monkeypatch, interfaces):
    import src.physics.exact_flux as EF
    try:
        F0, U0, p0, d0 = _run(monkeypatch, interfaces, precheck=False)
        F1, U1, p1, d1 = _run(monkeypatch, interfaces, precheck=True)
    finally:
        monkeypatch.undo()
        importlib.reload(EF)

    # the batch exercises both sides of the check
    assert int(d1["n_planar_skipped_nonplanar"]) >= 1, "no lane was skipped"
    assert int(d0["n_planar5_attempted"]) > int(d1["n_planar5_attempted"])
    assert int(np.asarray(d1["planar5_mask"], bool).sum()) >= 1, \
        "no coplanar lane was rescued -- the batch does not test that side"

    for key in ("exact_mask", "planar5_mask", "planar4_mask", "planar_rung"):
        assert np.array_equal(np.asarray(d0[key]), np.asarray(d1[key])), key
    assert torch.equal(p0, p1)
    for k in F0:
        assert torch.equal(F0[k], F1[k]), f"flux {k} moved under the pre-check"
        assert torch.equal(U0[k], U1[k]), f"state {k} moved under the pre-check"

    # a skipped lane is recorded as not planar, and only those differ
    r5_0, r5_1 = np.asarray(d0["reason5"]), np.asarray(d1["reason5"])
    moved = r5_0 != r5_1
    assert np.all(r5_1[moved] == 2), "a lane changed reason to something else"
    assert np.all(np.isin(r5_0[moved], (1, 2))), \
        "a lane the old path refused for another reason was skipped"
