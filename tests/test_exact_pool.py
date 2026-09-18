"""The worker pool (src/physics/exact_pool.py) changes where lanes are solved,
never what they solve to.

The pool splits a sweep's lanes over RMHD_POOL spawned worker processes.  Its
docstring's claim -- a lane's answer does not depend on which worker solved
it, because the batched solver is batch-independent by construction -- is
what every calea run relies on, and it is asserted here: the same interfaces
through a 2-worker pool and in-process give identical fluxes, states and
exact-lane masks.  The workers are fresh interpreters, so this is also the
check that a spawned process finds the installed rmhd and this checkout's src.

Sizing as in test_exact_flux_batched: a small batch and a short Newton
budget.  Both sides get the same budget, so the comparison is exact.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.physics.eos import hybrid_eos                       # noqa: E402
from src.physics.hlld import LAST_DIAG                       # noqa: E402
from src.physics.exact_flux import exact_flux_batched        # noqa: E402

GAMMA = 5.0 / 3.0
N_PROB = 16
MAX_ITER = 10


@pytest.fixture(scope="module")
def interfaces():
    from rmhd.eos import set_eos
    set_eos("ideal")
    from rmhd.riemann_dataset import generate_dataset
    eos = hybrid_eos(K=0.0, gamma=GAMMA, gamma_th=GAMMA)
    sols = generate_dataset(N_PROB, gamma=GAMMA, Bx_range=(0.3, 3.0), seed=31,
                            xi_window=(0.0, 0.0), verbose=False)
    Z = np.array([s.zones for s in sols])
    Bx = np.array([s.Bx for s in sols])

    def prim(slot):
        rho, Pt, vn, v1, v2_, B1, B2 = (Z[:, slot, j] for j in range(7))
        W2 = 1.0 / (1.0 - (vn ** 2 + v1 ** 2 + v2_ ** 2))
        eta = Bx * vn + B1 * v1 + B2 * v2_
        b2 = (Bx ** 2 + B1 ** 2 + B2 ** 2) / W2 + eta ** 2
        t = lambda a: torch.tensor(a, dtype=torch.float64)
        d = {"rho": t(rho), "vx": t(vn), "vy": t(v1), "vz": t(v2_),
             "p": t(Pt - 0.5 * b2), "Bx": t(Bx), "By": t(B1), "Bz": t(B2)}
        d["eps"] = eos.eps__press_rho(d["p"], d["rho"])
        return d

    return prim(0), prim(7), eos


def _solve(interfaces):
    sL, sR, eos = interfaces
    F, U, p_star = exact_flux_batched(sL, sR, eos, idir=0, max_iter=MAX_ITER)
    return F, U, p_star, dict(LAST_DIAG)


def test_pool_equals_in_process(interfaces, monkeypatch):
    from src.physics import exact_pool

    monkeypatch.setenv("RMHD_POOL", "0")
    F0, U0, p0, d0 = _solve(interfaces)

    monkeypatch.setenv("RMHD_POOL", "2")
    monkeypatch.setenv("RMHD_POOL_THREADS", "1")
    monkeypatch.setenv("RMHD_POOL_MIN", "1")        # pool even this small batch
    try:
        F1, U1, p1, d1 = _solve(interfaces)
    finally:
        exact_pool.shutdown()

    assert int(d0["n_exact"]) >= 2, "too few exact lanes to compare anything"
    assert np.array_equal(np.asarray(d0["exact_mask"]), np.asarray(d1["exact_mask"]))
    assert torch.equal(p0, p1)
    for k in F0:
        assert torch.equal(F0[k], F1[k]), f"flux {k} moved under the pool"
        assert torch.equal(U0[k], U1[k]), f"state {k} moved under the pool"
