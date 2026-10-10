"""The continuation rung (RMHD_CONTINUATION, src/physics/continuation.py).

Three things are asserted:

* it never touches a coplanar problem -- every lane there is planar, the
  rung is offered none, and the fluxes are bitwise those without it;
* where it answers, it answers with the seven-wave solver's own root: with
  the main solve starved to two Newton iterations it is offered the lanes
  that lost, and the ones it answers have the star pressure the seven-wave
  solver finds at its full budget -- all but the rare lane where the walk
  follows a branch to ANOTHER verified root. That is a property of the
  continuation, not of the host: on the tilted four-quadrant ledger's
  controls it happened on 1 of 1,187 answers (star pressure 0.4% off, from
  the plain polish after the full walk; calea job 40456), and on this batch
  calea's faces give 1 of 22 (3.6e-3) where the Mac's give none. Both answers
  pass production's acceptance; which one the equations select is the open
  evolutionary-selection question, not something this rung can settle;
* the worker pool changes where it runs, never what it answers.
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


def _faces(seed, n, coplanar_frac):
    from rmhd.eos import set_eos
    set_eos("ideal")
    from rmhd.riemann_dataset import generate_dataset
    eos = hybrid_eos(K=0.0, gamma=GAMMA, gamma_th=GAMMA)
    sols = generate_dataset(n, gamma=GAMMA, Bx_range=(0.05, 2.0), seed=seed,
                            xi_window=(0.0, 0.0), coplanar_frac=coplanar_frac,
                            verbose=False)
    Z = np.array([s.zones for s in sols])
    Bx = np.array([s.Bx for s in sols])

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


def _run(monkeypatch, faces, *, continuation, max_iter, pool=0):
    for k in ("RMHD_PLANAR5_FALLBACK", "RMHD_PLANAR5_LADDER",
              "RMHD_PLANAR5_CROSSED", "RMHD_PLANAR4"):
        monkeypatch.setenv(k, "1")
    monkeypatch.setenv("RMHD_CONTINUATION", "1" if continuation else "0")
    monkeypatch.setenv("RMHD_POOL", str(pool))
    monkeypatch.setenv("RMHD_POOL_THREADS", "1")
    monkeypatch.setenv("RMHD_POOL_MIN", "1")
    import src.physics.exact_flux as EF
    EF = importlib.reload(EF)
    sL, sR, eos = faces
    F, U, p = EF.exact_flux_batched(sL, sR, eos, idir=0, max_iter=max_iter)
    return F, U, p, dict(LAST_DIAG)


@pytest.fixture
def reset(monkeypatch):
    yield
    from src.physics import exact_pool
    import src.physics.exact_flux as EF
    exact_pool.shutdown()
    monkeypatch.undo()
    importlib.reload(EF)


def test_a_coplanar_problem_is_untouched(monkeypatch, reset):
    faces = _faces(95, 16, 1.0)
    F0, U0, p0, d0 = _run(monkeypatch, faces, continuation=False, max_iter=4)
    F1, U1, p1, d1 = _run(monkeypatch, faces, continuation=True, max_iter=4)
    assert int(d1["n_continuation_attempted"]) == 0
    assert np.array_equal(np.asarray(d0["exact_mask"]), np.asarray(d1["exact_mask"]))
    assert torch.equal(p0, p1)
    for k in F0:
        assert torch.equal(F0[k], F1[k]) and torch.equal(U0[k], U1[k]), k


def test_it_answers_with_the_seven_wave_root(monkeypatch, reset):
    faces = _faces(93, 24, 0.0)
    _, _, p2, d2 = _run(monkeypatch, faces, continuation=True, max_iter=2)
    _, _, p40, d40 = _run(monkeypatch, faces, continuation=False, max_iter=40)
    sel = np.asarray(d2["sel"])
    w = sel[np.asarray(d2["continuation_mask"], bool)]
    assert w.size >= 10, "the starved run should hand the rung most lanes"
    both = w[np.asarray(d40["exact_mask"], bool)[w]]
    assert both.size >= 10
    a, b = p2.reshape(-1)[both], p40.reshape(-1)[both]
    rel = ((a - b).abs() / b.abs()).numpy()
    same = rel < 1e-6
    # at most one lane in ten on another verified root (see the docstring)
    assert same.sum() >= 0.9 * both.size, rel
    # and every one the rung answered is verified exact
    assert int(d2["n_verified"]) >= int(d2["n_exact"]) - int(d2.get("n_degenerate_exact", 0))


def test_the_pool_changes_nothing(monkeypatch, reset):
    faces = _faces(93, 24, 0.0)
    F0, U0, p0, d0 = _run(monkeypatch, faces, continuation=True, max_iter=2)
    F1, U1, p1, d1 = _run(monkeypatch, faces, continuation=True, max_iter=2,
                          pool=2)
    assert int(d0["n_continuation_exact"]) >= 10
    for key in ("exact_mask", "continuation_mask", "reason_cont",
                "continuation_n_iter"):
        assert np.array_equal(np.asarray(d0[key]), np.asarray(d1[key])), key
    assert torch.equal(p0, p1)
    for k in F0:
        assert torch.equal(F0[k], F1[k]) and torch.equal(U0[k], U1[k]), k
