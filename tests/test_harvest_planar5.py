"""Harvested five-wave rows must be exact solutions, and tagged as their own.

Two independent things are checked here, because a row can be well-formed and
still wrong:

* the shard SCHEMA -- both record paths write the same keys, including the
  new ``source`` tag, and a second pass does not re-record the first pass's
  successes as failures;
* the row CONTENT -- evaluating the full seven-wave residual at the unknowns
  implied by a harvested planar row must give < 1e-8.  That is the same test
  the reduced three-wave solver fails, and the reason these rows may enter a
  training set at all.
"""
import importlib
import os
import sys
import tempfile
from pathlib import Path

import glob
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_RMHD = "/Users/miler/Codes/rmhd_final"
if _RMHD not in sys.path:
    sys.path.append(_RMHD)

from src.physics.eos import hybrid_eos                       # noqa: E402
from src.physics.hlld import LAST_DIAG                       # noqa: E402
from src.physics.harvest import Harvester                    # noqa: E402

GAMMA = 5.0 / 3.0
N_PROB = 24
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

    sols = generate_dataset(N_PROB, gamma=GAMMA, Bx_range=(0.01, 1.5), seed=91,
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


@pytest.fixture(scope="module")
def shards(eos, problems, tmp_path_factory):
    """Run one sweep with the rescue and harvesting on, and flush."""
    sL, sR, n = problems
    d = tmp_path_factory.mktemp("harvest")
    old = os.environ.get("RMHD_PLANAR5_FALLBACK")
    os.environ["RMHD_PLANAR5_FALLBACK"] = "1"
    try:
        import src.physics.exact_flux as EF
        EF = importlib.reload(EF)
        h = Harvester(str(d), GAMMA, only_retried=False,
                      max_solved=10 ** 6, max_unsolved=10 ** 6)
        EF.exact_flux_batched(sL, sR, eos, idir=0, max_iter=MAX_ITER,
                              harvester=h)
        diag = dict(LAST_DIAG)
        h.close()
    finally:
        if old is None:
            os.environ.pop("RMHD_PLANAR5_FALLBACK", None)
        else:
            os.environ["RMHD_PLANAR5_FALLBACK"] = old
        import src.physics.exact_flux as EF
        importlib.reload(EF)
    files = sorted(glob.glob(str(d / "solved_*.npz")))
    assert files, "no solved shard was written"
    z = [np.load(f) for f in files]
    merged = {k: np.concatenate([q[k] for q in z]) for k in
              ("U_L", "U_R", "zones", "speeds", "attempts", "source")}
    return merged, diag


def test_the_shard_carries_the_source_tag(shards):
    m, diag = shards
    assert set(np.unique(m["source"])) <= {0, 1}
    n5 = int((m["source"] == 1).sum())
    assert n5 == diag["n_planar5_exact"], (
        f"{n5} tagged rows against {diag['n_planar5_exact']} reported")
    assert n5 > 0, "the fixture produced no planar rows to check"


def test_every_row_has_every_field(shards):
    m, _ = shards
    n = m["U_L"].shape[0]
    assert m["U_R"].shape == (n, 8)
    assert m["zones"].shape == (n, 8, 7)
    assert m["speeds"].shape == (n, 7)
    assert m["attempts"].shape == (n,)
    assert m["source"].shape == (n,)
    assert np.isfinite(m["zones"]).all()
    assert np.isfinite(m["speeds"]).all()


def test_the_planar_rows_are_exact_solutions(shards):
    """The content check: the full seven-wave residual at each harvested row.

    A row is only worth training on if it solves the actual Riemann problem.
    """
    m, _ = shards
    five = m["source"] == 1
    if not five.any():
        pytest.skip("no planar rows")
    from batched import fullcontact_b as FB
    Z = m["zones"][five]
    UL, UR = m["U_L"][five], m["U_R"][five]
    left = [UL[:, j].copy() for j in range(7)]
    right = [UR[:, j].copy() for j in range(7)]
    Bn = UL[:, 7].copy()
    wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
    psi = lambda k: np.arctan2(Z[:, k, 6], Z[:, k, 5])
    unk6 = [np.log(np.maximum(Z[:, 1, 1], 1e-30)),
            np.log(np.maximum(np.hypot(Z[:, 3, 5], Z[:, 3, 6]), 1e-30)),
            psi(3),
            np.log(np.maximum(Z[:, 5, 1], 1e-30)),
            wrap(psi(2) - psi(1)),
            wrap(psi(6) - psi(5))]
    fv, _, _, _, err = FB.make_solver(GAMMA)["fullfuncv6"](left, right, unk6, Bn)
    nrm = np.where(err, np.inf, np.max(np.abs(fv), axis=0))
    assert float(np.max(nrm)) < 1e-8, (
        f"a harvested planar row misses the full system by {np.max(nrm):.2e}")


def test_the_planar_rows_have_zero_rotation(shards):
    """The structural signature the tag exists to mark."""
    m, _ = shards
    five = m["source"] == 1
    if not five.any():
        pytest.skip("no planar rows")
    Z = m["zones"][five]
    for a, b in ((1, 2), (5, 6)):            # R2 vs R3, R6 vs R7
        assert np.allclose(Z[:, a, 5], Z[:, b, 5], rtol=0, atol=0)
        assert np.allclose(Z[:, a, 6], Z[:, b, 6], rtol=0, atol=0)


def test_the_second_pass_did_not_refile_the_first(shards):
    """Rows are recorded once; the two passes own disjoint lanes."""
    m, diag = shards
    pair = np.concatenate([m["U_L"], m["U_R"]], axis=1)
    uniq = np.unique(pair, axis=0)
    assert uniq.shape[0] == pair.shape[0], "a row was harvested twice"
    assert m["U_L"].shape[0] == diag["n_exact"], (
        "harvested rows do not match the exact count")
