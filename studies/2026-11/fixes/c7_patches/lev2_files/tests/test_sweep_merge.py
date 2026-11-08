"""RMHD_MERGE_SWEEPS: one seven-wave pipeline call for an RK stage's x and y
sweeps changes nothing -- the stage's right-hand side, its CT updates, both
sweeps' diagnostics and the harvest the same, bit for bit.

The 16^2 magnetic rotor's first stage: every y-face is degenerate there
(B_y = 0 at t = 0), so the y-sweep reaches the three-wave branch of the
pipeline and the x-sweep the seven-wave one -- both halves of a merged call.
Measured 2026-11-07 on short runs as well (local, 3-4 steps at 32^2, 4Q /
rotor / Orszag-Tang, production and late-chain switches, in process and over
a pool): restart state, diag.csv and every harvest file identical.
"""
import functools
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.physics.eos import hybrid_eos  # noqa: E402
from src.physics.exact_flux import exact_flux_batched  # noqa: E402
from src.physics.grid import Grid2D  # noqa: E402
from src.physics.harvest import Harvester  # noqa: E402
from src.physics.initial_data2d import b_from_potential  # noqa: E402

pytest.importorskip("rmhd")


def _rotor_state(g, eos):
    from src.physics.driver2d import prims_to_cons_2d, sync_state
    from src.physics.initial_data2d import magnetic_rotor
    from src.physics.state import EVOLVED_KEYS, State2D
    prims, Az = magnetic_rotor(g, eos)
    Bxf, Byf = b_from_potential(Az, g.dx, g.dy)
    cons_all = prims_to_cons_2d(prims, g)
    st = State2D(cons={k: cons_all[k] for k in EVOLVED_KEYS},
                 Bxf=Bxf, Byf=Byf, prims=prims)
    return sync_state(st, g, eos)


def _stage(tmp, merge, monkeypatch):
    from src.physics.driver2d import ct_stage
    monkeypatch.setenv("RMHD_MERGE_SWEEPS", "1" if merge else "0")
    eos = hybrid_eos(K=0.0, gamma=5.0 / 3.0, gamma_th=5.0 / 3.0)
    g = Grid2D(-0.5, 0.5, 16, -0.5, 0.5, 16, ng=2)
    st = _rotor_state(g, eos)
    H = Harvester(str(tmp), 5.0 / 3.0, only_retried=False)
    fn = functools.partial(exact_flux_batched, tau_weak=1e-2, n_retries=2,
                           max_iter=40, harvester=H)
    rhs, dBx, dBy, diag = ct_stage(st, g, eos, flux_fn=fn)
    H.close()
    return rhs, dBx, dBy, diag


def _same(a, b):
    return torch.equal(a, b) or bool(torch.all((a == b) | (torch.isnan(a) & torch.isnan(b))))


def test_merged_stage_is_bitwise_the_two_sweeps(tmp_path, monkeypatch):
    r0, x0, y0, d0 = _stage(tmp_path / "m0", False, monkeypatch)
    r1, x1, y1, d1 = _stage(tmp_path / "m1", True, monkeypatch)
    for k in r0:
        assert _same(r0[k], r1[k]), k
    assert _same(x0, x1) and _same(y0, y1)
    for ax in ("x", "y"):
        a, b = d0[ax]["diag"], d1[ax]["diag"]
        for k in ("n_attempted", "n_exact", "n_verified", "frac_hlle_fallback", "mean_iters",
                  "n_fan_interior"):
            assert (a.get(k) == b.get(k)) or (a.get(k) != a.get(k) and b.get(k) != b.get(k)), (ax, k)
        assert _same(d0[ax]["p_star"], d1[ax]["p_star"])
    for f in sorted((tmp_path / "m0").glob("*.npz")):
        z0, z1 = np.load(f), np.load(tmp_path / "m1" / f.name)
        assert sorted(z0.files) == sorted(z1.files), f.name
        for k in z0.files:
            assert np.array_equal(z0[k], z1[k], equal_nan=z0[k].dtype.kind == "f"), (f.name, k)


def test_off_by_default(monkeypatch):
    from src.physics import sweep_merge
    monkeypatch.delenv("RMHD_MERGE_SWEEPS", raising=False)
    assert not sweep_merge.wanted()
