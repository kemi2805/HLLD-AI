"""tube_seed.run_tubes: N Riemann problems evolved as the columns of one array.

The columns never exchange data, but they share ONE time step -- the CFL
bound is the maximum over every column -- so a column's profile depends on
its batch exactly when a neighbour has faster waves.  Both halves of that are
pinned here, because the module used to claim nothing coupled the tubes.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.physics.eos import hybrid_eos                        # noqa: E402
from src.physics.tube_seed import run_tubes                   # noqa: E402

GAMMA = 5.0 / 3.0
EOS = hybrid_eos(K=0.0, gamma=GAMMA, gamma_th=GAMMA)
KW = dict(ncells=48, tend=0.04)

# solver-frame states (rho, P_tot, vx, vy, vz, By, Bz): a Brio-Wu-like tube,
# a quiet uniform one, and a violent one with faster waves
BW_L, BW_R = [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.125, 0.6, 0.0, 0.0, 0.0, -1.0, 0.0]
QUIET = [1.0, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0]
HOT_L, HOT_R = [1.0, 30.0, 0.5, 0.0, 0.0, 2.0, 0.0], [0.1, 0.1, -0.5, 0.0, 0.0, -2.0, 0.0]


def _run(pairs, Bn):
    UL = np.array([p[0] for p in pairs]); UR = np.array([p[1] for p in pairs])
    prof, t = run_tubes(UL, UR, np.array(Bn, float), EOS, **KW)
    return prof, t


def _same_column(a, b, ia, ib):
    return all(torch.equal(a[k][:, ia], b[k][:, ib]) for k in a)


def test_a_slower_neighbour_leaves_a_column_untouched():
    alone, t1 = _run([(BW_L, BW_R)], [0.5])
    paired, t2 = _run([(QUIET, QUIET), (BW_L, BW_R)], [0.0, 0.5])
    assert t1 == t2
    assert _same_column(alone, paired, 0, 1)


def test_a_faster_neighbour_changes_the_step_and_so_the_column():
    alone, _ = _run([(BW_L, BW_R)], [0.5])
    paired, _ = _run([(BW_L, BW_R), (HOT_L, HOT_R)], [0.5, 0.5])
    assert not _same_column(alone, paired, 0, 0)


def test_a_uniform_state_stays_uniform():
    prof, _ = _run([(QUIET, QUIET)], [0.3])
    for k in ("rho", "p", "vx", "By"):
        col = prof[k][:, 0].numpy()
        assert np.ptp(col) < 1e-12, f"{k} drifted by {np.ptp(col):.2e}"
