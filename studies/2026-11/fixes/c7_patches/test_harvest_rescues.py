"""RMHD_HARVEST_RESCUES: the later rungs' answers as solved rows.

With the switch, every solved row carries ``steer`` (k, 2), ``rot`` (k, 2)
and ``stage`` (k): a rescue rung's values where it recorded them, -1 / -1 / 0
for the rows of the seven-wave and five-wave passes, so the shard's columns
line up.  `record_unsolved` writes the lanes a gate refused after `record`
had counted them answered (exact_flux's reclassified lanes) with their own
reason.  Without the switch the solved schema is the old one.
"""
import glob

import numpy as np

from src.physics import harvest as HV
from src.physics.harvest import Harvester


def _lanes(n, seed):
    rng = np.random.default_rng(seed)
    state = lambda: [rng.uniform(0.1, 1.0, n) for _ in range(7)]
    return dict(left=state(), right=state(), Bn=rng.uniform(0.1, 1.0, n),
                zones=[state() for _ in range(6)],
                VsLv=[rng.uniform(-1, 0, n) for _ in range(3)],
                VsRv=[rng.uniform(0, 1, n) for _ in range(3)])


def _record(h, lanes, idx, source, extras=None):
    sub = lambda v: [np.asarray(c)[idx] for c in v]
    k = idx.size
    h.record(sub(lanes["left"]), sub(lanes["right"]), lanes["Bn"][idx],
             cls=np.zeros(k, np.int8), converged=np.ones(k, bool),
             zones=[sub(z) for z in lanes["zones"]],
             VsLv=sub(lanes["VsLv"]), VsRv=sub(lanes["VsRv"]),
             attempts=np.full(k, 2), accepted=np.ones(k, bool), source=source,
             seven_wave=np.ones(k, bool), extras=extras)


def _solved(tmp_path):
    (f,) = glob.glob(str(tmp_path / "solved_*.npz"))
    return np.load(f)


def test_rescue_rows_carry_their_steering_rotation_and_stage(tmp_path, monkeypatch):
    monkeypatch.setattr(HV, "_RESCUE_COLUMNS", True)
    h = Harvester(str(tmp_path), 5.0 / 3.0, only_retried=False)
    lanes = _lanes(6, 0)
    _record(h, lanes, np.arange(3), 0)                       # a seven-wave pass
    steer = np.array([[1, 2], [2, 2]], np.int8)              # (2, k) per side
    rot = np.array([[True, False], [False, False]])
    _record(h, lanes, np.array([4, 5]), 2,
            extras=dict(steer=steer, rot=rot, stage=np.array([3, 1], np.int8)))
    h.flush()
    z = _solved(tmp_path)
    assert z["source"].tolist() == [0, 0, 0, 2, 2]
    assert z["steer"].tolist() == [[-1, -1]] * 3 + [[1, 2], [2, 2]]
    assert z["rot"].tolist() == [[-1, -1]] * 3 + [[1, 0], [0, 0]]
    assert z["stage"].tolist() == [0, 0, 0, 3, 1]


def test_without_the_switch_the_schema_is_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(HV, "_RESCUE_COLUMNS", False)
    h = Harvester(str(tmp_path), 5.0 / 3.0, only_retried=False)
    lanes = _lanes(4, 1)
    _record(h, lanes, np.arange(2), 2,
            extras=dict(steer=np.ones((2, 2), np.int8), rot=np.zeros((2, 2), bool),
                        stage=np.ones(2, np.int8)))
    h.flush()
    assert sorted(_solved(tmp_path).files) == sorted(
        ["U_L", "U_R", "zones", "speeds", "attempts", "source", "gamma"])


def test_record_unsolved_writes_the_reclassified_lanes(tmp_path):
    h = Harvester(str(tmp_path), 5.0 / 3.0, only_retried=False)
    lanes = _lanes(5, 2)
    h.record_unsolved(lanes["left"], lanes["right"], lanes["Bn"], np.array([1, 3]),
                      reason=HV.R_RECLASSIFIED, cls=np.arange(5, dtype=np.int8),
                      attempts=np.full(5, 3))
    h.flush()
    (f,) = glob.glob(str(tmp_path / "unsolved_*.npz"))
    z = np.load(f)
    assert z["reason"].tolist() == [HV.R_RECLASSIFIED] * 2
    assert z["cls"].tolist() == [1, 3]
    assert np.array_equal(z["U_L"][:, 7], lanes["Bn"][[1, 3]])
    assert HV.R_REASONS[HV.R_RECLASSIFIED] == "reclassified"
