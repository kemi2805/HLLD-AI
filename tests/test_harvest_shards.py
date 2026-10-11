"""A harvest shard holds whole exact_flux calls.

One call records its seven-wave lanes and then, in a second `record`, the
planar rescue's answers to some of the same faces.  The harvester used to
flush inside `record` as soon as a buffer was full, which could put a face's
unsolved row in one shard and its planar-solved twin in the next -- and a
reader matching twins within a shard then counted the face as lost (on the
gate-open Orszag-Tang harvest: 114,542 "lost" against 38,411 in the coverage).
The flush now waits for the call's `record_coverage`.
"""
import glob

import numpy as np

from src.physics.harvest import Harvester


def _lanes(n, seed):
    rng = np.random.default_rng(seed)
    state = lambda: [rng.uniform(0.1, 1.0, n) for _ in range(7)]
    return dict(left=state(), right=state(), Bn=rng.uniform(0.1, 1.0, n),
                zones=[state() for _ in range(6)],
                VsLv=[rng.uniform(-1, 0, n) for _ in range(3)],
                VsRv=[rng.uniform(0, 1, n) for _ in range(3)])


def _record(h, lanes, idx, accepted, source):
    """`record` restricted to the lanes `idx`, as exact_flux's planar pass is."""
    sub = lambda v: [np.asarray(c)[idx] for c in v]
    k = idx.size
    h.record(sub(lanes["left"]), sub(lanes["right"]), lanes["Bn"][idx],
             cls=np.zeros(k, np.int8), converged=np.ones(k, bool),
             zones=[sub(z) for z in lanes["zones"]],
             VsLv=sub(lanes["VsLv"]), VsRv=sub(lanes["VsRv"]),
             attempts=np.full(k, 2), accepted=accepted, source=source,
             seven_wave=np.ones(k, bool))


def _call(h, lanes, n, rescued):
    """One exact_flux call: the seven-wave pass (lanes 0, 1 accepted), the
    planar pass over the `rescued` lanes, then the coverage."""
    acc = np.zeros(n, bool)
    acc[:2] = True
    _record(h, lanes, np.arange(n), acc, 0)
    _record(h, lanes, rescued, np.ones(rescued.size, bool), 1)
    exact = np.r_[0, 1, rescued]
    h.record_coverage(0, n, np.arange(n), exact,
                      seven=acc, planar5=np.isin(np.arange(n), rescued))


def _key(z, i):
    return z["U_L"][i].tobytes() + z["U_R"][i].tobytes()


def test_nothing_is_written_inside_a_call(tmp_path):
    h = Harvester(str(tmp_path), 5.0 / 3.0, only_retried=False, flush_every=4)
    lanes = _lanes(8, 0)
    acc = np.zeros(8, bool)
    acc[:2] = True
    _record(h, lanes, np.arange(8), acc, 0)        # 6 unsolved >= 4: full
    assert not glob.glob(str(tmp_path / "*.npz")), "flushed inside the call"
    _record(h, lanes, np.array([2, 3, 4]), np.ones(3, bool), 1)
    assert not glob.glob(str(tmp_path / "*.npz")), "flushed inside the call"
    h.record_coverage(0, 8, np.arange(8), np.arange(5))
    assert len(glob.glob(str(tmp_path / "unsolved_*.npz"))) == 1


def test_a_rescued_face_and_its_twin_share_a_shard(tmp_path):
    h = Harvester(str(tmp_path), 5.0 / 3.0, only_retried=False, flush_every=4)
    for seed in range(3):
        _call(h, _lanes(8, seed), 8, np.array([2, 3, 4]))
    h.close()
    shards = sorted(glob.glob(str(tmp_path / "unsolved_*.npz")))
    assert len(shards) == 3
    for fu in shards:
        u = np.load(fu)
        s = np.load(fu.replace("unsolved_", "solved_"))
        c = np.load(fu.replace("unsolved_", "coverage_"))
        twins = {_key(s, i) for i in np.flatnonzero(s["source"] == 1)}
        lost = [i for i in range(len(u["U_L"])) if _key(u, i) not in twins]
        # 6 unsolved per call, 3 of them rescued: 3 lost, as the coverage says
        assert len(lost) == 3
        n = int(c["n"][0])
        att = np.unpackbits(c["attempted"][0])[:n].sum()
        ex = np.unpackbits(c["exact"][0])[:n].sum()
        assert att - ex == len(lost)
