"""EnvelopeRecorder: the measured interface envelopes (docs/rotor_envelope.md,
ot_envelope.md) that set rmhd's training ranges came from this class, so its
fields are checked on states whose answers are known by hand."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.physics.envelope import FIELDS, EnvelopeRecorder, summarize  # noqa: E402

COL = {f: i for i, f in enumerate(FIELDS)}


def _state(n, vx=0.6, vy=0.0, Bx=0.3, By=0.4, Bz=0.0, rho=2.0, p=0.5, eps=0.375):
    full = lambda v: torch.full((n,), float(v), dtype=torch.float64)
    return {"rho": full(rho), "p": full(p), "eps": full(eps), "vx": full(vx),
            "vy": full(vy), "vz": full(0.0), "Bx": full(Bx), "By": full(By),
            "Bz": full(Bz)}


def test_fields_on_a_known_state():
    rec = EnvelopeRecorder(n_per_call=5, every=1)
    L = _state(10)
    R = _state(10, rho=1.0, p=0.1)
    rec.record(L, R, axis=0, eos=None)
    row = rec.rows[0][0]
    assert row.shape == (len(FIELDS),) and len(rec.rows[0]) == 5
    assert row[COL["W_L"]] == pytest.approx(1.25)            # v = 0.6
    assert row[COL["lrho_L"]] == pytest.approx(math.log10(2.0))
    assert row[COL["lp_R"]] == pytest.approx(math.log10(0.1))
    assert row[COL["Bn"]] == pytest.approx(0.3)               # |Bx| on an x-sweep
    assert row[COL["Bt_L"]] == pytest.approx(0.4)
    # b^2 = B^2/W^2 + (v.B)^2, h = 1 + eps + p/rho
    b2 = (0.3 ** 2 + 0.4 ** 2) / 1.25 ** 2 + (0.6 * 0.3) ** 2
    assert row[COL["sigma"]] == pytest.approx(b2 / (2.0 * (1 + 0.375 + 0.25)))


def test_normal_follows_the_sweep_axis():
    rec = EnvelopeRecorder(n_per_call=3, every=1)
    s = _state(4, vx=0.0, vy=0.5, Bx=0.3, By=0.4)
    rec.record(s, s, axis=1, eos=None)
    row = rec.rows[0][0]
    assert row[COL["Bn"]] == pytest.approx(0.4)               # By on a y-sweep
    assert row[COL["Bt_L"]] == pytest.approx(0.3)
    # transverse v on a y-sweep is (vz, vx) = 0: cos is defined, not NaN
    assert np.isfinite(row[COL["cos_vB"]])


def test_parallel_transverse_v_and_B_give_cos_one():
    rec = EnvelopeRecorder(n_per_call=2, every=1)
    s = _state(3, vx=0.1, vy=0.3, By=0.6)                     # vt ∥ Bt on x
    rec.record(s, s, axis=0, eos=None)
    assert rec.rows[0][0][COL["cos_vB"]] == pytest.approx(1.0)


def test_every_subsamples_calls_and_save_round_trips(tmp_path):
    rec = EnvelopeRecorder(n_per_call=2, every=2)
    s = _state(3)
    for _ in range(5):
        rec.record(s, s, axis=0, eos=None)
    assert len(rec.rows) == 3                                 # calls 1, 3, 5
    summ = rec.save(str(tmp_path / "env.npz"), run="test")
    z = np.load(tmp_path / "env.npz")
    assert z["samples"].shape == (6, len(FIELDS))
    assert list(z["fields"]) == list(FIELDS) and str(z["run"]) == "test"
    assert summ == summarize(z["samples"])
    assert summ["W_L"]["median"] == pytest.approx(1.25)


def test_keep_states_records_the_same_faces_and_round_trips(tmp_path):
    """keep_states adds the full primitives of the SAME sampled faces: it
    draws nothing from the RNG, so the summary rows are identical, and each
    kept state reproduces its summary row."""
    torch.manual_seed(3)
    n = 50
    L = _state(n)
    R = _state(n, rho=1.0, p=0.1)
    for d in (L, R):
        for k in ("vx", "vy", "By"):
            d[k] = d[k] * torch.rand(n, dtype=torch.float64)
    plain = EnvelopeRecorder(n_per_call=7, every=1, seed=5)
    full = EnvelopeRecorder(n_per_call=7, every=1, seed=5, keep_states=True)
    for ax in (0, 1, 0):
        plain.record(L, R, axis=ax, eos=None)
        full.record(L, R, axis=ax, eos=None)
    assert all(np.array_equal(a, b) for a, b in zip(plain.rows, full.rows))
    full.save(str(tmp_path / "env.npz"))
    z = np.load(tmp_path / "env.npz")
    keys = list(z["state_keys"])
    assert z["states_L"].shape == (21, len(keys)) == z["states_R"].shape
    assert list(z["axis"]) == [0] * 7 + [1] * 7 + [0] * 7
    assert list(z["call"]) == [1] * 7 + [2] * 7 + [3] * 7
    sL, s = z["states_L"], z["samples"]
    assert np.allclose(np.log10(sL[:, keys.index("rho")]), s[:, COL["lrho_L"]])
    bn = np.where(z["axis"] == 0, sL[:, keys.index("Bx")], sL[:, keys.index("By")])
    assert np.allclose(np.abs(bn), s[:, COL["Bn"]])
