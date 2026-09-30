"""The four-unknown planar rung in the flux path: last, and it must ADD only.

Same contract and the same reload idiom as test_planar5_flux.py, whose
fixture this reuses.  The rung sees only the lanes every rescue before it
lost, so every flux that was exact stays exact and bitwise the same; what it
gains it certifies inside `planar4_b.solve` (each shock's jump conditions,
the contact, the fan's order) and the rung marks verified.
"""
import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_planar5_flux import eos, problems, _run, MAX_ITER, GAMMA  # noqa: E402,F401


def test_off_is_the_default():
    import src.physics.exact_flux as EF
    EF = importlib.reload(EF)
    assert EF._PLANAR4 is False


def test_the_rung_only_adds(eos, problems):
    sL, sR, n = problems
    F0, p0, d0 = _run(sL, sR, eos, planar5=True)
    F1, p1, d1 = _run(sL, sR, eos, planar5=True, extra={"RMHD_PLANAR4": "1"})
    m0 = np.asarray(d0["exact_mask"], bool).reshape(-1)
    m1 = np.asarray(d1["exact_mask"], bool).reshape(-1)
    assert not (m0 & ~m1).any()
    for k in F0:
        a, b = F0[k].reshape(-1).numpy(), F1[k].reshape(-1).numpy()
        assert np.array_equal(a[m0], b[m0])
    assert np.array_equal(p0.reshape(-1).numpy()[m0], p1.reshape(-1).numpy()[m0])
    # the rung was offered exactly the lanes nothing else answered
    sel = np.asarray(d1["sel"])
    took = np.asarray(d1["planar4_mask"], bool)
    lost = ~m0[sel]
    assert d1["n_planar4_attempted"] == int(lost.sum())
    assert not (took & ~lost).any()
    assert d1["n_planar4_exact"] == int(took.sum())
    gained = np.zeros(m1.size, bool)
    gained[sel[took]] = True
    assert np.array_equal(gained, m1 & ~m0)
    assert np.asarray(d1["verified_mask"], bool)[took].all()
    # and every offered lane carries a reason; 8 = not offered
    r4 = np.asarray(d1["reason4"])
    assert set(np.unique(r4[lost])) <= set(range(8))
    assert (r4[~lost] == 8).all()
    assert np.array_equal(r4 == 0, took)
    assert (np.asarray(d1["planar4_n_iter"])[lost] >= 0).all()
    assert np.asarray(d1["planar4_kinds"]).shape == (2, sel.size)
    assert np.asarray(d0["reason4"]).size == sel.size and (np.asarray(d0["reason4"]) == 8).all()


def test_off_and_default_are_bitwise(eos, problems):
    sL, sR, n = problems
    F0, p0, d0 = _run(sL, sR, eos, planar5=True)
    F1, p1, d1 = _run(sL, sR, eos, planar5=True, extra={"RMHD_PLANAR4": "0"})
    for k in F0:
        assert np.array_equal(F0[k].numpy(), F1[k].numpy())
    assert np.array_equal(p0.numpy(), p1.numpy())
