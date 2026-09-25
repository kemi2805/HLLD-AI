"""The high-order reconstructions in 2D, and the provenance that makes a run
readable afterwards.

`tests/test_reconstruction_high_order.py` measures the schemes themselves --
order, overshoot, smooth extrema -- in one dimension.  What is untested until
here is everything the 2D machinery adds around them: wider ghost zones,
constrained transport reading a face the stencil has to reach, a restart file
whose arrays are a different size, and the per-face record of which solver
answered which interface.

The ghost-tightness test is the one that licenses ng = 3 and 4 rather than
assuming them: `ghosts_needed` claims that a physical face's reconstruction
never touches the PLM edge fallback, and that is checkable exactly, by
widening the array and demanding the same numbers to the bit.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.physics.ct import div_b
from src.physics.driver2d import prims_to_cons_2d, rk_step_ct, sync_state
from src.physics.eos import hybrid_eos
from src.physics.grid import Grid2D
from src.physics.harvest import Harvester
from src.physics.initial_data2d import b_from_potential, magnetic_rotor
from src.physics.reconstruction import ghosts_needed, reconstruct
from src.physics.state import EVOLVED_KEYS, State2D

HIGH = ("mp5", "mp7")
ROOT = Path(__file__).resolve().parents[1]


def _rotor(n, limiter):
    eos = hybrid_eos(K=0.0, gamma=5.0 / 3.0, gamma_th=5.0 / 3.0)
    g = Grid2D(-0.5, 0.5, n, -0.5, 0.5, n, ng=ghosts_needed(limiter))
    prims, Az = magnetic_rotor(g, eos)
    Bxf, Byf = b_from_potential(Az, g.dx, g.dy)
    c = prims_to_cons_2d(prims, g)
    st = State2D(cons={k: c[k] for k in EVOLVED_KEYS}, Bxf=Bxf, Byf=Byf,
                 prims=prims)
    return g, eos, sync_state(st, g, eos, "outflow", "outflow")


@pytest.mark.parametrize("limiter", ("mc",) + HIGH)
def test_divb_stays_at_machine_precision(limiter):
    """Constrained transport must not care how wide the stencil is.

    The corner EMF reaches one cell beyond the physical block in the
    TRANSVERSE index, where nothing is reconstructed, so ng = 3 or 4 is
    sufficient for it -- but that is an argument, and div(B) is the
    measurement.
    """
    g, eos, st = _rotor(32, limiter)
    dt = 2.0e-3
    for _ in range(6):
        st, _ = rk_step_ct(st, g, eos, dt, scheme="rk3", bc_x="outflow",
                           bc_y="outflow", limiter=limiter)
    worst = float(div_b(st.Bxf, st.Byf, g.dx, g.dy)[g.phys].abs().max())
    assert worst < 1e-10, (limiter, worst)


@pytest.mark.parametrize("limiter", HIGH)
def test_ghosts_needed_is_tight(limiter):
    """A physical face reconstructed with the minimum ghost count must equal
    the same face on a wider array, to the bit.

    If it does not, `ghosts_needed` is too small and the outermost physical
    faces are quietly taking the PLM edge fallback -- a boundary-order bug
    that no norm would show clearly.
    """
    torch.manual_seed(0)
    g = ghosts_needed(limiter)
    n = 40
    for axis in (0, 1):
        base = torch.rand(n + 2 * g, 9, dtype=torch.float64)
        if axis == 1:
            base = base.T.contiguous()
        wide_shape = list(base.shape)
        wide_shape[axis] = base.shape[axis] + 4
        wide = torch.rand(*wide_shape, dtype=torch.float64)
        sl = [slice(None)] * 2
        sl[axis] = slice(2, 2 + base.shape[axis])
        wide[tuple(sl)] = base
        L0, R0 = reconstruct(base, axis=axis, limiter=limiter)
        L1, R1 = reconstruct(wide, axis=axis, limiter=limiter)
        # physical faces: g .. n+g inclusive, shifted by 2 on the wide array
        f = [slice(None)] * 2
        f[axis] = slice(g, g + n + 1)
        fw = [slice(None)] * 2
        fw[axis] = slice(g + 2, g + 2 + n + 1)
        assert torch.equal(L0[tuple(f)], L1[tuple(fw)]), (limiter, axis, "L")
        assert torch.equal(R0[tuple(f)], R1[tuple(fw)]), (limiter, axis, "R")


@pytest.mark.parametrize("limiter", HIGH)
def test_a_restart_reproduces_an_uninterrupted_run(tmp_path, limiter):
    """Four steps must equal two plus two through restart.npz, to the bit.

    The 512^2 reference run restarts several times; if this is not exact it
    is not a reproducible object.
    """
    common = [sys.executable, "scripts/run_2d.py", "--problem", "rotor",
              "--n", "24", "--solver", "hlld", "--limiter", limiter,
              "--nsnap", "2", "--cfl", "0.25"]
    whole = tmp_path / "whole"
    part = tmp_path / "part"
    run = lambda args: subprocess.run(args, cwd=ROOT, capture_output=True,
                                      text=True, timeout=900)
    r = run(common + ["--max-steps", "4", "--out", str(whole)])
    assert r.returncode == 0, r.stdout + r.stderr
    r = run(common + ["--max-steps", "2", "--out", str(part)])
    assert r.returncode == 0, r.stdout + r.stderr
    r = run(common + ["--max-steps", "2", "--out", str(part),
                      "--restart", str(part / "restart.npz")])
    assert r.returncode == 0, r.stdout + r.stderr
    A = np.load(whole / "restart.npz")
    B = np.load(part / "restart.npz")
    assert float(A["t"]) == float(B["t"])
    for k in ["Bxf", "Byf"] + ["cons_" + c for c in EVOLVED_KEYS]:
        assert np.array_equal(A[k], B[k]), (limiter, k)


def test_a_restart_refuses_the_wrong_limiter(tmp_path):
    """Resuming an mp5 run as weno5z has the same ng and would run happily
    with the wrong scheme -- the stamp is what stops it."""
    out = tmp_path / "r"
    run = lambda args: subprocess.run(args, cwd=ROOT, capture_output=True,
                                      text=True, timeout=900)
    base = [sys.executable, "scripts/run_2d.py", "--problem", "rotor", "--n",
            "24", "--solver", "hlld", "--nsnap", "2", "--out", str(out)]
    r = run(base + ["--limiter", "mp5", "--max-steps", "2"])
    assert r.returncode == 0, r.stdout + r.stderr
    r = run(base + ["--limiter", "weno5z", "--max-steps", "1",
                    "--restart", str(out / "restart.npz")])
    assert r.returncode != 0 and "limiter" in (r.stdout + r.stderr)


def test_the_run_is_stamped(tmp_path):
    """Snapshots and run_meta.json must say which scheme made them, or a
    high-order run can be analysed with PLM faces and nobody sees it."""
    import json
    out = tmp_path / "s"
    r = subprocess.run([sys.executable, "scripts/run_2d.py", "--problem",
                        "rotor", "--n", "24", "--solver", "hlld", "--limiter",
                        "mp7", "--max-steps", "1", "--nsnap", "2", "--out",
                        str(out)], cwd=ROOT, capture_output=True, text=True,
                       timeout=900)
    assert r.returncode == 0, r.stdout + r.stderr
    meta = json.load(open(out / "run_meta.json"))
    assert meta["limiter"] == "mp7" and meta["ng"] == 4 and meta["n"] == 24
    z = np.load(out / "snap_000.npz")
    assert str(z["limiter"]) == "mp7" and int(z["ng"]) == 4


def test_coverage_records_which_solver_answered(tmp_path):
    """The per-face provenance round-trips: what goes into record_coverage
    comes back out of the shard, whether the mask covers the whole sweep or
    only the attempted subset."""
    h = Harvester(str(tmp_path), gamma=5.0 / 3.0)
    n = 40
    attempted = np.array([3, 5, 7, 11, 13], dtype=np.int64)
    exact = np.array([5, 7, 13], dtype=np.int64)
    routed = np.array([21], dtype=np.int64)
    weak = np.zeros(n, dtype=bool)
    weak[[0, 1, 2]] = True
    # over the attempted subset: the planar solver answered lanes 5 and 13
    planar5 = np.array([False, True, False, False, True])
    h.record_coverage(0, n, attempted, exact, routed, planar5=planar5,
                      weak=weak)
    h.flush()
    f = sorted(tmp_path.glob("coverage_*.npz"))
    assert len(f) == 1
    z = np.load(f[0])
    unpack = lambda k: np.unpackbits(z[k][0])[:n].astype(bool)
    assert np.array_equal(np.flatnonzero(unpack("attempted")), attempted)
    assert np.array_equal(np.flatnonzero(unpack("exact")), exact)
    assert np.array_equal(np.flatnonzero(unpack("routed")), routed)
    assert np.array_equal(np.flatnonzero(unpack("planar5")), [5, 13])
    assert np.array_equal(np.flatnonzero(unpack("weak")), [0, 1, 2])
    assert not unpack("seven").any()          # never passed: all false


def test_coverage_rejects_a_mask_of_the_wrong_length(tmp_path):
    h = Harvester(str(tmp_path), gamma=5.0 / 3.0)
    with pytest.raises(ValueError):
        h.record_coverage(0, 40, np.array([1, 2]), np.array([1]),
                          seven=np.zeros(7, dtype=bool))
