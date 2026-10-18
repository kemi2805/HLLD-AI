"""
The 2D hook for the p* network: run_2d.py --solver hlld_ai.

* a tube run along y with hlld_ai equals the same tube along x (the
  rotation test test_sweep2d.py runs for hlle/hlld), at fixed dt;
* run_2d.py with --solver hlld_ai runs a rotor, loads the network once,
  stamps it into run_meta.json, and the y-sweeps of the rotor's first step
  (B_y = 0 everywhere) go to HLLE for B_n = 0;
* --face-stats only observes: a plain hlld rotor run is identical to the
  bit with and without it.

The network is a small randomly initialised PressureNet with v3 stats
(pressure reference, asinh): what matters is the plumbing, not the weights.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.labelling import label, make_eos, to_prims  # noqa: E402
from src.data.sampling import sample_pairs  # noqa: E402
from src.models.network import PressureNet  # noqa: E402
from src.physics import ai_features as ai  # noqa: E402
from src.physics.grid import Grid2D  # noqa: E402
from src.physics.hlld import hlld_ai_flux  # noqa: E402
from src.physics.initial_data2d import shocktube_2d  # noqa: E402
from test_sweep2d import GAMMA, NG, ST1_L, ST1_R, cyc_state, evolve  # noqa: E402

HIDDEN = [16, 16]


def tiny_model(tmp, gamma=GAMMA, seed=0):
    """A random PressureNet + v3 stats (ref, asinh), written to tmp."""
    v = ai.FEATURE_VERSION_V3
    eos = make_eos(gamma)
    S = sample_pairs(3000, None, np.random.default_rng(seed), gamma)
    sL, sR = to_prims(S["L"], eos), to_prims(S["R"], eos)
    lab = label(sL, sR, eos, versions=(v,))
    ok = lab["valid"]
    X = torch.from_numpy(lab["X_" + v][ok])
    p_ref = ai.p_ref_from_prims(sL, sR, eos)[torch.from_numpy(ok)]
    Xs = ai.scale_by_ref(X, v, p_ref)
    scale = np.median(np.abs(Xs.numpy()), axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    Xt = torch.asinh(Xs / torch.as_tensor(scale))
    y = np.log10(lab["p_star"][ok] / p_ref.numpy())
    stats = dict(feature_version=np.array(v), x_transform=np.array("asinh"),
                 ref=np.array("ptot_avg"), x_scale=scale,
                 mu=Xt.mean(0).numpy(), sigma=Xt.std(0).numpy() + 1e-12,
                 y_mu=np.array(y.mean()), y_sigma=np.array(y.std() + 1e-12))
    torch.manual_seed(seed)
    net = PressureNet(n_input=ai.n_features(v), hidden=HIDDEN, activation="silu")
    ck, st = os.path.join(tmp, "tiny.pt"), os.path.join(tmp, "tiny_norm_stats.npz")
    torch.save(net.state_dict(), ck)
    np.savez(st, **stats)
    net.eval()
    return net, stats, ck, st


def test_rotated_shocktube_hlld_ai(tmp_path):
    """hlld_ai along y == hlld_ai along x after relabelling (fixed dt)."""
    net, stats, _, _ = tiny_model(str(tmp_path))
    eos = make_eos(GAMMA)
    f = lambda sL, sR, e, idir=0: hlld_ai_flux(sL, sR, e, net, stats, idir=idir)
    n, nsteps, dt = 32, 12, 3.0e-3
    gx = Grid2D(0.0, 1.0, n, 0.0, 1.0, n, ng=NG)
    px, _ = shocktube_2d(gx, eos, ST1_L, ST1_R, direction="x", interface=0.5)
    _, px = evolve(gx, eos, px, nsteps, dt=dt, flux_fn=f, limiter="mc")
    gy = Grid2D(0.0, 1.0, n, 0.0, 1.0, n, ng=NG)
    py, _ = shocktube_2d(gy, eos, cyc_state(ST1_L), cyc_state(ST1_R),
                         direction="y", interface=0.5)
    _, py = evolve(gy, eos, py, nsteps, dt=dt, flux_fn=f, limiter="mc")
    py_back = cyc_state(py, 2)
    ph = gx.phys
    worst = 0.0
    for k in ("rho", "p", "vx", "vy", "vz", "Bx", "By", "Bz"):
        a, b = px[k][ph], py_back[k][ph].T
        worst = max(worst, float((a - b).abs().max())
                    / max(float(a.abs().max()), 1e-30))
    assert worst < 1e-12, f"y-run differs from x-run by {worst:.3e}"


def _run(tmp, *args):
    env = dict(os.environ, PYTHONPATH=str(ROOT), TORCH_THREADS="1",
               OMP_NUM_THREADS="1")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_2d.py"),
                        "--problem", "rotor", "--n", "16", "--tend", "0.03",
                        "--nsnap", "2", "--restart-every", "0",
                        "--log-every", "1", *args],
                       cwd=tmp, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return r


def test_run_2d_hlld_ai_rotor(tmp_path):
    _, _, ck, st = tiny_model(str(tmp_path), gamma=5.0 / 3.0)
    out = tmp_path / "ai"
    _run(tmp_path, "--solver", "hlld_ai", "--ai-checkpoint", ck,
         "--ai-norm-stats", st, "--ai-hidden", "16,16", "--face-stats",
         "--out", str(out))
    meta = json.load(open(out / "run_meta.json"))
    assert meta["solver"] == "hlld_ai" and meta["ai"]["mode"] == "oneshot"
    assert meta["ai"]["feature_version"] == ai.FEATURE_VERSION_V3
    assert meta["ai"]["ref"] == "ptot_avg"
    fs = json.load(open(out / "face_stats.json"))["totals"]
    for ax in ("axis0", "axis1"):
        t = fs[ax]
        assert t["faces"] == t["used"] + t["hlle_bn_zero"] + t["hlle_wave"] + t["hlle_other"]
    # the rotor starts with B_y = 0: its first y-sweeps have B_n = 0 everywhere
    rows = np.genfromtxt(out / "face_stats.csv", delimiter=",", names=True)
    first_y = rows[(rows["step"] == 1) & (rows["axis"] == 1)][0]
    assert first_y["hlle_bn_zero"] == first_y["faces"]
    assert fs["axis0"]["hlle_other"] == 0          # one-shot: no root-find
    snap = np.load(out / "snap_fin.npz")
    assert np.isfinite(snap["rho"]).all()
    # the warm start goes through the same hook
    _run(tmp_path, "--solver", "hlld_ai", "--ai-checkpoint", ck,
         "--ai-norm-stats", st, "--ai-hidden", "16,16", "--ai-mode",
         "warmstart", "--face-stats", "--out", str(tmp_path / "ws"))
    assert json.load(open(tmp_path / "ws" / "run_meta.json"))["ai"]["mode"] == "warmstart"


def test_face_stats_only_observes(tmp_path):
    """A plain hlld rotor: identical to the bit with and without --face-stats."""
    _run(tmp_path, "--solver", "hlld", "--out", str(tmp_path / "plain"))
    _run(tmp_path, "--solver", "hlld", "--face-stats",
         "--out", str(tmp_path / "tally"))
    a, b = tmp_path / "plain", tmp_path / "tally"
    assert (a / "diag.csv").read_bytes() == (b / "diag.csv").read_bytes()
    for f in sorted(p.name for p in a.glob("snap_*.npz")):
        za, zb = np.load(a / f), np.load(b / f)
        for k in za.files:
            x, y = za[k], zb[k]
            if x.dtype.kind == "f":
                assert np.array_equal(x.view(np.uint64), y.view(np.uint64)), (f, k)
            else:
                assert np.array_equal(x, y), (f, k)
    fs = json.load(open(b / "face_stats.json"))["totals"]
    assert fs["axis1"]["hlle_bn_zero"] > 0
