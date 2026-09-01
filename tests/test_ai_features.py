"""
The p* feature vector: one definition, shared by training and inference.

The generator (src/data/generate.py) and the inference path
(hlld.hlld_ai_flux) each used to build this vector by hand.  Two copies of a
15-term expression is how the feature set stayed pinned to the x direction:
fixing one could not fix the other, and nothing compared them.  These tests
assert the properties that make a single shared definition worth having.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.physics import ai_features as ai
from src.physics.eos import hybrid_eos
from src.physics.hlld import compute_srmhd_fluxes, energy_form

GAMMA = 5.0 / 3.0
VEC = [("vx", "vy", "vz"), ("Bx", "By", "Bz")]


def cyc(d, n=1):
    out = dict(d)
    for tri in VEC:
        if tri[0] in d:
            for i, k in enumerate(tri):
                out[k] = d[tri[(i - n) % 3]]
    return out


def states(n=3000, seed=7):
    torch.manual_seed(seed)
    eos = hybrid_eos(K=0.0, gamma=GAMMA, gamma_th=GAMMA)

    def one():
        s = {
            "rho": 10.0 ** torch.empty(n).uniform_(-1, 1).double(),
            "p": 10.0 ** torch.empty(n).uniform_(-2, 1).double(),
            "vx": torch.empty(n).uniform_(-0.6, 0.6).double(),
            "vy": torch.empty(n).uniform_(-0.5, 0.5).double(),
            "vz": torch.empty(n).uniform_(-0.5, 0.5).double(),
            "Bx": torch.empty(n).uniform_(-2, 2).double(),
            "By": torch.empty(n).uniform_(-2, 2).double(),
            "Bz": torch.empty(n).uniform_(-2, 2).double(),
        }
        s["eps"] = eos.eps__press_rho(s["p"], s["rho"])
        return s

    sL, sR = one(), one()
    for k in ("Bx", "By", "Bz"):
        sR[k] = sL[k]          # normal field is continuous across a face
    return eos, sL, sR


def feats(eos, sL, sR, idir):
    uL, uR, fL, fR, cmax, cmin = compute_srmhd_fluxes(sL, sR, eos, idir)
    UL, FL = energy_form(uL, fL)
    UR, FR = energy_form(uR, fR)
    RL, RR = ai.r_vectors(UL, FL, UR, FR, cmin, cmax)
    return ai.build_pstar_features(sL, sR, uL, uR, RL, RR, cmin, cmax, idir)


def test_feature_count_and_version():
    assert ai.N_FEATURES == 15 == len(ai.FEATURE_NAMES)
    assert isinstance(ai.FEATURE_VERSION, str) and ai.FEATURE_VERSION


@pytest.mark.parametrize("n_cyc", [1, 2], ids=["idir1", "idir2"])
def test_features_are_rotation_invariant(n_cyc):
    """The SAME feature vector for a problem and its rotation.

    This is what makes the trained network reusable for y-sweeps without
    retraining: the transverse quantities enter only as magnitudes, so the
    feature set already respects rotations about the normal.
    """
    eos, sL, sR = states()
    X0 = feats(eos, sL, sR, 0)
    Xn = feats(eos, cyc(sL, n_cyc), cyc(sR, n_cyc), n_cyc)
    rel = float((X0 - Xn).abs().max() / X0.abs().max())
    assert rel < 1e-13, f"features not rotation invariant: {rel:.3e}"


def test_features_are_invariant_under_B_flip():
    """SRMHD is invariant under B -> -B, and so must the features be.

    Every feature except a raw signed B_n is already invariant; feeding
    |B_n| completes it.  This is why the existing checkpoint -- trained with
    Bx > 0 only -- stays valid where the normal field is negative, which
    happens on every y-sweep and throughout the rotor.
    """
    eos, sL, sR = states()
    X0 = feats(eos, sL, sR, 0)
    fL = {**sL, **{k: -sL[k] for k in ("Bx", "By", "Bz")}}
    fR = {**sR, **{k: -sR[k] for k in ("Bx", "By", "Bz")}}
    X1 = feats(eos, fL, fR, 0)
    rel = float((X0 - X1).abs().max() / X0.abs().max())
    assert rel < 1e-13, f"features not invariant under B -> -B: {rel:.3e}"


def test_generator_and_inference_build_identical_features():
    """The two call sites must produce bit-identical vectors.

    Reproduces the generator's call path and the inference path's call path
    on the same states.  Before ai_features.py existed these were separate
    hand-written expressions and could drift apart silently -- which is
    exactly what a train/inference mismatch looks like: no error, just a
    network being fed something it was never trained on.
    """
    eos, sL, sR = states(n=1500)

    # generator path (src/data/generate.py)
    uL, uR, fL_, fR_, cmax, cmin = compute_srmhd_fluxes(sL, sR, eos, 0)
    UL, FL = energy_form(uL, fL_)
    UR, FR = energy_form(uR, fR_)
    RL, RR = ai.r_vectors(UL, FL, UR, FR, cmin, cmax)
    X_gen = ai.build_pstar_features(sL, sR, uL, uR, RL, RR, cmin, cmax, idir=0)

    # inference path: run hlld_ai_flux with a stub net and capture its input
    from src.physics import hlld as H

    seen = {}

    class _Net(torch.nn.Module):
        def forward(self, x):
            seen["X"] = x.double()
            return torch.zeros(x.shape[0], 1, dtype=x.dtype)

    norm = {"mu": np.zeros(ai.N_FEATURES), "sigma": np.ones(ai.N_FEATURES),
            "y_mu": 0.0, "y_sigma": 1.0}
    H.hlld_ai_flux(sL, sR, eos, _Net(), norm, idir=0)

    # the stub sees the normalised vector; with mu=0, sigma=1 that is the raw
    # vector, cast to float32 by the model call
    assert torch.allclose(seen["X"], X_gen.float().double(), rtol=0, atol=0), (
        "generator and inference feature vectors differ"
    )
