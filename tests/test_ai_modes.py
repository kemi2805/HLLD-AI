"""
The two AI prediction modes, and the feature-version guard.

``oneshot`` uses the network's p* directly; ``warmstart`` uses it to seed the
same secant iteration the classical solver runs.  The paper reports both, so
the properties that distinguish them are asserted here rather than assumed.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.physics import ai_features as ai
from src.physics import hlld as H
from src.physics.eos import hybrid_eos

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_idir import random_states, GAMMA  # noqa: E402

NORM = {"mu": np.zeros(ai.N_FEATURES), "sigma": np.ones(ai.N_FEATURES),
        "y_mu": 0.0, "y_sigma": 1.0}


def _net_returning(p):
    """A stub network that returns a prescribed p* (in the model's log space)."""
    class _N(torch.nn.Module):
        def forward(self, x):
            return torch.log10(torch.clamp(p, min=1e-30)).unsqueeze(1).float()
    return _N()


def _setup(n=2000, rel_err=1e-3):
    eos = hybrid_eos(K=0.0, gamma=GAMMA, gamma_th=GAMMA)
    sL, sR = random_states(n, eos, max_W=2.5)
    _, _, p_true = H.hlld_flux(sL, sR, eos, idir=0)
    cold = dict(H.LAST_DIAG)
    return eos, sL, sR, p_true, cold, _net_returning(p_true * (1 + rel_err))


def test_warmstart_converges_regardless_of_prediction_error():
    """warmstart must reach solver tolerance even from a poor prediction."""
    eos, sL, sR, p_true, _, _ = _setup()
    for rel in (1e-1, 1e-3):
        net = _net_returning(p_true * (1 + rel))
        _, _, ps = H.hlld_ai_flux(sL, sR, eos, net, NORM, idir=0,
                                  mode="warmstart")
        m = (p_true > 0) & (ps > 0)
        err = float(((ps[m] - p_true[m]).abs() / p_true[m]).max())
        assert err < 1e-6, f"warmstart did not converge (rel={rel}): {err:.2e}"


def test_oneshot_returns_the_prediction_unmodified():
    """oneshot performs no iteration, so it inherits the prediction error."""
    eos, sL, sR, p_true, _, _ = _setup()
    rel = 1e-3
    net = _net_returning(p_true * (1 + rel))
    _, _, ps = H.hlld_ai_flux(sL, sR, eos, net, NORM, idir=0, mode="oneshot")
    d = dict(H.LAST_DIAG)
    m = (p_true > 0) & (ps > 0)
    err = float(((ps[m] - p_true[m]).abs() / p_true[m]).max())
    assert err == pytest.approx(rel, rel=0.05)
    assert d["mean_iters"] == 0.0


def test_warmstart_costs_fewer_iterations_than_a_cold_start():
    """The headline cost claim: a warm start needs fewer secant iterations.

    Both arms use the SAME root-finder and tolerance, so the comparison is
    of the starting bracket alone.
    """
    eos, sL, sR, p_true, cold, net = _setup(rel_err=1e-3)
    H.hlld_ai_flux(sL, sR, eos, net, NORM, idir=0, mode="warmstart")
    warm = dict(H.LAST_DIAG)
    assert warm["mean_iters"] < cold["mean_iters"], (
        f"warm start ({warm['mean_iters']:.2f}) not cheaper than cold "
        f"({cold['mean_iters']:.2f})"
    )


def test_prediction_residual_is_reported():
    """LAST_DIAG must expose |residual(p_nn)|.

    A one-shot p* is only trustworthy if it is near a root, and the HLLD
    residual is multi-rooted, so this is the diagnostic that says whether
    the network's answer actually satisfies the jump conditions.
    """
    eos, sL, sR, p_true, _, net = _setup()
    H.hlld_ai_flux(sL, sR, eos, net, NORM, idir=0, mode="oneshot")
    d = dict(H.LAST_DIAG)
    assert "median_resid_nn" in d and "max_resid_nn" in d
    assert np.isfinite(d["median_resid_nn"])


def test_unknown_mode_is_rejected():
    eos, sL, sR, p_true, _, net = _setup(n=50)
    with pytest.raises(ValueError):
        H.hlld_ai_flux(sL, sR, eos, net, NORM, idir=0, mode="nonsense")


def test_feature_version_mismatch_is_refused():
    """A stale norm_stats must fail loudly, not silently mispredict."""
    import tempfile, os
    from src.physics.driver import _load_ai_solver
    from src.models.network import PressureNet

    with tempfile.TemporaryDirectory() as d:
        ck = os.path.join(d, "m.pt")
        st = os.path.join(d, "s.npz")
        net = PressureNet(n_input=ai.N_FEATURES, hidden=[8], activation="silu")
        torch.save(net.state_dict(), ck)
        np.savez(st, mu=np.zeros(ai.N_FEATURES), sigma=np.ones(ai.N_FEATURES),
                 y_mu=np.array(0.0), y_sigma=np.array(1.0),
                 feature_version=np.array("v0-stale"))
        cfg = {"ml": {"checkpoint": ck, "norm_stats": st,
                      "n_input": ai.N_FEATURES, "hidden": [8],
                      "activation": "silu"}}
        with pytest.raises(ValueError, match="feature-version mismatch"):
            _load_ai_solver(cfg, "cpu")
