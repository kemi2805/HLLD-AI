"""Newton-Raphson for the HLLD star pressure (opt-in; the secant is production).

These pin the root-finder itself, on functions whose roots are known, and
the one property production depends on: that the default is still the
secant.  Whether Newton is a better choice for the rotor is a measurement,
not a test -- scripts/pstar_methods.py, and docs/what_we_solve.md.
"""
import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.physics import hlld as H


def test_the_default_is_still_the_secant():
    if "RMHD_PSTAR_METHOD" in os.environ:
        pytest.skip("RMHD_PSTAR_METHOD is set in this environment")
    assert H._PSTAR_METHOD == "secant"


@pytest.mark.parametrize("derivative", ["ad", "fd"])
def test_newton_finds_a_known_root_quadratically(derivative):
    roots = torch.tensor([0.5, 1.7, 3.0, 12.0], dtype=torch.float64)
    f = lambda p: p * p - roots * roots
    x, err, it, cost = H.newton_pstar(f, roots * 1.3, tol=1e-12,
                                      max_iter=30, derivative=derivative)
    assert torch.all(err == 0)
    assert torch.allclose(x, roots, rtol=1e-12, atol=0)
    # quadratic convergence from 30% away: a handful of steps, not dozens
    assert int(it.max()) <= 7, it


def test_the_bracket_keeps_newton_on_the_bracketed_root():
    """f has roots at 1, 2 and 3.  Started at 2.2 with the derivative
    pointing away, plain Newton leaves [1.5, 2.5]; the guarded form must
    return the root the bracket holds."""
    f = lambda p: (p - 1.0) * (p - 2.0) * (p - 3.0)
    lo = torch.tensor([1.5], dtype=torch.float64)
    hi = torch.tensor([2.5], dtype=torch.float64)
    x, err, _, _ = H.newton_pstar(f, torch.tensor([2.45], dtype=torch.float64),
                                  tol=1e-12, bracket=(lo, hi, f(lo), f(hi)))
    assert int(err[0]) == 0 and abs(float(x[0]) - 2.0) < 1e-10


def test_a_flat_derivative_fails_the_lane_instead_of_dividing_by_zero():
    f = lambda p: torch.ones_like(p)          # no root, f' = 0
    x, err, _, _ = H.newton_pstar(f, torch.tensor([1.0], dtype=torch.float64))
    assert int(err[0]) > 0 and torch.isfinite(x).all()


def test_the_forward_difference_from_two_close_values_converges():
    """Newton with f'(p) from p and p + h -- one extra evaluation per step."""
    roots = torch.tensor([0.5, 1.7, 3.0, 12.0], dtype=torch.float64)
    f = lambda p: p * p - roots * roots
    x, err, it, cost = H.newton_pstar(f, roots * 1.3, tol=1e-12,
                                      derivative="fd1", fd_h=1.5e-8)
    assert torch.all(err == 0)
    assert torch.allclose(x, roots, rtol=1e-12, atol=0)
    # two evaluations per step: f(p) and f(p + h), plus the backtracking ones
    assert cost["backward"] == 0
