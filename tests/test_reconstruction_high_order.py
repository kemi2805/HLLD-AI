"""The high-order reconstructions: do they converge at their order, and do
they overshoot?

Two properties, and they pull against each other -- a scheme can have either
one alone trivially (PCM never overshoots; the unlimited interpolation is
exactly fifth order and rings).  What is tested here is that each scheme has
both at once.

* ORDER.  Reconstruct the cell averages of a smooth periodic function and
  compare the face values with the exact ones.  The measured slope of the
  error against resolution must reach the scheme's nominal order.  The cell
  averages are computed exactly, not sampled, or the test measures the
  quadrature error instead.
* NO SPIKES.  On a step, the reconstructed face values must stay inside the
  data's own range, up to a tolerance that is zero for the MP schemes (they
  are monotonicity-preserving by construction) and small for WENO-Z (which
  is not).
* SMOOTH EXTREMA.  At a smooth maximum a TVD limiter clips and drops to
  first order.  The MP schemes must not: this is the reason to have them,
  so it is measured rather than asserted in a comment.
"""
import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.physics.reconstruction import (  # noqa: E402
    HIGH_ORDER, ghosts_needed, reconstruct,
)

# NOT torch.set_default_dtype: it is global to the pytest session, and the
# warm-start network is float32 -- setting it here made every later test that
# feeds a tensor to the model die with "mat1 and mat2 must have the same
# dtype" (30 failures, calea job 1426).  Each tensor below carries its own.
ORDER = {"weno5z": 5, "mp5": 5, "mp7": 7}


def _averages(f, F, n, w):
    """Exact cell averages of ``f`` on ``n`` cells of width ``w`` covering
    [0, 1), from its antiderivative ``F``."""
    i = torch.arange(n, dtype=torch.float64)
    xl, xr = i * w, (i + 1) * w
    return (F(xr) - F(xl)) / w


@pytest.mark.parametrize("limiter", HIGH_ORDER)
def test_order_of_accuracy(limiter):
    f = lambda x: torch.sin(2.0 * math.pi * x)
    F = lambda x: -torch.cos(2.0 * math.pi * x) / (2.0 * math.pi)
    errs = []
    for n in (32, 64, 128, 256):
        w = 1.0 / n
        # periodic padding: the stencil is then complete on every physical
        # face, which is what isolates the scheme from the edge fallback
        g = ghosts_needed(limiter)
        Q = _averages(f, F, n, w)
        Qp = torch.cat([Q[-g:], Q, Q[:g]])
        QL, QR = reconstruct(Qp, axis=0, limiter=limiter)
        x_face = torch.arange(n + 1, dtype=torch.float64) * w
        exact = f(x_face)
        got_L = QL[g:g + n + 1]          # left side of each physical face
        got_R = QR[g:g + n + 1]
        errs.append(max(float((got_L - exact).abs().max()),
                        float((got_R - exact).abs().max())))
    # levels whose error has reached round-off measure double precision, not
    # the scheme: mp7 is at 3e-14 by 256 cells and its apparent rate falls to
    # 6.3 there for that reason alone
    keep = [e for e in errs if e > 1e-13]
    rates = [math.log2(keep[k] / keep[k + 1]) for k in range(len(keep) - 1)]
    assert len(rates) >= 2, (limiter, errs)
    assert min(rates[1:]) > ORDER[limiter] - 0.5, (limiter, errs, rates)


@pytest.mark.parametrize("limiter", HIGH_ORDER)
def test_no_new_extrema_at_a_step(limiter):
    n, g = 64, ghosts_needed("mp7")
    Q = torch.where(torch.arange(n + 2 * g) < (n // 2 + g),
                    torch.tensor(1.0, dtype=torch.float64),
                    torch.tensor(0.1, dtype=torch.float64))
    QL, QR = reconstruct(Q, axis=0, limiter=limiter)
    lo, hi = float(Q.min()), float(Q.max())
    span = hi - lo
    over = max(float((QL - hi).max()), float((QR - hi).max()),
               float((lo - QL).max()), float((lo - QR).max())) / span
    # the MP schemes create no new extremum at all; WENO-Z is allowed the
    # small overshoot it is known to have
    bar = 0.0 if limiter.startswith("mp") else 0.02
    assert over <= bar + 1e-14, (limiter, over)


@pytest.mark.parametrize("limiter", HIGH_ORDER)
def test_a_smooth_extremum_is_not_clipped(limiter):
    """A TVD limiter flattens a smooth peak; these must not."""
    n, g = 200, ghosts_needed(limiter)
    w = 1.0 / n
    f = lambda x: torch.exp(-200.0 * (x - 0.5) ** 2)
    x = (torch.arange(n + 2 * g, dtype=torch.float64) - g + 0.5) * w
    Q = f(x)
    peak = int(torch.argmax(Q))
    QL, _ = reconstruct(Q, axis=0, limiter=limiter)
    _, QRmc = reconstruct(Q, axis=0, limiter="mc")
    # the face value just past the peak, against the cell average there
    hi_ho = float(QL[peak + 1])
    hi_mc = float(reconstruct(Q, axis=0, limiter="mc")[0][peak + 1])
    exact = float(f(torch.tensor((peak - g + 1.0) * w, dtype=torch.float64)))
    assert abs(hi_ho - exact) < abs(hi_mc - exact), (limiter, hi_ho, hi_mc, exact)


def test_ghosts_needed():
    assert ghosts_needed("mc") == 2
    assert ghosts_needed("weno5z") == ghosts_needed("mp5") == 3
    assert ghosts_needed("mp7") == 4
    with pytest.raises(ValueError):
        ghosts_needed("nonesuch")


@pytest.mark.parametrize("limiter", HIGH_ORDER)
def test_constant_state_is_exact(limiter):
    Q = torch.full((40,), 2.5, dtype=torch.float64)
    QL, QR = reconstruct(Q, axis=0, limiter=limiter)
    assert torch.equal(QL, torch.full_like(QL, 2.5))
    assert torch.equal(QR, torch.full_like(QR, 2.5))


@pytest.mark.parametrize("limiter", HIGH_ORDER)
def test_axis_1_matches_axis_0(limiter):
    torch.manual_seed(0)
    Q = torch.rand(30, 7, dtype=torch.float64)
    L0, R0 = reconstruct(Q, axis=0, limiter=limiter)
    L1, R1 = reconstruct(Q.T.contiguous(), axis=1, limiter=limiter)
    assert torch.allclose(L0, L1.T) and torch.allclose(R0, R1.T)
