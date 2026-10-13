"""The four-quadrant Riemann problem with a uniform in-plane field: the
states are the published ones, the field is what was asked for, and the
staggered field built from the potential is divergence-free to round-off.
"""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.physics.eos import hybrid_eos                                   # noqa: E402
from src.physics.grid import Grid2D                                      # noqa: E402
from src.physics.initial_data2d import b_from_potential, riemann2d       # noqa: E402

GAMMA = 5.0 / 3.0


def _setup(n=16, **kw):
    eos = hybrid_eos(K=0.0, gamma=GAMMA, gamma_th=GAMMA)
    g = Grid2D(-0.5, 0.5, n, -0.5, 0.5, n, ng=2)
    prims, Az = riemann2d(g, eos, **kw)
    Bxf, Byf = b_from_potential(Az, g.dx, g.dy)
    return g, prims, Bxf, Byf


def test_the_four_quadrants_are_the_published_states():
    g, prims, _, _ = _setup()
    X, Y = g.X, g.Y
    quads = {  # (right, top): rho, p, vx, vy
        (False, True): (0.1, 1.0, 0.99, 0.0),
        (True, True): (0.1, 0.01, 0.0, 0.0),
        (False, False): (0.5, 1.0, 0.0, 0.0),
        (True, False): (0.1, 1.0, 0.0, 0.99),
    }
    for (right, top), (rho, p, vx, vy) in quads.items():
        m = ((X > 0) == right) & ((Y > 0) == top)
        assert m.any()
        for k, v in (("rho", rho), ("p", p), ("vx", vx), ("vy", vy)):
            assert torch.all(prims[k][m] == v), (right, top, k)
    assert torch.all(prims["vz"] == 0) and torch.all(prims["Bz"] == 0)
    v2 = prims["vx"] ** 2 + prims["vy"] ** 2
    assert float(v2.max()) < 1.0


def test_the_field_is_uniform_at_the_asked_strength_and_angle():
    for B0, ang in ((0.5, 45.0), (1.0, 0.0), (0.3, 90.0)):
        g, prims, Bxf, Byf = _setup(B0=B0, angle_deg=ang)
        bx, by = B0 * math.cos(math.radians(ang)), B0 * math.sin(math.radians(ang))
        assert torch.allclose(Bxf, torch.full_like(Bxf, bx), atol=1e-12)
        assert torch.allclose(Byf, torch.full_like(Byf, by), atol=1e-12)
        assert torch.allclose(prims["Bx"], torch.full_like(prims["Bx"], bx), atol=1e-15)
        assert torch.allclose(prims["By"], torch.full_like(prims["By"], by), atol=1e-15)


def test_the_45_degree_problem_is_exactly_diagonal_symmetric():
    """x <-> y maps the problem onto itself (top-left and bottom-right
    exchange, the field at 45 degrees is unchanged), and the arrays must
    say so to the bit: cos and sin of pi/4 differ in the last bit, which a
    canonical-orientation run turned into exact-vs-HLLD fluxes on mirror
    faces in its first step."""
    g, prims, Bxf, Byf = _setup(n=32, B0=0.5, angle_deg=45.0)
    for a, b in (("rho", "rho"), ("p", "p"), ("vx", "vy"), ("Bx", "By"), ("Bz", "Bz")):
        assert torch.equal(prims[a], prims[b].T), (a, b)
    assert torch.equal(Bxf, Byf.T)


def test_the_staggered_field_is_divergence_free():
    g, _, Bxf, Byf = _setup(n=32)
    div = ((Bxf[1:, :] - Bxf[:-1, :]) / g.dx + (Byf[:, 1:] - Byf[:, :-1]) / g.dy)
    assert float(div.abs().max()) < 1e-12


def test_the_out_of_plane_tilt_keeps_the_in_plane_field_divergence_free():
    g, prims, Bxf, Byf = _setup(n=32, B0=0.5, angle_deg=45.0, angle_z_deg=60.0)
    c, s = math.cos(math.radians(60.0)), math.sin(math.radians(60.0))
    bx = 0.5 * math.cos(math.radians(45.0)) * c
    assert torch.allclose(Bxf, torch.full_like(Bxf, bx), atol=1e-12)
    assert torch.allclose(prims["Bz"], torch.full_like(prims["Bz"], 0.5 * s), atol=1e-15)
    B2 = prims["Bx"] ** 2 + prims["By"] ** 2 + prims["Bz"] ** 2
    assert torch.allclose(B2, torch.full_like(B2, 0.25), atol=1e-14)
    div = ((Bxf[1:, :] - Bxf[:-1, :]) / g.dx + (Byf[:, 1:] - Byf[:, :-1]) / g.dy)
    assert float(div.abs().max()) < 1e-12


def test_beckwith_and_stones_orszag_tang_is_theirs():
    """`orszag_tang_bs` is Beckwith & Stone (2011)'s relativistic Orszag-Tang
    vortex: their density, pressure and field, v = 0.5 per component (the
    standard orientation), and run_2d gives it Gamma = 5/3; ours differs only
    in the speed."""
    from src.physics.initial_data2d import orszag_tang, orszag_tang_bs
    eos = hybrid_eos(K=0.0, gamma=GAMMA, gamma_th=GAMMA)
    g = Grid2D(0.0, 1.0, 32, 0.0, 1.0, 32, ng=2)
    bs, Az = orszag_tang_bs(g, eos)
    ours, Az2 = orszag_tang(g, eos)
    X, Y = g.X, g.Y
    assert torch.allclose(bs["vx"], -0.5 * torch.sin(2 * math.pi * Y), atol=1e-15)
    assert torch.allclose(bs["vy"], 0.5 * torch.sin(2 * math.pi * X), atol=1e-15)
    assert torch.all(bs["rho"] == 25.0 / (36.0 * math.pi))
    assert torch.all(bs["p"] == 5.0 / (12.0 * math.pi))
    for k in ("Bx", "By", "Bz", "rho", "p"):
        assert torch.equal(bs[k], ours[k]), k
    assert torch.equal(Az, Az2)
    src = (Path(__file__).resolve().parents[1] / "scripts" / "run_2d.py").read_text()
    assert '"orszag_tang_bs": dict(box=(0.0, 1.0), gamma=5.0 / 3.0' in src
