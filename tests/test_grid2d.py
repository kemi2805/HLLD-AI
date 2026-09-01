"""
Grid2D geometry and boundary conditions.

The staggered boundary rules are the easiest place in the 2D extension to
introduce a silent off-by-one, because a face-staggered array has n+1
physical entries along its normal axis where a cell-centred array has n.
Every ghost value below is therefore checked against a hand-computed index
rather than against another implementation.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.physics.grid import Grid1D, Grid2D
from src.physics.state import flat, unflat, State2D, combine


NG = 2


def g2(nx=8, ny=6, ng=NG):
    return Grid2D(0.0, 1.0, nx, -0.5, 0.5, ny, ng=ng)


# ── geometry ─────────────────────────────────────────────────────────────


def test_matches_grid1d_coordinates():
    """The x axis of Grid2D must agree with Grid1D for the same parameters."""
    g1 = Grid1D(0.0, 1.0, 8, ng=NG)
    g = Grid2D(0.0, 1.0, 8, -0.5, 0.5, 6, ng=NG)
    assert torch.equal(g.x, g1.x)
    assert g.dx == g1.dx
    assert g.nxt == g1.ntotal


def test_face_coordinates_bracket_cells():
    """xf[i] and xf[i+1] must bracket the cell centre x[i], exactly."""
    g = g2()
    assert torch.allclose(g.xf[:-1] + 0.5 * g.dx, g.x, atol=0, rtol=1e-15)
    assert torch.allclose(g.yf[:-1] + 0.5 * g.dy, g.y, atol=0, rtol=1e-15)
    # physical domain endpoints land exactly on faces
    assert float(g.xf[g.ng]) == pytest.approx(0.0, abs=1e-15)
    assert float(g.xf[g.ng + g.nx]) == pytest.approx(1.0, abs=1e-15)


def test_array_shapes():
    g = g2(nx=8, ny=6)
    assert tuple(g.zeros().shape) == (g.nxt, g.nyt)
    assert tuple(g.zeros_xf().shape) == (g.nxt + 1, g.nyt)
    assert tuple(g.zeros_yf().shape) == (g.nxt, g.nyt + 1)
    assert tuple(g.zeros_corner().shape) == (g.nxt + 1, g.nyt + 1)


# ── cell-centred boundary conditions ─────────────────────────────────────


def test_cc_outflow_copies_edge_values():
    g = g2()
    ng, nx, ny = g.ng, g.nx, g.ny
    v = torch.arange(g.nxt * g.nyt, dtype=torch.float64).reshape(g.nxt, g.nyt)
    ref = v.clone()
    g.apply_bc_cc({"v": v}, "outflow", "outflow")

    # every left-x ghost row equals the first physical row (post y-fill)
    for i in range(ng):
        assert torch.equal(v[i, :], v[ng, :])
    for i in range(ng + nx, g.nxt):
        assert torch.equal(v[i, :], v[ng + nx - 1, :])
    # y ghosts equal the edge physical columns
    for j in range(ng):
        assert torch.equal(v[:, j], v[:, ng])
    for j in range(ng + ny, g.nyt):
        assert torch.equal(v[:, j], v[:, ng + ny - 1])
    # interior untouched
    assert torch.equal(v[g.phys], ref[g.phys])


def test_cc_periodic_wraps():
    g = g2()
    ng, nx, ny = g.ng, g.nx, g.ny
    v = torch.arange(g.nxt * g.nyt, dtype=torch.float64).reshape(g.nxt, g.nyt)
    g.apply_bc_cc({"v": v}, "periodic", "periodic")
    # ghost i (i < ng) mirrors physical cell ng + nx - (ng - i)
    for i in range(ng):
        assert torch.equal(v[i, :], v[nx + i, :]), f"left x ghost {i}"
    for i in range(ng):
        assert torch.equal(v[ng + nx + i, :], v[ng + i, :]), f"right x ghost {i}"
    for j in range(ng):
        assert torch.equal(v[:, j], v[:, ny + j])
        assert torch.equal(v[:, ng + ny + j], v[:, ng + j])


def test_cc_constant_holds_reference():
    g = g2()
    ng, nx = g.ng, g.nx
    ref = torch.full((g.nxt, g.nyt), 7.0, dtype=torch.float64)
    v = torch.zeros(g.nxt, g.nyt, dtype=torch.float64)
    g.apply_bc_cc({"v": v}, "constant", "constant", U_ref={"v": ref})
    assert torch.all(v[:ng, :] == 7.0)
    assert torch.all(v[ng + nx :, :] == 7.0)
    assert torch.all(v[g.phys] == 0.0)


# ── staggered boundary conditions ────────────────────────────────────────


def test_face_outflow_uses_inclusive_last_face():
    """Bxf has nx+1 physical faces: ng .. ng+nx INCLUSIVE."""
    g = g2()
    ng, nx = g.ng, g.nx
    Bxf = torch.arange((g.nxt + 1) * g.nyt, dtype=torch.float64).reshape(
        g.nxt + 1, g.nyt
    )
    Byf = torch.zeros(g.nxt, g.nyt + 1, dtype=torch.float64)
    last_phys = Bxf[ng + nx, :].clone()
    first_phys = Bxf[ng, :].clone()
    g.apply_bc_face(Bxf, Byf, "outflow", "outflow")
    for i in range(ng):
        assert torch.equal(Bxf[i, ng:-ng], first_phys[ng:-ng])
    for i in range(ng + nx + 1, g.nxt + 1):
        assert torch.equal(Bxf[i, ng:-ng], last_phys[ng:-ng])


def test_face_periodic_collapses_the_seam():
    """The faces at ng and ng+nx are the SAME physical face."""
    g = g2()
    ng, nx = g.ng, g.nx
    Bxf = torch.randn(g.nxt + 1, g.nyt, dtype=torch.float64)
    Byf = torch.randn(g.nxt, g.nyt + 1, dtype=torch.float64)
    g.apply_bc_face(Bxf, Byf, "periodic", "periodic")

    # seam must be bit-identical, not merely close
    assert torch.equal(Bxf[ng + nx, :], Bxf[ng, :]), "x seam face not collapsed"
    assert torch.equal(Byf[:, ng + g.ny], Byf[:, ng]), "y seam face not collapsed"
    # ghosts wrap onto distinct physical faces
    for i in range(ng):
        assert torch.equal(Bxf[i, :], Bxf[nx + i, :]), f"left face ghost {i}"
        assert torch.equal(Bxf[ng + nx + 1 + i, :], Bxf[ng + 1 + i, :])


# ── divergence ───────────────────────────────────────────────────────────


def _from_vector_potential(g, Az):
    """Bxf = dAz/dy, Byf = -dAz/dx, from a corner-valued Az."""
    Bxf = (Az[:, 1:] - Az[:, :-1]) / g.dy
    Byf = -(Az[1:, :] - Az[:-1, :]) / g.dx
    return Bxf, Byf


def test_div_b_zero_from_vector_potential():
    """Any field built as the curl of a corner Az is discretely div-free."""
    g = g2(nx=16, ny=16)
    XF = g.xf.unsqueeze(1).expand(g.nxt + 1, g.nyt + 1)
    YF = g.yf.unsqueeze(0).expand(g.nxt + 1, g.nyt + 1)
    # a deliberately non-trivial potential
    Az = torch.sin(2 * torch.pi * XF) * torch.cos(3 * torch.pi * YF) + 0.3 * XF
    Bxf, Byf = _from_vector_potential(g, Az)
    d = g.div_b(Bxf, Byf)
    scale = max(float(Bxf.abs().max()), float(Byf.abs().max()))
    assert float(d.abs().max()) / (scale / g.dx) < 1e-13


def test_div_b_uniform_field_is_zero():
    g = g2()
    Bxf = torch.full((g.nxt + 1, g.nyt), 1.0, dtype=torch.float64)
    Byf = torch.zeros(g.nxt, g.nyt + 1, dtype=torch.float64)
    assert float(g.div_b(Bxf, Byf).abs().max()) == 0.0


# ── state helpers ────────────────────────────────────────────────────────


def test_flat_unflat_is_a_view_and_roundtrips():
    g = g2()
    d = {"D": g.zeros() + 3.0, "tau": g.zeros() + 5.0}
    f = flat(d)
    assert f["D"].shape == (g.nxt * g.nyt,)
    # a view, not a copy: writing through the flat form must be visible
    f["D"][0] = 99.0
    assert float(d["D"][0, 0]) == 99.0
    back = unflat(f, g.shape)
    assert torch.equal(back["D"], d["D"])


def test_combine_applies_identical_weights_to_cons_and_faces():
    g = g2()
    def mk(c):
        return State2D(
            cons={"D": g.zeros() + c},
            Bxf=g.zeros_xf() + c,
            Byf=g.zeros_yf() + c,
        )
    out = combine([mk(1.0), mk(2.0)], [0.25, 0.75])
    want = 0.25 * 1.0 + 0.75 * 2.0
    assert torch.allclose(out.cons["D"], torch.full_like(out.cons["D"], want))
    assert torch.allclose(out.Bxf, torch.full_like(out.Bxf, want))
    assert torch.allclose(out.Byf, torch.full_like(out.Byf, want))


def test_combine_preserves_zero_divergence():
    """An affine combination of div-free states stays div-free.

    This is the whole CT/Runge-Kutta consistency argument, so it is asserted
    rather than assumed.
    """
    g = g2(nx=16, ny=16)
    XF = g.xf.unsqueeze(1).expand(g.nxt + 1, g.nyt + 1)
    YF = g.yf.unsqueeze(0).expand(g.nxt + 1, g.nyt + 1)
    states = []
    for k in (1.0, 2.0, 3.0):
        Az = torch.sin(k * torch.pi * XF) * torch.cos(k * torch.pi * YF)
        Bxf, Byf = _from_vector_potential(g, Az)
        states.append(State2D(cons={"D": g.zeros()}, Bxf=Bxf, Byf=Byf))
    out = combine(states, [1.0 / 3.0, 2.0 / 3.0, -1.0 / 3.0])
    scale = max(float(out.Bxf.abs().max()), float(out.Byf.abs().max()))
    assert float(g.div_b(out.Bxf, out.Byf).abs().max()) / (scale / g.dx) < 1e-13


def test_combine_rejects_mismatched_weights():
    g = g2()
    s = State2D(cons={"D": g.zeros()}, Bxf=g.zeros_xf(), Byf=g.zeros_yf())
    with pytest.raises(ValueError):
        combine([s, s], [1.0])
