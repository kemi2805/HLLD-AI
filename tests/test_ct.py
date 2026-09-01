"""
Constrained transport: divergence preservation and EMF correctness.

The two decisive tests here are

  * ``test_div_b_telescopes`` — the CT update preserves div(B) to bit-zero,
    checked on random data so it tests the discrete operator itself,
    independently of any physics; and

  * ``test_corner_emf_reduces_to_face_value`` — when E^z varies along only
    one axis the corner EMF must equal the corresponding face value
    EXACTLY.  That is the Gardiner-Stone design requirement, and it pins
    down every sign and index in the eight ``(1 +/- S)`` upwind terms.  A
    transcription error in any one of them breaks it.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.physics.ct import (
    cell_centred_B, corner_emf, ct_rhs, ct_update, div_b, emf_cell,
)
from src.physics.grid import Grid2D
from src.physics.initial_data2d import b_from_potential

torch.manual_seed(4242)

NXT, NYT = 14, 12
DX, DY = 0.07, 0.11


def _rand_faces():
    Bxf = torch.randn(NXT + 1, NYT, dtype=torch.float64)
    Byf = torch.randn(NXT, NYT + 1, dtype=torch.float64)
    return Bxf, Byf


# ── divergence ───────────────────────────────────────────────────────────


def test_div_b_telescopes():
    """The CT update changes div(B) by exactly zero, for ARBITRARY Ez.

    This is a property of the discrete curl alone, so random inputs are the
    right test: it must hold regardless of whether Ez came from a physical
    EMF.
    """
    Bxf, Byf = _rand_faces()
    Ez = torch.randn(NXT + 1, NYT + 1, dtype=torch.float64)
    d0 = div_b(Bxf, Byf, DX, DY)
    Bxf2, Byf2 = ct_update(Bxf, Byf, Ez, dt=0.37, dx=DX, dy=DY)
    d1 = div_b(Bxf2, Byf2, DX, DY)
    change = float((d1 - d0).abs().max())
    # The bar is the ROUND-OFF FLOOR of the divergence itself, not an
    # absolute number: div_b differences O(1) face values divided by dx, so
    # its own magnitude is ~1/dx and a few ulps of that is the best any
    # implementation can do.  Anything above this floor means the four EMF
    # contributions are not cancelling.
    floor = 2.3e-16 * max(float(d0.abs().max()), float(d1.abs().max()))
    assert change <= 50 * floor, (
        f"div(B) changed by {change:.3e}, round-off floor is {floor:.3e}"
    )


def test_div_b_stays_zero_over_many_updates():
    g = Grid2D(0.0, 1.0, 16, 0.0, 1.0, 16, ng=2)
    XF = g.xf.unsqueeze(1).expand(g.nxt + 1, g.nyt + 1)
    YF = g.yf.unsqueeze(0).expand(g.nxt + 1, g.nyt + 1)
    Az = torch.sin(2 * torch.pi * XF) * torch.cos(2 * torch.pi * YF)
    Bxf, Byf = b_from_potential(Az, g.dx, g.dy)
    assert float(div_b(Bxf, Byf, g.dx, g.dy).abs().max()) < 1e-12

    for _ in range(200):
        Ez = torch.randn(g.nxt + 1, g.nyt + 1, dtype=torch.float64)
        Bxf, Byf = ct_update(Bxf, Byf, Ez, 1e-3, g.dx, g.dy)
    scale = max(float(Bxf.abs().max()), float(Byf.abs().max()))
    assert float(div_b(Bxf, Byf, g.dx, g.dy).abs().max()) / (scale / g.dx) < 1e-13


# ── corner EMF ───────────────────────────────────────────────────────────


def _fields_from_ez(ez_cell, ez_facex, ez_facey, rho_x, rho_y):
    return dict(Ec=ez_cell, Efx=ez_facex, Efy=ez_facey,
                Fx_rho=rho_x, Fy_rho=rho_y)


@pytest.mark.parametrize("sgn", [-1.0, 0.0, 1.0], ids=["neg", "zero", "pos"])
def test_corner_emf_reduces_to_face_value_y_dependence(sgn):
    """E^z depending only on y  =>  corner EMF equals the y-face value exactly.

    With no x-dependence the x-face EMF equals the cell-centred value, so
    every upwind term must cancel identically and leave Efy.  Checked for
    all three mass-flux signs, including exactly zero.
    """
    ey = torch.randn(NYT, dtype=torch.float64)
    Ec = ey.unsqueeze(0).expand(NXT, NYT).contiguous()
    Efx = ey.unsqueeze(0).expand(NXT + 1, NYT).contiguous()   # = cell value
    Efy = torch.randn(NYT + 1, dtype=torch.float64).unsqueeze(0).expand(
        NXT, NYT + 1).contiguous()
    Fx_rho = torch.full((NXT + 1, NYT), sgn, dtype=torch.float64)
    Fy_rho = torch.full((NXT, NYT + 1), sgn, dtype=torch.float64)

    Ez = corner_emf(Ec, Efx, Efy, Fx_rho, Fy_rho, upwind=True)
    got = Ez[1:NXT, 1:NYT]
    want = Efy[1:NXT, 1:NYT]
    # Exact in real arithmetic, but Eavg and the upwind correction group the
    # 0.25*(Ec_j + Ec_j-1) terms differently, so they cancel only to
    # round-off.  A wrong sign or index in any of the eight (1 +/- S) terms
    # would show up at O(1), not at 1e-16.
    err = float((got - want).abs().max()) / max(float(want.abs().max()), 1e-30)
    assert err < 1e-14, f"max rel |Ez - Efy| = {err:.3e}"


@pytest.mark.parametrize("sgn", [-1.0, 0.0, 1.0], ids=["neg", "zero", "pos"])
def test_corner_emf_reduces_to_face_value_x_dependence(sgn):
    """E^z depending only on x  =>  corner EMF equals the x-face value exactly."""
    ex = torch.randn(NXT, dtype=torch.float64)
    Ec = ex.unsqueeze(1).expand(NXT, NYT).contiguous()
    Efy = ex.unsqueeze(1).expand(NXT, NYT + 1).contiguous()   # = cell value
    Efx = torch.randn(NXT + 1, dtype=torch.float64).unsqueeze(1).expand(
        NXT + 1, NYT).contiguous()
    Fx_rho = torch.full((NXT + 1, NYT), sgn, dtype=torch.float64)
    Fy_rho = torch.full((NXT, NYT + 1), sgn, dtype=torch.float64)

    Ez = corner_emf(Ec, Efx, Efy, Fx_rho, Fy_rho, upwind=True)
    got = Ez[1:NXT, 1:NYT]
    want = Efx[1:NXT, 1:NYT]
    # Exact in real arithmetic, but Eavg and the upwind correction group the
    # 0.25*(Ec_j + Ec_j-1) terms differently, so they cancel only to
    # round-off.  A wrong sign or index in any of the eight (1 +/- S) terms
    # would show up at O(1), not at 1e-16.
    err = float((got - want).abs().max()) / max(float(want.abs().max()), 1e-30)
    assert err < 1e-14, f"max rel |Ez - Efx| = {err:.3e}"


def test_zero_mass_flux_gives_plain_average():
    """sign(0) == 0 must make the upwind terms cancel to the flux-CT average.

    Uses torch.sign deliberately: copysign(1, 0) = +1 would pick a direction
    at the zero crossing and break the rotor's discrete symmetry.
    """
    Ec = torch.randn(NXT, NYT, dtype=torch.float64)
    Efx = torch.randn(NXT + 1, NYT, dtype=torch.float64)
    Efy = torch.randn(NXT, NYT + 1, dtype=torch.float64)
    zero_x = torch.zeros(NXT + 1, NYT, dtype=torch.float64)
    zero_y = torch.zeros(NXT, NYT + 1, dtype=torch.float64)

    up = corner_emf(Ec, Efx, Efy, zero_x, zero_y, upwind=True)

    # With S = 0 every (1 +/- S) weight is 1, so both neighbours in each pair
    # are weighted EQUALLY -- no handedness.  Write that limit out explicitly.
    I, Im = slice(1, NXT), slice(0, NXT - 1)
    J, Jm = slice(1, NYT), slice(0, NYT - 1)
    yf, yfm = Efy[I, J], Efy[Im, J]
    xf, xfm = Efx[I, J], Efx[I, Jm]
    NE, NW, SE, SW = Ec[I, J], Ec[Im, J], Ec[I, Jm], Ec[Im, Jm]
    Eavg = 0.25 * ((yf + yfm) + (xf + xfm))
    dEdy = 0.125 * ((yf - SE) + (yfm - SW) - (NE - yf) - (NW - yfm))
    dEdx = 0.125 * ((xf - NW) + (xfm - SW) - (NE - xf) - (SE - xfm))
    assert torch.allclose(up[I, J], Eavg + (dEdy + dEdx), rtol=0, atol=1e-15)

    # The sharp statement this rests on: sign(0) is 0, not +1.  copysign
    # would return +1 and pick a direction at the zero crossing, breaking
    # the discrete rotational symmetry the rotor test relies on.
    assert float(torch.sign(torch.zeros(1, dtype=torch.float64))[0]) == 0.0


def test_upwind_false_is_the_four_face_average():
    Ec = torch.randn(NXT, NYT, dtype=torch.float64)
    Efx = torch.randn(NXT + 1, NYT, dtype=torch.float64)
    Efy = torch.randn(NXT, NYT + 1, dtype=torch.float64)
    Fx = torch.randn(NXT + 1, NYT, dtype=torch.float64)
    Fy = torch.randn(NXT, NYT + 1, dtype=torch.float64)
    Ez = corner_emf(Ec, Efx, Efy, Fx, Fy, upwind=False)
    want = 0.25 * ((Efy[1:NXT, 1:NYT] + Efy[0:NXT - 1, 1:NYT])
                   + (Efx[1:NXT, 1:NYT] + Efx[1:NXT, 0:NYT - 1]))
    assert torch.equal(Ez[1:NXT, 1:NYT], want)


def test_upwind_variant_still_divergence_free():
    """Both EMF variants must be exactly divergence-preserving."""
    Bxf, Byf = _rand_faces()
    Ec = torch.randn(NXT, NYT, dtype=torch.float64)
    Efx = torch.randn(NXT + 1, NYT, dtype=torch.float64)
    Efy = torch.randn(NXT, NYT + 1, dtype=torch.float64)
    Fx = torch.randn(NXT + 1, NYT, dtype=torch.float64)
    Fy = torch.randn(NXT, NYT + 1, dtype=torch.float64)
    for upwind in (True, False):
        Ez = corner_emf(Ec, Efx, Efy, Fx, Fy, upwind=upwind)
        d0 = div_b(Bxf, Byf, DX, DY)
        b1, b2 = ct_update(Bxf, Byf, Ez, 0.25, DX, DY)
        d1 = div_b(b1, b2, DX, DY)
        floor = 2.3e-16 * max(float(d0.abs().max()), float(d1.abs().max()))
        assert float((d1 - d0).abs().max()) <= 50 * floor


# ── cell-centred reconstruction of B ─────────────────────────────────────


def test_cell_centred_B_of_uniform_field():
    Bxf = torch.full((NXT + 1, NYT), 2.5, dtype=torch.float64)
    Byf = torch.full((NXT, NYT + 1), -1.25, dtype=torch.float64)
    Bcx, Bcy = cell_centred_B(Bxf, Byf)
    assert torch.allclose(Bcx, torch.full_like(Bcx, 2.5))
    assert torch.allclose(Bcy, torch.full_like(Bcy, -1.25))
    assert Bcx.shape == (NXT, NYT)


def test_uniform_field_is_preserved_by_a_full_ct_cycle():
    """Uniform B with uniform v must not change under the CT update."""
    g = Grid2D(0.0, 1.0, 12, 0.0, 1.0, 12, ng=2)
    Bxf = torch.full((g.nxt + 1, g.nyt), 1.0, dtype=torch.float64)
    Byf = torch.zeros(g.nxt, g.nyt + 1, dtype=torch.float64)
    vx = torch.full(g.shape, 0.3, dtype=torch.float64)
    vy = torch.full(g.shape, -0.2, dtype=torch.float64)

    for _ in range(10):
        Bcx, Bcy = cell_centred_B(Bxf, Byf)
        Ec = emf_cell(Bcx, Bcy, vx, vy)
        # uniform field + uniform velocity => uniform EMF on every face
        Efx = torch.full((g.nxt + 1, g.nyt), float(Ec[0, 0]), dtype=torch.float64)
        Efy = torch.full((g.nxt, g.nyt + 1), float(Ec[0, 0]), dtype=torch.float64)
        Fx = torch.full((g.nxt + 1, g.nyt), 0.3, dtype=torch.float64)
        Fy = torch.full((g.nxt, g.nyt + 1), -0.2, dtype=torch.float64)
        Ez = corner_emf(Ec, Efx, Efy, Fx, Fy)
        Bxf, Byf = ct_update(Bxf, Byf, Ez, 1e-3, g.dx, g.dy)
        g.apply_bc_face(Bxf, Byf, "periodic", "periodic")

    ph = g.phys_xf
    assert float((Bxf[ph] - 1.0).abs().max()) < 1e-14
    assert float(Byf[g.phys_yf].abs().max()) < 1e-14


def test_ct_rhs_shapes():
    Ez = torch.randn(NXT + 1, NYT + 1, dtype=torch.float64)
    dBxf, dByf = ct_rhs(Ez, DX, DY)
    assert dBxf.shape == (NXT + 1, NYT)
    assert dByf.shape == (NXT, NYT + 1)


# ── end-to-end: CT inside the Runge-Kutta driver ─────────────────────────


def _rotor_state(g, eos, **kw):
    from src.physics.driver2d import prims_to_cons_2d, sync_state
    from src.physics.initial_data2d import magnetic_rotor
    from src.physics.state import EVOLVED_KEYS, State2D

    prims, Az = magnetic_rotor(g, eos, **kw)
    Bxf, Byf = b_from_potential(Az, g.dx, g.dy)
    cons_all = prims_to_cons_2d(prims, g)
    st = State2D(cons={k: cons_all[k] for k in EVOLVED_KEYS},
                 Bxf=Bxf, Byf=Byf, prims=prims)
    return sync_state(st, g, eos)


@pytest.mark.parametrize("emf_mode", ["solver", "hll"])
def test_end_to_end_divergence_stays_at_roundoff(emf_mode):
    """A real rotor run keeps div(B) at round-off through full RK3 steps.

    This is the integration test the unit tests above cannot replace: it
    exercises the EMF construction from actual Riemann-solver output, the
    staggered boundary conditions, and the Runge-Kutta combination of
    cell-centred and face-centred state together.
    """
    from src.physics.driver2d import compute_dt_2d, rk_step_ct
    from src.physics.eos import hybrid_eos

    eos = hybrid_eos(K=0.0, gamma=5.0 / 3.0, gamma_th=5.0 / 3.0)
    g = Grid2D(-0.5, 0.5, 32, -0.5, 0.5, 32, ng=2)
    st = _rotor_state(g, eos)

    d0 = float(div_b(st.Bxf, st.Byf, g.dx, g.dy)[g.phys].abs().max())
    assert d0 < 1e-12, f"initial div(B) = {d0:.3e} (vector potential broken)"

    for _ in range(4):
        dt = compute_dt_2d(st.prims, g, eos, 0.25)
        st, _ = rk_step_ct(st, g, eos, dt, scheme="rk3", emf_mode=emf_mode)

    Bmax = max(float(st.Bxf.abs().max()), float(st.Byf.abs().max()))
    dfin = float(div_b(st.Bxf, st.Byf, g.dx, g.dy)[g.phys].abs().max())
    assert dfin / (Bmax / g.dx) < 1e-13, (
        f"div(B) grew to {dfin:.3e} (normalised {dfin/(Bmax/g.dx):.3e})"
    )


def test_rotor_initial_data_matches_reference_setup():
    """Guard the published rotor parameters against silent drift."""
    from src.physics.eos import hybrid_eos
    from src.physics.initial_data2d import magnetic_rotor

    eos = hybrid_eos(K=0.0, gamma=5.0 / 3.0, gamma_th=5.0 / 3.0)
    g = Grid2D(-0.5, 0.5, 64, -0.5, 0.5, 64, ng=2)
    prims, Az = magnetic_rotor(g, eos)
    ph = g.phys
    rho = prims["rho"][ph]
    assert float(rho.max()) == 10.0 and float(rho.min()) == 1.0
    assert float(prims["p"][ph].max()) == 1.0
    assert float(prims["Bx"][ph].min()) == 1.0
    assert float(prims["By"][ph].abs().max()) == 0.0
    # v_max = r0 * omega = 0.995 at the rim  ->  W ~ 10
    v = torch.sqrt(prims["vx"][ph] ** 2 + prims["vy"][ph] ** 2)
    assert 0.985 < float(v.max()) <= 0.995
    Bxf, Byf = b_from_potential(Az, g.dx, g.dy)
    assert float((Bxf - 1.0).abs().max()) < 1e-14
    assert float(Byf.abs().max()) < 1e-14


# ── discrete symmetry of the rotor ───────────────────────────────────────

ROTOR_PARITY = {"rho": +1, "p": +1, "vx": -1, "vy": -1, "Bx": +1, "By": +1}


def rotor_symmetry_error(st, g):
    """Max relative violation of the rotor's exact pi-rotation symmetry.

    The rotor initial data is invariant under a pi rotation about z combined
    with SRMHD's B -> -B invariance, so for all time
        rho(-r)=rho(r), p(-r)=p(r), v(-r)=-v(r), Bx(-r)=+Bx(r), By(-r)=+By(r)
    """
    ph = g.phys
    worst = 0.0
    for k, par in ROTOR_PARITY.items():
        a = st.prims[k][ph]
        b = torch.flip(a, dims=(0, 1))
        worst = max(worst, float((a - par * b).abs().max())
                    / max(float(a.abs().max()), 1e-30))
    return worst


def test_upwind_sign_deadband_suppresses_noise():
    """Round-off-level fluxes must not select an upwind direction.

    sign() is discontinuous at zero, so without a deadband a 1e-12 flux --
    which is what the HLL dissipation term leaves behind in a region at
    rest -- picks a direction and injects an O(1) asymmetry.
    """
    from src.physics.ct import _upwind_sign

    F = torch.tensor([1.0, -2.0, 1e-13, -1e-13, 0.0], dtype=torch.float64)
    s = _upwind_sign(F, 1e-9)
    assert float(s[0]) == 1.0 and float(s[1]) == -1.0
    assert float(s[2]) == 0.0 and float(s[3]) == 0.0 and float(s[4]) == 0.0


def test_rotor_preserves_discrete_rotational_symmetry():
    """The evolved rotor keeps its pi-rotation symmetry to round-off.

    This is a far sharper regression test than inspecting contour plots, and
    it is exactly what a handedness in the upwind EMF selector destroys: the
    same run without the deadband drifts to ~1e-1 within a few steps.
    """
    from src.physics.driver2d import rk_step_ct
    from src.physics.eos import hybrid_eos

    eos = hybrid_eos(K=0.0, gamma=5.0 / 3.0, gamma_th=5.0 / 3.0)
    g = Grid2D(-0.5, 0.5, 32, -0.5, 0.5, 32, ng=2)
    st = _rotor_state(g, eos)

    assert rotor_symmetry_error(st, g) < 1e-13
    for _ in range(8):
        st, _ = rk_step_ct(st, g, eos, 6e-4, scheme="rk3")
    err = rotor_symmetry_error(st, g)
    assert err < 1e-9, f"pi-rotation symmetry degraded to {err:.3e}"


def test_zero_normal_field_does_not_overflow():
    """B_n = 0 is a genuine HLLD singularity and must fall back, not overflow.

    The rotor starts with By identically zero, so every y-sweep hits it.
    Before the guard this produced |F| ~ 1e268 rather than a wrong-but-finite
    answer.
    """
    from src.physics.driver2d import prims_to_cons_2d, sync_state
    from src.physics.eos import hybrid_eos
    from src.physics.sweep import sweep

    eos = hybrid_eos(K=0.0, gamma=5.0 / 3.0, gamma_th=5.0 / 3.0)
    g = Grid2D(-0.5, 0.5, 24, -0.5, 0.5, 24, ng=2)
    st = _rotor_state(g, eos)

    Fy, ay = sweep(st.prims, g, eos, 1)          # normal field is By == 0
    assert ay["diag"]["n_bn_zero"] == ay["diag"]["n_interfaces"]
    for k, v in Fy.items():
        assert torch.isfinite(v).all(), f"non-finite flux in {k}"
        assert float(v.abs().max()) < 1e6, f"{k} overflowed to {float(v.abs().max()):.2e}"


# ── Orszag-Tang: the periodic staggered seam ─────────────────────────────


def _ot_state(g, eos):
    from src.physics.driver2d import prims_to_cons_2d, sync_state
    from src.physics.initial_data2d import orszag_tang
    from src.physics.state import EVOLVED_KEYS, State2D

    prims, Az = orszag_tang(g, eos)
    Bxf, Byf = b_from_potential(Az, g.dx, g.dy)
    cons_all = prims_to_cons_2d(prims, g)
    st = State2D(cons={k: cons_all[k] for k in EVOLVED_KEYS},
                 Bxf=Bxf, Byf=Byf, prims=prims)
    return sync_state(st, g, eos, "periodic", "periodic")


def test_orszag_tang_initial_data():
    """Standard relativistic OT parameters and an exactly div-free start."""
    import math
    from src.physics.eos import hybrid_eos
    from src.physics.initial_data2d import orszag_tang

    eos = hybrid_eos(K=0.0, gamma=4.0 / 3.0, gamma_th=4.0 / 3.0)
    g = Grid2D(0.0, 1.0, 64, 0.0, 1.0, 64, ng=2)
    prims, Az = orszag_tang(g, eos)
    ph = g.phys
    assert float(prims["rho"][0, 0]) == pytest.approx(25 / (36 * math.pi))
    assert float(prims["p"][0, 0]) == pytest.approx(5 / (12 * math.pi))
    v = torch.sqrt(prims["vx"][ph] ** 2 + prims["vy"][ph] ** 2)
    assert 0.95 < float(v.max()) <= 0.99          # W ~ 7
    Bxf, Byf = b_from_potential(Az, g.dx, g.dy)
    assert float(div_b(Bxf, Byf, g.dx, g.dy)[ph].abs().max()) < 1e-13


def test_orszag_tang_periodic_seam_stays_divergence_free():
    """div(B) must not drift at the periodic seam.

    This is the specific reason Orszag-Tang is in the suite.  Under
    periodicity the staggered faces at ``ng`` and ``ng+n`` are the SAME
    physical face; if they are merely *numerically close* rather than kept
    identical, div(B) develops a seam-localised error that the rotor (with
    outflow boundaries) would never reveal.
    """
    from src.physics.driver2d import compute_dt_2d, rk_step_ct
    from src.physics.eos import hybrid_eos

    eos = hybrid_eos(K=0.0, gamma=4.0 / 3.0, gamma_th=4.0 / 3.0)
    g = Grid2D(0.0, 1.0, 32, 0.0, 1.0, 32, ng=2)
    st = _ot_state(g, eos)
    ng, nx, ny = g.ng, g.nx, g.ny

    for _ in range(10):
        dt = compute_dt_2d(st.prims, g, eos, 0.25)
        st, _ = rk_step_ct(st, g, eos, dt, scheme="rk3",
                           bc_x="periodic", bc_y="periodic")

    # the seam faces must remain bit-identical, not merely close
    assert torch.equal(st.Bxf[ng], st.Bxf[ng + nx]), "x seam faces diverged"
    assert torch.equal(st.Byf[:, ng], st.Byf[:, ng + ny]), "y seam faces diverged"

    d = div_b(st.Bxf, st.Byf, g.dx, g.dy)[g.phys]
    Bmax = max(float(st.Bxf.abs().max()), float(st.Byf.abs().max()))
    assert float(d.abs().max()) / (Bmax / g.dx) < 1e-13

    # and the error must not be concentrated at the seam
    edge = float(d[0, :].abs().max())
    bulk = float(d[nx // 4:3 * nx // 4, :].abs().max())
    assert edge <= max(20 * bulk, 1e-18), (
        f"div(B) is seam-localised: edge {edge:.2e} vs bulk {bulk:.2e}"
    )
