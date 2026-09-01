"""
2D directional sweeps: rotation equivalence and 1D degeneracy.

These are the tests that certify the 2D fluid machinery before constrained
transport is added.  The strongest of them is ``test_rotated_shocktube``: an
entire evolution run along y must reproduce the same run along x, component
for component.  It exercises reconstruction, the Riemann solve, the flux
divergence, the boundary conditions and the Runge-Kutta update all at once,
in a way a single-sweep check cannot.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.physics.eos import hybrid_eos
from src.physics.grid import Grid1D, Grid2D
from src.physics.driver2d import (
    compute_dt_2d, cons_to_prims_2d, prims_to_cons_2d, rk_step,
)
from src.physics.initial_data2d import shocktube_2d
from src.physics.hlld import hlle_flux, hlld_flux

GAMMA = 2.0
NG = 2

# ST1 (Mattia & Mignone 2021): co-planar relativistic Brio-Wu
ST1_L = dict(rho=1.0, p=1.0, vx=0.0, vy=0.0, vz=0.0, Bx=0.5, By=1.0, Bz=0.0)
ST1_R = dict(rho=0.125, p=0.1, vx=0.0, vy=0.0, vz=0.0, Bx=0.5, By=-1.0, Bz=0.0)


def cyc_state(s: dict, n: int = 1) -> dict:
    """Cyclic component relabelling C^n: component i <- component (i-n)%3."""
    out = dict(s)
    for tri in (("vx", "vy", "vz"), ("Bx", "By", "Bz")):
        for i, k in enumerate(tri):
            out[k] = s[tri[(i - n) % 3]]
    return out


def evolve(grid, eos, prims, nsteps, *, dt=None, cfl=0.4, flux_fn=hlld_flux,
           limiter="mc", bc_x="outflow", bc_y="outflow"):
    """Evolve ``nsteps``.  A fixed ``dt`` may be supplied.

    The rotation test MUST pass a fixed dt.  ``compute_dt_2d`` is correctly
    not invariant under the cyclic relabelling: that map sends vy -> vz, and
    a 2D CFL condition only sees vx and vy, so an x-run limited by
    ``max(|vx|+cf) + max(|vy|+cf)`` becomes a y-run limited by
    ``max(|vz|+cf) + max(|vx|+cf)``.  For coplanar data (vz = 0) those differ.
    Letting each run choose its own step would conflate that with the
    spatial discretisation the test is actually checking.
    """
    cons = prims_to_cons_2d(prims, grid)
    grid.apply_bc_cc(cons, bc_x, bc_y)
    grid.apply_bc_cc(prims, bc_x, bc_y)
    for _ in range(nsteps):
        step = compute_dt_2d(prims, grid, eos, cfl) if dt is None else dt
        cons, prims, _ = rk_step(cons, prims, grid, eos, step, scheme="rk2",
                                 bc_x=bc_x, bc_y=bc_y, flux_fn=flux_fn,
                                 limiter=limiter)
    return cons, prims


@pytest.mark.parametrize("limiter", ["pcm", "mc"])
@pytest.mark.parametrize("flux_fn", [hlle_flux, hlld_flux], ids=["hlle", "hlld"])
def test_rotated_shocktube(flux_fn, limiter):
    """A tube run along y equals the same tube run along x, after relabelling.

    Under the cyclic relabelling C (new_x = old_z, new_y = old_x,
    new_z = old_y) the x-axis becomes the y-axis, so a tube whose normal is
    y, built from C-transformed states, must evolve into the C-transform of
    the x-tube's evolution — with the spatial profile along y matching the
    profile along x.
    """
    eos = hybrid_eos(K=0.0, gamma=GAMMA, gamma_th=GAMMA)
    n = 48
    nsteps = 25
    dt = 3.0e-3          # fixed: see evolve() docstring

    gx = Grid2D(0.0, 1.0, n, 0.0, 1.0, n, ng=NG)
    px, _ = shocktube_2d(gx, eos, ST1_L, ST1_R, direction="x", interface=0.5)
    _, px = evolve(gx, eos, px, nsteps, dt=dt, flux_fn=flux_fn, limiter=limiter)

    gy = Grid2D(0.0, 1.0, n, 0.0, 1.0, n, ng=NG)
    py, _ = shocktube_2d(gy, eos, cyc_state(ST1_L), cyc_state(ST1_R),
                         direction="y", interface=0.5)
    _, py = evolve(gy, eos, py, nsteps, dt=dt, flux_fn=flux_fn, limiter=limiter)

    # invert the relabelling on the y-run (C^-1 = C^2)
    py_back = cyc_state(py, 2)

    ph = gx.phys
    worst, offender = 0.0, None
    for k in ("rho", "p", "vx", "vy", "vz", "Bx", "By", "Bz"):
        a = px[k][ph]            # varies along axis 0
        b = py_back[k][ph].T     # varies along axis 0 after transpose
        scale = max(float(a.abs().max()), 1e-30)
        r = float((a - b).abs().max() / scale)
        if r > worst:
            worst, offender = r, k
    assert worst < 1e-12, (
        f"{flux_fn.__name__}/{limiter}: y-run differs from x-run by "
        f"{worst:.3e} (worst variable: {offender})"
    )


@pytest.mark.parametrize("limiter", ["pcm", "mc"])
def test_uniform_in_transverse_direction(limiter):
    """A tube along x must stay exactly uniform in y."""
    eos = hybrid_eos(K=0.0, gamma=GAMMA, gamma_th=GAMMA)
    g = Grid2D(0.0, 1.0, 40, 0.0, 1.0, 8, ng=NG)
    prims, _ = shocktube_2d(g, eos, ST1_L, ST1_R, direction="x")
    _, prims = evolve(g, eos, prims, 15, limiter=limiter)
    ph = g.phys
    for k in ("rho", "p", "vx", "By"):
        col = prims[k][ph]
        spread = float((col - col[:, :1]).abs().max())
        assert spread == 0.0, f"{k} varies along y by {spread:.3e}"


def test_matches_1d_solver():
    """The 2D code reduces to the 1D code on a y-uniform problem.

    Runs the same shock tube through driver.rk2_step on a Grid1D and through
    the 2D unsplit stepper, with reconstruction and time step matched.
    """
    from src.physics import driver as d1
    import src.physics.hlld as H

    eos = hybrid_eos(K=0.0, gamma=GAMMA, gamma_th=GAMMA)
    n, nsteps, dt = 64, 20, 2.0e-3

    # ── 1D reference ─────────────────────────────────────────────────
    g1 = Grid1D(0.0, 1.0, n, ng=NG)
    from src.physics import initial_data as ID
    pL, pR, xi = ID.generic(ST1_L, ST1_R, 0.5, dtype=torch.float64,
                            device="cpu", eos=eos)
    p1 = ID.init_shocktube(g1, pL, pR, xi)
    c1 = d1.prims_to_cons_grid(p1, g1)
    g1.apply_outflow_bc(p1)
    g1.apply_outflow_bc(c1)
    dt_t = torch.tensor(dt, dtype=torch.float64)
    for _ in range(nsteps):
        c1, p1 = d1.rk2_step(c1, p1, g1, eos, dt_t, 1e-10, flux_fn=hlld_flux,
                             limiter="pcm")

    # ── 2D, one physical row in y, periodic there ────────────────────
    g2 = Grid2D(0.0, 1.0, n, 0.0, 1.0, 1, ng=NG)
    p2, _ = shocktube_2d(g2, eos, ST1_L, ST1_R, direction="x")
    c2 = prims_to_cons_2d(p2, g2)
    g2.apply_bc_cc(c2, "outflow", "periodic")
    g2.apply_bc_cc(p2, "outflow", "periodic")
    for _ in range(nsteps):
        c2, p2, _ = rk_step(c2, p2, g2, eos, dt, scheme="rk2",
                            bc_x="outflow", bc_y="periodic",
                            flux_fn=hlld_flux, limiter="pcm",
                            vel_var="v")

    worst, offender = 0.0, None
    for k in ("rho", "p", "vx", "vy", "By"):
        a = p1[k][g1.phys]
        b = p2[k][g2.phys][:, 0]
        scale = max(float(a.abs().max()), 1e-30)
        r = float((a - b).abs().max() / scale)
        if r > worst:
            worst, offender = r, k
    assert worst < 1e-12, (
        f"2D does not reduce to 1D: {worst:.3e} (worst: {offender})"
    )
