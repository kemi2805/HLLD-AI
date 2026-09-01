"""
The exact Riemann solver used as a Godunov flux (scalar reference).

Ground truth comes from ``riemann_dataset.generate_dataset(...,
xi_window=(0.0, 0.0))``, which forward-constructs solutions from a left state
plus seven wave parameters -- exact by construction, no root-find -- and keeps
only those whose fan straddles the interface ray.  The exact state at xi = 0
is therefore known independently of anything the flux function does.

These tests are slow (seconds per interface) because this is the SCALAR
reference: its job is to pin the physics down and be the answer the batched
kernels must reproduce, not to be fast.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_RMHD = "/Users/miler/Codes/rmhd_final"
if _RMHD not in sys.path:
    sys.path.append(_RMHD)

from src.physics.eos import hybrid_eos
from src.physics.hlld import LAST_DIAG, compute_srmhd_fluxes, hlld_flux
from src.physics.exact_flux import exact_flux, relative_jump

GAMMA = 5.0 / 3.0


@pytest.fixture(scope="module")
def eos():
    return hybrid_eos(K=0.0, gamma=GAMMA, gamma_th=GAMMA)


def _prim(state, Bn, eos):
    """Solver-frame 7-vector (TOTAL pressure) -> HLLD primitive dict."""
    rho, Ptot, vn, vt1, vt2, Bt1, Bt2 = state
    v2 = vn * vn + vt1 * vt1 + vt2 * vt2
    W2 = 1.0 / (1.0 - v2)
    eta = Bn * vn + Bt1 * vt1 + Bt2 * vt2
    b2 = (Bn * Bn + Bt1 * Bt1 + Bt2 * Bt2) / W2 + eta * eta
    t = lambda v: torch.tensor([v], dtype=torch.float64)
    d = {"rho": t(rho), "vx": t(vn), "vy": t(vt1), "vz": t(vt2),
         "p": t(Ptot - 0.5 * b2), "Bx": t(Bn), "By": t(Bt1), "Bz": t(Bt2)}
    d["eps"] = eos.eps__press_rho(d["p"], d["rho"])
    return d


@pytest.fixture(scope="module")
def cases():
    from batched.ray_scalar import claim_rmhd_namespace
    claim_rmhd_namespace()
    from eos import set_eos
    set_eos("ideal")
    from riemann_dataset import generate_dataset
    return generate_dataset(6, gamma=GAMMA, Bx_range=(0.5, 3.0), seed=777,
                            xi_window=(0.0, 0.0), verbose=False)


def test_rejects_a_non_ideal_eos(eos):
    """Solving a different EOS than intended produces plausible output.

    So the mismatch is refused rather than discovered later from bad physics:
    rmhd_final's enthalpy is the ideal-gas one, which coincides with
    hybrid_eos only for K = 0 and gamma_th == gamma.
    """
    bad = hybrid_eos(K=100.0, gamma=GAMMA, gamma_th=GAMMA)
    n = 2
    s = {k: torch.ones(n, dtype=torch.float64) for k in
         ("rho", "vx", "vy", "vz", "p", "Bx", "By", "Bz", "eps")}
    for k in ("vx", "vy", "vz"):
        s[k] = torch.zeros(n, dtype=torch.float64)
    with pytest.raises(ValueError, match="ideal gas"):
        exact_flux(s, s, bad, idir=0)


def test_uniform_state_gives_the_physical_flux(eos):
    """L == R must reproduce F(U) exactly (via the weak gate, which is right)."""
    n = 4
    torch.manual_seed(1)
    s = {"rho": torch.rand(n, dtype=torch.float64) + 0.5,
         "vx": torch.rand(n, dtype=torch.float64) * 0.3,
         "vy": torch.rand(n, dtype=torch.float64) * 0.2,
         "vz": torch.zeros(n, dtype=torch.float64),
         "p": torch.rand(n, dtype=torch.float64) + 0.5,
         "Bx": torch.ones(n, dtype=torch.float64),
         "By": torch.rand(n, dtype=torch.float64) * 0.5 + 0.3,
         "Bz": torch.rand(n, dtype=torch.float64) * 0.3}
    s["eps"] = eos.eps__press_rho(s["p"], s["rho"])
    _, _, F_ref, _, _, _ = compute_srmhd_fluxes(s, s, eos, 0)
    F, _, _ = exact_flux(s, {k: v.clone() for k, v in s.items()}, eos, idir=0)
    worst = max(float((F[k] - F_ref[k]).abs().max()) for k in F)
    assert worst < 1e-9, f"uniform state gave |F - F(U)| = {worst:.3e}"


def test_weak_jump_is_gated_to_hlld(eos):
    """Near-uniform interfaces must not reach the exact solver.

    There the exact and linearised fluxes agree to O(jump^2), while the exact
    solver would be both wasteful and ill-conditioned: multi-root ranking by
    RH residual is meaningless when every residual is at round-off.
    """
    n = 4
    torch.manual_seed(2)
    sL = {"rho": torch.rand(n, dtype=torch.float64) + 1.0,
          "vx": torch.rand(n, dtype=torch.float64) * 0.2,
          "vy": torch.zeros(n, dtype=torch.float64),
          "vz": torch.zeros(n, dtype=torch.float64),
          "p": torch.rand(n, dtype=torch.float64) + 1.0,
          "Bx": torch.ones(n, dtype=torch.float64),
          "By": torch.full((n,), 0.5, dtype=torch.float64),
          "Bz": torch.zeros(n, dtype=torch.float64)}
    sL["eps"] = eos.eps__press_rho(sL["p"], sL["rho"])
    sR = {k: v.clone() for k, v in sL.items()}
    sR["rho"] = sL["rho"] * (1.0 + 1e-10)      # far below tau_weak
    sR["eps"] = eos.eps__press_rho(sR["p"], sR["rho"])
    assert float(relative_jump(sL, sR).max()) < 1e-6
    exact_flux(sL, sR, eos, idir=0)
    d = dict(LAST_DIAG)
    assert d["n_weak_gate"] == n and d["n_exact"] == 0


def test_exact_flux_beats_hlld_against_constructed_truth(eos, cases):
    """The point of the whole exercise, measured.

    For each forward-constructed solution the exact state at xi = 0 is known,
    so the true Godunov flux is known.  Where the exact solver converges its
    flux must be far closer to that truth than HLLD's.
    """
    from batched import ray_scalar as RS

    n_exact, worst_exact, hlld_errs = 0, 0.0, []
    for s in cases:
        left, right = list(s.zones[0]), list(s.zones[7])
        zones = [list(z) for z in s.zones[1:7]]
        VsLv = [s.speeds[0], s.speeds[1], s.speeds[2]]
        VsRv = [s.speeds[6], s.speeds[5], s.speeds[4]]
        st_true, _ = RS.state_at_xi(left, right, zones, VsLv, VsRv, s.Bx,
                                    GAMMA, nsamples=800)
        pt = _prim(st_true, s.Bx, eos)
        _, _, F_true, _, _, _ = compute_srmhd_fluxes(pt, pt, eos, 0)

        pL, pR = _prim(left, s.Bx, eos), _prim(right, s.Bx, eos)
        Fe, _, _ = exact_flux(pL, pR, eos, idir=0)
        d = dict(LAST_DIAG)
        if not d["n_exact"]:
            continue
        Fh, _, _ = hlld_flux(pL, pR, eos, idir=0)
        sc = max(max(float(F_true[q].abs().max()) for q in F_true), 1e-30)
        ee = max(float((Fe[q] - F_true[q]).abs().max()) for q in Fe) / sc
        eh = max(float((Fh[q] - F_true[q]).abs().max()) for q in Fh) / sc
        n_exact += 1
        worst_exact = max(worst_exact, ee)
        hlld_errs.append(eh)

    assert n_exact >= 2, f"exact path engaged on only {n_exact} cases"
    assert worst_exact < 1e-7, (
        f"exact flux error {worst_exact:.3e} against constructed truth"
    )
    assert max(hlld_errs) > 100 * worst_exact, (
        "HLLD was not meaningfully worse, so this test proves nothing"
    )


def test_diagnostics_are_reported(eos, cases):
    """Fallback must be measured, not hidden."""
    s = cases[0]
    pL = _prim(list(s.zones[0]), s.Bx, eos)
    pR = _prim(list(s.zones[7]), s.Bx, eos)
    exact_flux(pL, pR, eos, idir=0)
    d = dict(LAST_DIAG)
    for key in ("solver", "n_interfaces", "n_weak_gate", "n_exact",
                "n_hlld_fallback", "frac_hlld_fallback", "n_fan_interior",
                "n_full7", "n_coplanar", "n_alfven_degen"):
        assert key in d, f"LAST_DIAG missing {key}"
    assert d["solver"] == "exact"
    assert d["n_exact"] + d["n_hlld_fallback"] == d["n_interfaces"]
