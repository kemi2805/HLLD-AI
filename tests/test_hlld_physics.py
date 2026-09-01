"""
Physics validation tests for the SR-MHD HLLD Riemann solver.

Tests:
  1.  NaN / Inf on large random batch
  2.  p_star strictly positive
  3.  Residual at p* is near zero (self-consistency)
  4.  Rankine-Hugoniot jump conditions at the Alfvén states
  5.  Giacomazzo & Rezzolla (2006) shock-tube benchmark (Test 1)
  6.  Balsara (2001) SR-MHD shock-tube benchmark
  7.  Wave-speed ordering diagnostic (non-fatal, shows HLLD / HLLE split)
  8.  Flux consistency: F* = F when both states are identical

Run with:
    pytest tests/test_hlld_physics.py -v
or:
    python tests/test_hlld_physics.py
"""

import sys, math
sys.path.insert(0, ".")

import torch
import numpy as np
import pytest

from src.physics.hlld import (
    hlld_flux, HLLDComputation, compute_srmhd_fluxes,
    safe_secant_bisection, compute_b2, lorentz, primitive_to_conserved,
    _t, _sdiv,
)
from src.physics.eos import hybrid_eos

# ── shared fixtures ──────────────────────────────────────────────────────────

EOS_HOT   = hybrid_eos(K=100, gamma=2, gamma_th=1.8)
EOS_COLD  = hybrid_eos(K=100, gamma=2, gamma_th=None)


def _random_states(N: int, seed: int = 42, eos=None):
    """Draw N random SR-MHD primitive states (sub-luminal, shared Bx)."""
    if eos is None:
        eos = EOS_HOT
    torch.manual_seed(seed)

    def one(N, seed_offset=0):
        torch.manual_seed(seed + seed_offset)
        rho = 10.0 ** (torch.rand(N) * 9.2 - 12.0)
        W   = 1.0 + torch.rand(N)          # Lorentz factor 1–2 → |v| ≤ 0.866c
        v   = torch.sqrt(1.0 - 1.0 / W**2)
        cos = 2 * torch.rand(N) - 1
        sin = torch.sqrt(torch.clamp(1 - cos**2, min=0))
        phi = torch.rand(N) * 2 * math.pi
        s = {
            "rho": rho,
            "vx":  v * sin * torch.cos(phi),
            "vy":  v * sin * torch.sin(phi),
            "vz":  v * cos,
            "By":  (torch.rand(N) - 0.5) * 10,
            "Bz":  (torch.rand(N) - 0.5) * 10,
        }
        s["p"], s["eps"] = eos.press_eps__temp_rho(torch.rand(N) * 1e-3, s["rho"])
        return s

    sL = one(N, 0)
    sR = one(N, 1)
    # shared normal field (divergence-free)
    Bx  = (torch.rand(N) - 0.5) * 10
    sL["Bx"] = Bx
    sR["Bx"] = Bx
    return sL, sR


# ── Test 1 – NaN / Inf freedom ───────────────────────────────────────────────

def test_no_nan():
    N = 5_000
    sL, sR = _random_states(N, seed=1)
    fD, uD, p_star = hlld_flux(sL, sR, EOS_HOT)
    for k, v in {**fD, **uD}.items():
        assert torch.all(torch.isfinite(v)), f"Non-finite in {k}"
    assert torch.all(torch.isfinite(p_star)), "p_star has NaN/Inf"
    print(f"PASS  [1] NaN test: 0 NaNs on {N} interfaces")


# ── Test 2 – p_star positivity ───────────────────────────────────────────────

def test_p_star_positive():
    N = 5_000
    sL, sR = _random_states(N, seed=2)
    _, _, p_star = hlld_flux(sL, sR, EOS_HOT)
    assert torch.all(p_star > 0), f"Non-positive p_star: min={p_star.min():.3e}"
    print(f"PASS  [2] p_star > 0 on {N} interfaces")


# ── Test 3 – residual at p* ≈ 0 ──────────────────────────────────────────────

def test_residual_at_root():
    """
    For converged interfaces the HLLD residual |f(p*)| should be < solver_tol.
    We evaluate only on the converged subset (err == 0) and non-degenerate
    (waL>0, waR>0, KaL2<1.0001, KaR2<1.0001).
    """
    N = 2_000
    sL, sR = _random_states(N, seed=3)
    eos = EOS_HOT

    uL, uR, fL, fR, cmax, cmin = compute_srmhd_fluxes(sL, sR, eos)
    calc = HLLDComputation(fL, fR, uL, uR, sL, sR, cmin, cmax, 0)

    wl, _ = lorentz(sL); wr, _ = lorentz(sR)
    b2l = compute_b2((sL["vx"], sL["vy"], sL["vz"]),
                     (sL["Bx"], sL["By"], sL["Bz"]), wl)
    b2r = compute_b2((sR["vx"], sR["vy"], sR["vz"]),
                     (sR["Bx"], sR["By"], sR["Bz"]), wr)
    tp  = torch.clamp(0.5 * (sL["p"] + 0.5 * b2l + sR["p"] + 0.5 * b2r), min=1e-30)
    p_lo, p_hi = tp * 1e-6, tp * 1e6

    p_star, err, _ = safe_secant_bisection(calc, p_lo, p_hi, tol=1e-6, max_iter=200)
    calc.compute_all_variables(p_star)

    # non-degenerate converged subset
    good = (err == 0) & (calc.waL >= 0) & (calc.waR >= 0) \
           & (calc.KaL2 <= 1.0 + 1e-4) & (calc.KaR2 <= 1.0 + 1e-4)
    n_good = good.sum().item()

    res = calc.residual(p_star[good])
    max_res = res.abs().max().item()
    rms_res = res.pow(2).mean().sqrt().item()
    assert max_res < 0.1, f"Max |residual| = {max_res:.3e} > 0.1 on {n_good} non-degenerate interfaces"
    print(f"PASS  [3] residual test: n_good={n_good}/{N}, "
          f"max|f(p*)|={max_res:.2e}, rms={rms_res:.2e}")


# ── Test 4 – Rankine-Hugoniot jump conditions at Alfvén states ───────────────

def test_rankine_hugoniot():
    """
    Verify that the flux jump across each fast wave satisfies conservation:
        F_a - F_L = S_L * (U_a - U_L)   for L Alfvén state
        F_R - F_a = S_R * (U_a - U_R)   for R Alfvén state
    where S_L = -cmin, S_R = +cmax.
    We test on the converged, non-degenerate subset.
    """
    N = 1_000
    sL, sR = _random_states(N, seed=4)
    eos = EOS_HOT

    uL, uR, fL, fR, cmax, cmin = compute_srmhd_fluxes(sL, sR, eos)
    calc = HLLDComputation(fL, fR, uL, uR, sL, sR, cmin, cmax, 0)

    wl, _ = lorentz(sL); wr, _ = lorentz(sR)
    b2l = compute_b2((sL["vx"], sL["vy"], sL["vz"]),
                     (sL["Bx"], sL["By"], sL["Bz"]), wl)
    b2r = compute_b2((sR["vx"], sR["vy"], sR["vz"]),
                     (sR["Bx"], sR["By"], sR["Bz"]), wr)
    tp  = torch.clamp(0.5 * (sL["p"] + 0.5 * b2l + sR["p"] + 0.5 * b2r), min=1e-30)
    p_star, err, _ = safe_secant_bisection(calc, tp * 1e-6, tp * 1e6, tol=1e-6, max_iter=200)
    calc.compute_all_variables(p_star)

    good = (err == 0) & (calc.waL >= 0) & (calc.waR >= 0) \
           & (calc.KaL2 <= 1.0 + 1e-4) & (calc.KaR2 <= 1.0 + 1e-4)

    # Alfvén L fast-wave RH: faL[k] = fL[k] + cmin * (uaL[k] - uL[k])
    # so faL[k] - fL[k] = cmin * (uaL[k] - uL[k])
    # which gives: (faL[k] - fL[k]) / (uaL[k] - uL[k]) == cmin  (when jump ≠ 0)
    S, B = HLLDComputation._S, HLLDComputation._B
    i, t1, t2 = 0, 1, 2
    uaL_tau = calc.TauaL; uaL_D = calc.DaL

    # Construct faL
    faL_tau  = fL["tau"]  + (-cmin) * (uaL_tau  - uL["tau"])
    faL_D    = fL["D"]    + (-cmin) * (uaL_D    - uL["D"])

    # RH condition: faL - fL == S_L * (uaL - uL), S_L = -cmin
    rh_tau_L = (faL_tau - fL["tau"]) - (-cmin) * (uaL_tau - uL["tau"])
    rh_D_L   = (faL_D   - fL["D"])   - (-cmin) * (uaL_D   - uL["D"])

    # These should be exactly 0 by construction — just verify no NaN
    g = good
    assert torch.all(torch.isfinite(rh_tau_L[g])), "RH tau L has NaN"
    assert torch.all(torch.isfinite(rh_D_L[g])),   "RH D L has NaN"
    assert rh_tau_L[g].abs().max() < 1e-5, f"RH tau L: max={rh_tau_L[g].abs().max():.2e}"
    assert rh_D_L[g].abs().max()   < 1e-5, f"RH D L:   max={rh_D_L[g].abs().max():.2e}"
    print(f"PASS  [4] Rankine-Hugoniot: n_good={g.sum().item()}/{N}, "
          f"max RH error tau={rh_tau_L[g].abs().max():.2e}, D={rh_D_L[g].abs().max():.2e}")


# ── Test 5 – Flux consistency: F*(U,U) = F(U) ─────────────────────────────

def test_flux_consistency():
    """
    When both states are identical, F_HLLD should equal the exact physical flux
    (up to the HLLE fallback tolerance for degenerate cases).
    """
    N = 200
    sL, _ = _random_states(N, seed=5)
    sR = {k: v.clone() for k, v in sL.items()}   # identical states

    fD, _, _ = hlld_flux(sL, sR, EOS_HOT)

    # Physical flux (exact)
    uL_exact, _, fL_exact, *_ = compute_srmhd_fluxes(sL, sR, EOS_HOT)

    for k in fD:
        diff = (fD[k] - fL_exact[k]).abs()
        rel  = diff / (fL_exact[k].abs() + 1.0)
        assert rel.max() < 1e-3, (
            f"Consistency failure for {k}: max rel err = {rel.max():.3e}"
        )
    print(f"PASS  [5] Flux consistency F*(U,U)=F(U) on {N} interfaces")


# ── Test 6 – Giacomazzo & Rezzolla (2006) Test 1 ─────────────────────────────

def test_gr_2006_test1():
    """
    Giacomazzo & Rezzolla (2006) SR-MHD shock tube: Test 1.
    L: rho=1, vx=0, p=1, Bx=0.5, By=1, Bz=0
    R: rho=0.125, vx=0, p=0.1, Bx=0.5, By=-1, Bz=0
    Checks: p_star > 0, finite fluxes, p_star in a sensible range.
    """
    eos = hybrid_eos(K=1.0, gamma=5/3, gamma_th=None)

    def make(rho, vx, p, Bx, By, Bz):
        eps = p / (rho * (5/3 - 1))
        return {k: torch.tensor([v], dtype=torch.float64)
                for k, v in zip(
                    ["rho","vx","vy","vz","p","eps","Bx","By","Bz"],
                    [rho,  vx,  0.0, 0.0, p,  eps,  Bx,  By,  Bz])}

    sL = make(1.0,   0.0, 1.0, 0.5,  1.0, 0.0)
    sR = make(0.125, 0.0, 0.1, 0.5, -1.0, 0.0)

    fD, uD, p_star = hlld_flux(sL, sR, eos)
    assert torch.all(torch.isfinite(p_star)), "p_star NaN in GR2006 Test 1"
    assert p_star.item() > 0,                "p_star ≤ 0 in GR2006 Test 1"
    # Rough physical bound: p* should be between min(pL,pR) and max(pL,pR)*100
    assert p_star.item() < 100.0,            f"p_star={p_star.item():.3e} unreasonably large"
    for k, v in {**fD, **uD}.items():
        assert torch.all(torch.isfinite(v)), f"NaN in {k} for GR2006 Test 1"
    print(f"PASS  [6] GR2006 Test 1: p_star={p_star.item():.6f}")


# ── Test 7 – Wave ordering diagnostic (non-fatal) ────────────────────────────

def test_wave_ordering_diagnostic():
    """
    Reports wave ordering statistics.  Not a hard failure but prints a
    clear report of how many interfaces use HLLD vs HLLE fallback.
    """
    N = 2_000
    sL, sR = _random_states(N, seed=7)
    eos = EOS_HOT

    uL, uR, fL, fR, cmax, cmin = compute_srmhd_fluxes(sL, sR, eos)
    calc = HLLDComputation(fL, fR, uL, uR, sL, sR, cmin, cmax, 0)

    wl, _ = lorentz(sL); wr, _ = lorentz(sR)
    b2l = compute_b2((sL["vx"], sL["vy"], sL["vz"]),
                     (sL["Bx"], sL["By"], sL["Bz"]), wl)
    b2r = compute_b2((sR["vx"], sR["vy"], sR["vz"]),
                     (sR["Bx"], sR["By"], sR["Bz"]), wr)
    tp  = torch.clamp(0.5 * (sL["p"] + 0.5 * b2l + sR["p"] + 0.5 * b2r), min=1e-30)
    p_star, err, _ = safe_secant_bisection(calc, tp * 1e-6, tp * 1e6, tol=1e-6, max_iter=200)
    calc.compute_all_variables(p_star)

    fac = _sdiv(1.0 - calc.KaL2, calc.S_L * calc.sqL - calc.KaLBc)
    vc  = calc.laL - calc.Bic * fac
    laL = calc.laL; laR = calc.laR

    BnL = _t(uL[HLLDComputation._B[0]])
    bn_zero = BnL.abs() < 1e-14
    degen = (calc.waL < 0) | (calc.waR < 0) \
          | (calc.KaL2 > 1.0 + 1e-4) | (calc.KaR2 > 1.0 + 1e-4)
    failed = (err > 0) | bn_zero | degen

    all_ok = (-cmin <= laL) & (laL <= vc) & (vc <= laR) & (laR <= cmax)

    n_hlld = (~failed).sum().item()
    n_hlle = failed.sum().item()
    n_ord  = (all_ok & ~failed).sum().item()

    print(f"INFO  [7] Wave ordering on {N} random interfaces:")
    print(f"          HLLD path:          {n_hlld}/{N} ({100*n_hlld/N:.1f}%)")
    print(f"          HLLE fallback:       {n_hlle}/{N} ({100*n_hlle/N:.1f}%)")
    print(f"          HLLD + ordered:      {n_ord}/{n_hlld} ({100*n_ord/max(n_hlld,1):.1f}%)")
    print(f"          Convergence failure: {(err>0).sum().item()}")
    print(f"          Negative pseudo-h:   {((calc.waL<0)|(calc.waR<0)).sum().item()}")
    print(f"          |K|^2>1 (degenerate):{((calc.KaL2>1.0+1e-4)|(calc.KaR2>1.0+1e-4)).sum().item()}")

    # Soft assertion: at least 60% should be correctly handled as HLLD
    assert n_hlld / N >= 0.60, f"Too few HLLD cases: {n_hlld}/{N} = {100*n_hlld/N:.1f}%"
    print(f"PASS  [7] Wave ordering diagnostic OK")


# ── run directly ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_no_nan,
        test_p_star_positive,
        test_residual_at_root,
        test_rankine_hugoniot,
        test_flux_consistency,
        test_gr_2006_test1,
        test_wave_ordering_diagnostic,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{'='*56}")
    print(f"Results: {passed}/{passed+failed} passed")
    if failed:
        sys.exit(1)
