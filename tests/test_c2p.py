"""
Round-trip test for the Kastaun C2P:

    primitives  →  P2C  →  conserved  →  C2P  →  recovered primitives

Tests:
  1.  Pure hydro (B = 0)         — large random batch
  2.  Full MHD  (B ≠ 0)          — large random batch
  3.  Mildly relativistic        — W in [1, 2]
  4.  Highly relativistic        — W in [2, 10]
  5.  Low-density / atmosphere   — rho in [1e-10, 1e-6]
  6.  Strong magnetic field      — |B| >> p
  7.  Single known state         — exact values, hand-checkable

Run with:
    pytest tests/test_c2p.py -v
or:
    python tests/test_c2p.py
"""

import sys, math
sys.path.insert(0, ".")

import torch
import pytest

from src.physics.eos  import hybrid_eos
from src.physics.hlld import primitive_to_conserved
from src.physics.c2p  import conservative_to_primitive


# ── shared EOS fixtures ───────────────────────────────────────────────────────

EOS_IDEAL = hybrid_eos(K=1.0, gamma=5/3, gamma_th=None)   # cold polytrope
EOS_HOT   = hybrid_eos(K=1.0, gamma=5/3, gamma_th=4/3)    # hybrid

DTYPE  = torch.float64
DEVICE = "cpu"


# ── primitive-state builder ───────────────────────────────────────────────────

def _make_prims(
    N:   int,
    eos: hybrid_eos,
    W_range:  tuple = (1.0, 2.0),
    rho_range: tuple = (1e-3, 1e2),
    B_mag:    float  = 1.0,
    zero_B:   bool   = False,
    seed:     int    = 0,
) -> dict[str, torch.Tensor]:
    """
    Sample N random sub-luminal primitive states.

    rho   : log-uniform in rho_range
    W     : uniform in W_range  (Lorentz factor → v = sqrt(1 - 1/W²))
    p/eps : from EOS cold branch (press_cold_eps_cold__rho)
    B     : isotropic direction, magnitude uniform in [0, B_mag]
    """
    torch.manual_seed(seed)

    log_lo, log_hi = math.log10(rho_range[0]), math.log10(rho_range[1])
    rho = 10.0 ** (log_lo + (log_hi - log_lo) * torch.rand(N, dtype=DTYPE, device=DEVICE))

    # cold eps/p as starting point — then optionally add thermal
    p_cold, eps_cold = eos.press_cold_eps_cold__rho(rho)
    if eos.is_cold:
        p   = p_cold
        eps = eps_cold
    else:
        # add a small thermal component
        temp = torch.rand(N, dtype=DTYPE, device=DEVICE) * 1e-2
        p, eps = eos.press_eps__temp_rho(temp, rho)

    # velocity
    W_lo, W_hi = W_range
    W = W_lo + (W_hi - W_lo) * torch.rand(N, dtype=DTYPE, device=DEVICE)
    v = torch.sqrt(1.0 - 1.0 / W**2)

    cos_t = 2.0 * torch.rand(N, dtype=DTYPE, device=DEVICE) - 1.0
    sin_t = torch.sqrt(torch.clamp(1.0 - cos_t**2, min=0.0))
    phi   = 2.0 * math.pi * torch.rand(N, dtype=DTYPE, device=DEVICE)

    vx = v * sin_t * torch.cos(phi)
    vy = v * sin_t * torch.sin(phi)
    vz = v * cos_t

    # magnetic field
    if zero_B:
        Bx = By = Bz = torch.zeros(N, dtype=DTYPE, device=DEVICE)
    else:
        B_mag_t = B_mag * torch.rand(N, dtype=DTYPE, device=DEVICE)
        cos_b   = 2.0 * torch.rand(N, dtype=DTYPE, device=DEVICE) - 1.0
        sin_b   = torch.sqrt(torch.clamp(1.0 - cos_b**2, min=0.0))
        phi_b   = 2.0 * math.pi * torch.rand(N, dtype=DTYPE, device=DEVICE)
        Bx = B_mag_t * sin_b * torch.cos(phi_b)
        By = B_mag_t * sin_b * torch.sin(phi_b)
        Bz = B_mag_t * cos_b

    return {
        "rho": rho, "vx": vx, "vy": vy, "vz": vz,
        "p": p, "eps": eps,
        "Bx": Bx, "By": By, "Bz": Bz,
    }


# ── round-trip check ──────────────────────────────────────────────────────────

def _roundtrip(
    prims_in: dict[str, torch.Tensor],
    eos:      hybrid_eos,
    rho_tol:  float = 1e-6,
    v_tol:    float = 1e-6,
    eps_tol:  float = 1e-5,
    label:    str   = "",
) -> None:
    """
    P2C → C2P and assert recovered primitives match originals.
    Prints per-variable max relative errors.
    """
    cons = primitive_to_conserved(prims_in)
    prims_out = conservative_to_primitive(cons, eos, atmo_rho=1e-12)

    N = prims_in["rho"].shape[0]

    def _rel(key, out=prims_out, ref=prims_in):
        err = (out[key] - ref[key]).abs()
        denom = ref[key].abs().clamp(min=1e-30)
        return (err / denom).max().item()

    def _abs(key, out=prims_out, ref=prims_in):
        return (out[key] - ref[key]).abs().max().item()

    rho_err = _rel("rho")
    vx_err  = _abs("vx")
    vy_err  = _abs("vy")
    vz_err  = _abs("vz")
    eps_err = _rel("eps")

    tag = f"[{label}  N={N}]"
    print(f"  {tag}  rho_rel={rho_err:.2e}  "
          f"vx_abs={vx_err:.2e}  vy_abs={vy_err:.2e}  vz_abs={vz_err:.2e}  "
          f"eps_rel={eps_err:.2e}")

    assert rho_err < rho_tol, f"{tag} rho rel error {rho_err:.2e} > {rho_tol}"
    assert vx_err  < v_tol,   f"{tag} vx  abs error {vx_err:.2e}  > {v_tol}"
    assert vy_err  < v_tol,   f"{tag} vy  abs error {vy_err:.2e}  > {v_tol}"
    assert vz_err  < v_tol,   f"{tag} vz  abs error {vz_err:.2e}  > {v_tol}"
    assert eps_err < eps_tol, f"{tag} eps rel error {eps_err:.2e} > {eps_tol}"


# ── Test 1 – pure hydro (B = 0) ───────────────────────────────────────────────

def test_c2p_hydro_cold():
    prims = _make_prims(2000, EOS_IDEAL, zero_B=True, seed=1)
    _roundtrip(prims, EOS_IDEAL, label="hydro-cold B=0")

def test_c2p_hydro_hot():
    prims = _make_prims(2000, EOS_HOT, zero_B=True, seed=2)
    _roundtrip(prims, EOS_HOT, label="hydro-hot B=0")


# ── Test 2 – full MHD ─────────────────────────────────────────────────────────

def test_c2p_mhd_cold():
    prims = _make_prims(2000, EOS_IDEAL, B_mag=5.0, seed=3)
    _roundtrip(prims, EOS_IDEAL, label="MHD-cold")

def test_c2p_mhd_hot():
    prims = _make_prims(2000, EOS_HOT, B_mag=5.0, seed=4)
    _roundtrip(prims, EOS_HOT, label="MHD-hot")


# ── Test 3 – mildly relativistic ─────────────────────────────────────────────

def test_c2p_mildly_relativistic():
    prims = _make_prims(2000, EOS_HOT, W_range=(1.0, 2.0), B_mag=3.0, seed=5)
    _roundtrip(prims, EOS_HOT, label="W=[1,2]")


# ── Test 4 – highly relativistic ─────────────────────────────────────────────

def test_c2p_highly_relativistic():
    prims = _make_prims(2000, EOS_HOT, W_range=(2.0, 10.0), B_mag=3.0, seed=6)
    _roundtrip(prims, EOS_HOT, label="W=[2,10]", rho_tol=1e-5, v_tol=1e-5, eps_tol=1e-4)


# ── Test 5 – low-density (near atmosphere) ────────────────────────────────────

def test_c2p_low_density():
    prims = _make_prims(500, EOS_HOT,
                        rho_range=(1e-8, 1e-4), B_mag=0.1,
                        seed=7)
    _roundtrip(prims, EOS_HOT, label="low-rho",
               rho_tol=1e-5, v_tol=1e-5, eps_tol=1e-4)


# ── Test 6 – magnetically dominated ──────────────────────────────────────────

def test_c2p_magnetically_dominated():
    """B² >> rho*h — the regime most likely to stress the Kastaun inversion."""
    prims = _make_prims(1000, EOS_HOT,
                        rho_range=(1e-3, 1e-1), B_mag=20.0,
                        seed=8)
    _roundtrip(prims, EOS_HOT, label="B-dominated",
               rho_tol=1e-5, v_tol=1e-5, eps_tol=1e-4)


# ── Test 7 – single known state ───────────────────────────────────────────────

def test_c2p_known_state():
    """
    Exact single-cell round-trip with hand-chosen values.
    Komissarov (1999) shock-tube 1, left state.
    rho=1, vx=0, p=1, eps=1.5, Bx=1, By=1, Bz=0  →  W=1
    """
    eos = hybrid_eos(K=1.0, gamma=5/3, gamma_th=4/3)

    def t(v):
        return torch.tensor([v], dtype=DTYPE, device=DEVICE)

    prims_in = {
        "rho": t(1.0),  "vx": t(0.0), "vy": t(0.0), "vz": t(0.0),
        "p":   t(1.0),  "eps": t(1.5),
        "Bx":  t(1.0),  "By": t(1.0), "Bz": t(0.0),
    }

    cons = primitive_to_conserved(prims_in)
    prims_out = conservative_to_primitive(cons, eos, atmo_rho=1e-12)

    for key in ("rho", "vx", "vy", "vz", "eps"):
        ref = prims_in[key].item()
        rec = prims_out[key].item()
        rel = abs(rec - ref) / (abs(ref) + 1e-30)
        print(f"  [known state]  {key:4s}  ref={ref:.6f}  rec={rec:.6f}  rel={rel:.2e}")
        assert rel < 1e-6 or abs(rec - ref) < 1e-9, \
            f"{key}: ref={ref:.6f}  recovered={rec:.6f}  rel_err={rel:.2e}"

    print("PASS  [7] known state (Komissarov left state)")


# ── run directly ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_c2p_hydro_cold,
        test_c2p_hydro_hot,
        test_c2p_mhd_cold,
        test_c2p_mhd_hot,
        test_c2p_mildly_relativistic,
        test_c2p_highly_relativistic,
        test_c2p_low_density,
        test_c2p_magnetically_dominated,
        test_c2p_known_state,
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
