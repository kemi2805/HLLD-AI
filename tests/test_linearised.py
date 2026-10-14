"""The linearised (Roe-type) solver: consistent, second order in the jump
where HLLD is first order, and confined to the faces below the gate.

The order claim is measured against the exact solver on coplanar problems
whose discontinuity is shrunk about the mean state by 1/3 per level.
"""
import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.physics.eos import hybrid_eos                       # noqa: E402
from src.physics.hlld import compute_srmhd_fluxes, hlld_flux  # noqa: E402
from src.physics.linearised import linearised_flux, LAST_LINEAR  # noqa: E402

GAMMA = 5.0 / 3.0
KEYS = ("D", "Sx", "Sy", "Sz", "tau", "Bx", "By", "Bz")


@pytest.fixture(scope="module")
def eos():
    return hybrid_eos(K=0.0, gamma=GAMMA, gamma_th=GAMMA)


def _prim(eos, **kw):
    d = {k: torch.tensor(v, dtype=torch.float64) for k, v in kw.items()}
    d["eps"] = eos.eps__press_rho(d["p"], d["rho"])
    return d


@pytest.fixture(scope="module")
def pairs(eos):
    """Twelve coplanar Riemann problems, as (L, R) primitive dicts."""
    from rmhd.eos import set_eos
    set_eos("ideal")
    from rmhd.riemann_dataset import generate_dataset
    sols = generate_dataset(12, gamma=GAMMA, Bx_range=(0.05, 1.5), seed=91,
                            xi_window=(0.0, 0.0), coplanar_frac=1.0,
                            verbose=False)
    Z = np.array([s.zones for s in sols])
    Bx = np.array([s.Bx for s in sols])

    def prim(slot):
        rho, Pt, vn, v1, v2_, B1, B2 = (Z[:, slot, j] for j in range(7))
        v2 = vn ** 2 + v1 ** 2 + v2_ ** 2
        W2 = 1.0 / (1.0 - v2)
        eta = Bx * vn + B1 * v1 + B2 * v2_
        b2 = (Bx ** 2 + B1 ** 2 + B2 ** 2) / W2 + eta ** 2
        return _prim(eos, rho=rho, p=Pt - 0.5 * b2, vx=vn, vy=v1, vz=v2_,
                     Bx=Bx, By=B1, Bz=B2)
    return prim(0), prim(7)


def _shrink(eos, L, R, s):
    """The discontinuity scaled by ``s`` about the mean, per variable."""
    L2, R2 = {}, {}
    for k in ("rho", "p", "vx", "vy", "vz", "By", "Bz"):
        m = 0.5 * (L[k] + R[k])
        L2[k] = m + s * (L[k] - m)
        R2[k] = m + s * (R[k] - m)
    L2["Bx"], R2["Bx"] = L["Bx"], R["Bx"]
    L2["eps"] = eos.eps__press_rho(L2["p"], L2["rho"])
    R2["eps"] = eos.eps__press_rho(R2["p"], R2["rho"])
    return L2, R2


def _vec(F):
    return np.stack([F[k].detach().numpy().reshape(-1) for k in KEYS])


def _rel(Fa, Fb):
    return np.sqrt(((Fa - Fb) ** 2).sum(0)) / np.sqrt((Fb ** 2).sum(0))


def test_zero_jump_is_the_physical_flux(eos, pairs):
    L, _ = pairs
    F, U, p, ok = linearised_flux(L, L, eos, 0, return_mask=True)
    assert ok.all()
    _, _, f, _, _, _ = compute_srmhd_fluxes(L, L, eos, 0)
    for k in KEYS:
        assert torch.equal(F[k], f[k])


def test_the_same_face_in_every_sweep_direction(eos, pairs):
    """A physical interface gives the same flux whichever axis is normal."""
    L, R = pairs
    Fx = _vec(linearised_flux(L, R, eos, 0)[0])
    perm = {"rho": "rho", "p": "p", "eps": "eps", "vx": "vy", "vy": "vx",
            "vz": "vz", "Bx": "By", "By": "Bx", "Bz": "Bz"}
    Ly = {perm[k]: v for k, v in L.items()}
    Ry = {perm[k]: v for k, v in R.items()}
    Fy = linearised_flux(Ly, Ry, eos, 1)[0]
    Fy = np.stack([Fy[k].numpy() for k in
                   ("D", "Sy", "Sx", "Sz", "tau", "By", "Bx", "Bz")])
    assert np.allclose(Fx, Fy, rtol=1e-12, atol=1e-14)


def test_second_order_where_hlld_is_first(eos, pairs):
    """Shrink the discontinuity by 1/3 per level: the linearised flux's
    distance from the exact flux must fall by ~9 per level (HLLD's by ~3),
    and at the weakest level it must be the closer of the two on most
    faces."""
    import src.physics.exact_flux as EF
    L, R = pairs
    levels = (0.1, 0.1 / 3, 0.1 / 9)
    e_lin, e_hlld, ok_all = [], [], []
    for s in levels:
        Ls, Rs = _shrink(eos, L, R, s)
        Fe, _, _ = EF.exact_flux_batched(Ls, Rs, eos, idir=0, tau_weak=0.0,
                                         n_retries=1, max_iter=40)
        ex = np.asarray(EF.LAST_DIAG["exact_mask"], bool)
        Fl, _, _, ok = linearised_flux(Ls, Rs, eos, 0, return_mask=True)
        Fh, _, _ = hlld_flux(Ls, Rs, eos, 0)
        Fe, Fl, Fh = _vec(Fe), _vec(Fl), _vec(Fh)
        good = ex & np.asarray(ok, bool)
        ok_all.append(good)
        e_lin.append(np.where(good, _rel(Fl, Fe), np.nan))
        e_hlld.append(np.where(good, _rel(Fh, Fe), np.nan))
    good = np.logical_and.reduce(ok_all)
    assert good.sum() >= 6, "too few faces solved exactly at every level"
    e_lin = np.stack(e_lin)[:, good]
    e_hlld = np.stack(e_hlld)[:, good]
    r_lin = np.median(e_lin[0] / e_lin[2])
    r_hlld = np.median(e_hlld[0] / e_hlld[2])
    # two levels of 1/3: second order gives 81, first order 9
    assert r_lin > 25, f"linearised error fell by only {r_lin:.1f} over 1/9"
    assert r_lin > 2.0 * r_hlld
    assert (e_lin[2] < e_hlld[2]).mean() >= 0.75


def test_on_by_default_and_only_the_weak_faces_change(eos, pairs):
    import src.physics.exact_flux as EF
    old = os.environ.get("RMHD_WEAK_FLUX")
    os.environ.pop("RMHD_WEAK_FLUX", None)
    EF = importlib.reload(EF)
    assert EF._WEAK_FLUX == "linear"
    L, R = pairs
    Ls, Rs = _shrink(eos, L, R, 0.01)
    try:
        os.environ["RMHD_WEAK_FLUX"] = "hlld"
        EF = importlib.reload(EF)
        Fh, _, ph = EF.exact_flux_batched(Ls, Rs, eos, idir=0, tau_weak=1e-2)
        weak = np.asarray(EF.LAST_DIAG["n_weak"])
        os.environ.pop("RMHD_WEAK_FLUX", None)
        EF = importlib.reload(EF)
        Fl, _, pl = EF.exact_flux_batched(Ls, Rs, eos, idir=0, tau_weak=1e-2)
        d = dict(EF.LAST_DIAG)
    finally:
        if old is None:
            os.environ.pop("RMHD_WEAK_FLUX", None)
        else:
            os.environ["RMHD_WEAK_FLUX"] = old
        EF = importlib.reload(EF)
    assert d["n_weak"] == weak and d["n_weak_linear"] > 0
    assert d["n_weak_linear"] <= d["n_weak"]
    # every face the linearised solver answered moved, no other face did,
    # and none of them is called exact
    moved = np.any(np.stack([(Fl[k] != Fh[k]).numpy() for k in KEYS]), axis=0)
    lin = np.asarray(d["weak_linear_mask"], bool)
    assert lin.sum() == d["n_weak_linear"]
    # a face it answered may still carry HLLD's flux to the bit: a supersonic
    # face, where both are the physical flux of the upwind state
    assert not (moved & ~lin).any()
    assert moved.sum() >= 0.5 * lin.sum()
    attempted = np.zeros(moved.size, bool)
    attempted[np.asarray(d["sel"], int)] = True
    assert not (lin & attempted).any()
    assert d["n_exact"] == np.asarray(d["exact_mask"], bool).sum() <= attempted.sum()


def _fallback_run(eos, Ls, Rs, tau_fb, tmp_path, **kw):
    """exact_flux_batched with the linearised flux on and the weak-fallback
    threshold ``tau_fb``, through a harvester; returns (F, diag, harvester)."""
    import src.physics.exact_flux as EF
    from src.physics.harvest import Harvester
    old = {k: os.environ.get(k) for k in ("RMHD_WEAK_FLUX", "RMHD_WEAK_FALLBACK_TAU")}
    os.environ.pop("RMHD_WEAK_FLUX", None)
    os.environ["RMHD_WEAK_FALLBACK_TAU"] = str(tau_fb)
    try:
        EF = importlib.reload(EF)
        H = Harvester(str(tmp_path), GAMMA)
        F, _, _ = EF.exact_flux_batched(Ls, Rs, eos, idir=0, harvester=H, **kw)
        d = dict(EF.LAST_DIAG)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        importlib.reload(EF)
    return F, d, H


def test_a_lost_weak_face_takes_the_linearised_flux(eos, pairs, tmp_path):
    """With the gate open, a weak face every exact rung loses gets the
    linearised flux (not HLLD), is never called exact, and is recorded."""
    import src.physics.exact_flux as EF
    L, R = pairs
    Ls, Rs = _shrink(eos, L, R, 0.01)
    # shrunk to 1%, most faces are weak; the relative jump is per variable,
    # so a few stay at 1e-2 or more and must keep HLLD
    jump = EF.relative_jump(Ls, Rs).numpy().reshape(-1)
    weak = jump < 1e-2
    # one Newton iteration: most lanes are lost, so the fallback is exercised
    F, d, H = _fallback_run(eos, Ls, Rs, 1e-2, tmp_path, tau_weak=0.0,
                            max_iter=1, n_retries=0)
    n = jump.size
    # the gate is open, but a face whose fan lies on one side is still not
    # attempted (HLLD is exact there)
    att = np.zeros(n, bool)
    sel = np.asarray(d["sel"], int)
    att[sel] = True
    assert (att & weak).sum() >= 6
    ex = np.asarray(d["exact_mask"], bool)
    lin = np.zeros(n, bool)
    lin[sel[np.asarray(d["linear_fallback_mask"], bool)]] = True
    assert d["n_linear_fallback"] == lin.sum() > 0
    assert d["n_linear_fallback_offered"] == (att & ~ex & weak).sum()
    assert not (lin & ex).any()
    assert not (lin & ~(att & weak)).any()
    assert d["n_hlld_after_attempt"] == (att & ~ex & ~lin).sum()
    Fl, _, _, ok = linearised_flux(Ls, Rs, eos, 0, return_mask=True)
    Fh, _, _ = hlld_flux(Ls, Rs, eos, 0)
    Fv, Flv, Fhv = _vec(F), _vec(Fl), _vec(Fh)
    assert np.allclose(Fv[:, lin], Flv[:, lin], rtol=1e-12, atol=1e-15)
    rest = ~ex & ~lin                      # lost and strong, or not attempted
    assert np.array_equal(Fv[:, rest], Fhv[:, rest])
    # the record: the coverage map, the per-decade histogram, the totals
    H.flush()
    z = np.load(sorted(tmp_path.glob("coverage_*.npz"))[0])
    cov = np.unpackbits(z["linear_fallback"][0])[:n].astype(bool)
    assert np.array_equal(cov, lin)
    h = z["jump_hist"][0]
    rows = H.JUMP_HIST_ROWS
    assert h[rows.index("all")].sum() == n
    assert h[rows.index("attempted")].sum() == att.sum()
    assert h[rows.index("linear_fallback")].sum() == lin.sum()
    assert h[rows.index("exact")].sum() == ex.sum()
    t = H.summary()["faces"]
    assert t["faces"] == n and t["attempted"] == att.sum()
    assert t["linear_fallback"] == lin.sum()
    assert t["hlld_after_attempt"] == (att & ~ex & ~lin).sum()


def test_the_fallback_is_off_at_zero_and_never_touches_strong_faces(eos, pairs, tmp_path):
    import src.physics.exact_flux as EF
    L, R = pairs
    Ls, Rs = _shrink(eos, L, R, 0.01)
    F0, d0, _ = _fallback_run(eos, Ls, Rs, 0.0, tmp_path / "a", tau_weak=0.0,
                              max_iter=1, n_retries=0)
    assert d0["n_linear_fallback"] == 0
    ex = np.asarray(d0["exact_mask"], bool)
    Fh, _, _ = hlld_flux(Ls, Rs, eos, 0)
    assert np.array_equal(_vec(F0)[:, ~ex], _vec(Fh)[:, ~ex])
    # full-strength problems: a lost face with a jump of 1e-2 or more keeps
    # HLLD, and with the production gate nothing is ever offered
    jump = EF.relative_jump(L, R).numpy().reshape(-1)
    _, d1, _ = _fallback_run(eos, L, R, 1e-2, tmp_path / "b", tau_weak=0.0,
                             max_iter=1, n_retries=0)
    lin = np.zeros(jump.size, bool)
    lin[np.asarray(d1["sel"], int)[np.asarray(d1["linear_fallback_mask"], bool)]] = True
    assert (jump[lin] < 1e-2).all()
    _, d2, _ = _fallback_run(eos, L, R, 1e-2, tmp_path / "c", tau_weak=1e-2,
                             max_iter=1, n_retries=0)
    assert d2["n_linear_fallback_offered"] == 0
