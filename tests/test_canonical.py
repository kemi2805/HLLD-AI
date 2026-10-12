"""Every face solved in one canonical orientation (RMHD_CANONICAL,
src/physics/canonical.py).

Two faces related by a symmetry of the Riemann problem -- the reflection
(left and right exchanged, normal components negated), a signed exchange of
the tangential axes, the sign of B -- must get related answers.  Without the
canonical orientation the exact solver gives one face an exact flux and its
mirror image HLLD's on 0.4-0.7% of Orszag-Tang's pairs (calea job 40966).
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.physics import canonical as CAN                     # noqa: E402
from src.physics.eos import hybrid_eos                       # noqa: E402
from src.physics.hlld import LAST_DIAG, hlld_flux            # noqa: E402

GAMMA = 5.0 / 3.0
EOS = hybrid_eos(K=0.0, gamma=GAMMA, gamma_th=GAMMA)
MAX_ITER = 4
# the OT pi rotation on an x-face (reflection + t1 negated), the
# four-quadrant diagonal (t1 <-> t2), and B -> -B
NAMED = {"pi rotation": (1, 0, -1, 1, 1), "diagonal": (0, 1, 1, 1, 1),
         "field sign": (0, 0, 1, 1, -1)}


def _random(seed, n, idir):
    rng = np.random.default_rng(seed)
    t = lambda a: torch.tensor(a, dtype=torch.float64)
    Bn = rng.normal(size=n)
    out = []
    for _ in range(2):
        v = rng.normal(size=(3, n))
        v *= rng.uniform(0.0, 0.9, n) / np.linalg.norm(v, axis=0)
        B = rng.normal(size=(3, n))
        planar = rng.random(n) < 0.3             # a zero component: ties
        v[2][planar] = 0.0
        B[2][planar] = 0.0
        B[0] = Bn
        n_, t1, t2 = idir, (idir + 1) % 3, (idir + 2) % 3
        d = {"rho": t(rng.uniform(0.1, 2.0, n)), "p": t(rng.uniform(0.05, 2.0, n))}
        for vec, comps in (("v", v), ("B", B)):
            d[vec + "xyz"[n_]], d[vec + "xyz"[t1]], d[vec + "xyz"[t2]] = (
                t(comps[0]), t(comps[1]), t(comps[2]))
        d["eps"] = EOS.eps__press_rho(d["p"], d["rho"])
        out.append(d)
    return out


def _hlld(cL, cR, eos, idir=0, _record_idir=None, **kw):
    return hlld_flux(cL, cR, eos, idir=idir)


@pytest.mark.parametrize("idir", [0, 1, 2])
def test_every_image_reduces_to_the_same_problem(idir):
    sL, sR = _random(1 + idir, 400, idir)
    cL, cR, _ = CAN.to_canonical(sL, sR, idir)
    for h in range(len(CAN.GROUP)):
        hL, hR = CAN.transform(sL, sR, idir, np.full(400, h))
        dL, dR, _ = CAN.to_canonical(hL, hR, 0)
        for k in cL:
            assert torch.equal(cL[k], dL[k]) and torch.equal(cR[k], dR[k]), (h, k)


@pytest.mark.parametrize("idir", [0, 1, 2])
def test_the_answer_maps_back(idir):
    """HLLD is mirror-symmetric to its rounding, so on the canonical problems,
    mapped back, it must give its own answer -- a sign wrong anywhere in the
    map would show at O(1).  Lanes where either orientation fell back to HLLE
    (p_star < 0) are left out: HLLD's failure test is not symmetric."""
    sL, sR = _random(11 + idir, 600, idir)
    F0, U0, p0 = hlld_flux(sL, sR, EOS, idir=idir)
    F1, U1, p1 = CAN.canonical_flux(_hlld, sL, sR, EOS, idir)
    ok = ((p0 >= 0) & (p1 >= 0)).numpy()
    assert ok.mean() > 0.7
    rel = lambda A, B: (sum((A[k] - B[k]).abs() for k in B)
                        / sum(B[k].abs() for k in B)).numpy()
    assert rel(F1, F0)[ok].max() < 1e-8
    assert rel(U1, U0)[ok].max() < 1e-8
    assert np.abs((p1 - p0).numpy()[ok]).max() < 1e-8
    assert (np.asarray(LAST_DIAG["canonical_g"]) != 0).mean() > 0.5


# ── the exact flux ────────────────────────────────────────────────────────
def _states(seed, n, coplanar_frac):
    from rmhd.eos import set_eos
    set_eos("ideal")
    from rmhd.riemann_dataset import generate_dataset
    sols = generate_dataset(n, gamma=GAMMA, Bx_range=(0.05, 2.0), seed=seed,
                            xi_window=(0.0, 0.0), coplanar_frac=coplanar_frac,
                            verbose=False)
    return np.array([s.zones for s in sols]), np.array([s.Bx for s in sols])


@pytest.fixture(scope="module")
def interfaces():
    Z1, B1 = _states(91, 16, 0.0)
    Z2, B2 = _states(92, 16, 1.0)
    Z, Bx = np.concatenate([Z1, Z2]), np.concatenate([B1, B2])

    def prim(slot):
        rho, Pt, vn, v1, v2_, Bt1, Bt2 = (Z[:, slot, j] for j in range(7))
        W2 = 1.0 / (1.0 - (vn ** 2 + v1 ** 2 + v2_ ** 2))
        eta = Bx * vn + Bt1 * v1 + Bt2 * v2_
        b2 = (Bx ** 2 + Bt1 ** 2 + Bt2 ** 2) / W2 + eta ** 2
        t = lambda a: torch.tensor(a, dtype=torch.float64)
        d = {"rho": t(rho), "vx": t(vn), "vy": t(v1), "vz": t(v2_),
             "p": t(Pt - 0.5 * b2), "Bx": t(Bx), "By": t(Bt1), "Bz": t(Bt2)}
        d["eps"] = EOS.eps__press_rho(d["p"], d["rho"])
        return d

    # the generator's own faces, and the same faces mirrored (so the batch
    # holds faces the canonical choice moves AND faces it keeps)
    sL, sR = prim(0), prim(7)
    mL, mR = CAN.transform(sL, sR, 0, np.full(32, 16))      # the reflection
    cat = lambda a, b: {k: torch.cat([a[k], b[k]]) for k in a}
    return cat(sL, mL), cat(sR, mR)


def _exact(monkeypatch, sL, sR, canonical):
    for k in ("RMHD_PLANAR5_FALLBACK", "RMHD_PLANAR5_LADDER",
              "RMHD_PLANAR5_CROSSED", "RMHD_PLANAR4"):
        monkeypatch.setenv(k, "1")
    monkeypatch.setenv("RMHD_POOL", "0")
    monkeypatch.setenv("RMHD_CANONICAL", "1" if canonical else "0")
    import src.physics.exact_flux as EF
    EF = importlib.reload(EF)
    F, U, p = EF.exact_flux_batched(sL, sR, EOS, idir=0, max_iter=MAX_ITER)
    return F, U, p, dict(LAST_DIAG)


@pytest.mark.parametrize("name", list(NAMED) + ["every element"])
def test_mirror_faces_get_mirror_answers(monkeypatch, interfaces, name):
    import src.physics.exact_flux as EF
    sL, sR = interfaces
    n = sL["rho"].numel()
    h = (np.arange(n) % len(CAN.GROUP) if name == "every element"
         else np.full(n, CAN.GROUP.index(NAMED[name])))
    hL, hR = CAN.transform(sL, sR, 0, h)
    try:
        F, U, p, d = _exact(monkeypatch, sL, sR, True)
        Fh, Uh, ph, dh = _exact(monkeypatch, hL, hR, True)
    finally:
        monkeypatch.undo()
        importlib.reload(EF)
    assert d["n_attempted"] > 0 and d["n_exact"] > 0
    assert np.array_equal(d["exact_mask"].numpy(), dh["exact_mask"].numpy())
    Fb, Ub = CAN._back(Fh, h, 0, flux=True), CAN._back(Uh, h, 0, flux=False)
    for k in F:
        assert torch.equal(Fb[k], F[k]), k
        assert torch.equal(Ub[k], U[k]), k
    assert torch.equal(ph, p)


def test_it_agrees_with_the_plain_solver(monkeypatch, interfaces):
    """The canonical answer is the plain solver's on another member of the
    orbit: where both are exact they agree to the solve's tolerance (a lane
    may land on another verified root, as the continuation's do).  Which
    lanes each solves differs with the frame -- on this fixture the plain
    solver solves 29 of 32 faces as generated and 23 of their mirror images
    -- so the solve rate is measured on whole runs, not here."""
    import src.physics.exact_flux as EF
    sL, sR = interfaces
    try:
        F0, _, _, d0 = _exact(monkeypatch, sL, sR, False)
        F1, _, _, d1 = _exact(monkeypatch, sL, sR, True)
    finally:
        monkeypatch.undo()
        importlib.reload(EF)
    both = np.flatnonzero(d0["exact_mask"].numpy() & d1["exact_mask"].numpy())
    assert both.size >= 4
    rel = (sum((F1[k] - F0[k]).abs() for k in F0)
           / sum(F0[k].abs() for k in F0)).numpy()[both]
    assert (rel < 1e-6).mean() >= 0.9


def test_rounding_noise_does_not_change_the_orientation():
    """A mirror image off by rounding noise must be solved in the same
    orientation, also where two values of the key nearly tie (rho_L ~ rho_R,
    v_t ~ 0: the four-quadrant problem's uniform quadrants, where the exact
    lexicographic choice put 1,364 of 4,032 mirror pairs in different
    orientations at t = 0.05)."""
    sL, sR = _random(21, 600, 0)
    rng = np.random.default_rng(22)
    tie = rng.random(600) < 0.5
    for k in ("rho", "p", "vx", "vy", "vz", "By", "Bz"):
        sR[k] = torch.where(torch.as_tensor(tie), sL[k], sR[k])   # L = R ...
    for k in ("vy", "By"):                                         # ... v_t, B_t ~ 0
        sL[k] = torch.where(torch.as_tensor(tie), 1e-14 * sL[k], sL[k])
        sR[k] = torch.where(torch.as_tensor(tie), 1e-14 * sR[k], sR[k])
    h = rng.integers(0, len(CAN.GROUP), 600)
    hL, hR = CAN.transform(sL, sR, 0, h)
    noisy = lambda d: {k: v * (1.0 + 1e-13 * torch.as_tensor(rng.standard_normal(600)))
                       if k not in ("Bx",) else v for k, v in d.items()}
    hL, hR = noisy(hL), noisy(hR)
    a = CAN.to_canonical(sL, sR, 0)
    b = CAN.to_canonical(hL, hR, 0)
    for side in (0, 1):
        for k in ("rho", "p", "vx", "vy", "vz", "Bx", "By", "Bz"):
            d = (a[side][k] - b[side][k]).abs() / (a[side][k].abs() + 1.0)
            assert float(d.max()) < 1e-11, (side, k)


def test_the_normal_field_is_never_negative():
    sL, sR = _random(31, 600, 0)
    cL, cR, _ = CAN.to_canonical(sL, sR, 0)
    assert bool((cL["Bx"] >= 0).all()) and bool((cR["Bx"] >= 0).all())
