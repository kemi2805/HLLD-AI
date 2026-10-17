"""
The 2026-10 p* data pipeline: hlld_flux's row_diag hook, the v3 features,
the union-envelope sampler, the labeller, and the shared normalisation.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.labelling import label, label_batched, make_eos, to_prims
from src.data.sampling import COMPONENTS, component_counts, merged_cfg, sample_pairs
from src.physics import ai_features as ai
from src.physics import hlld as H
from src.physics.hlld import compute_srmhd_fluxes, energy_form, hlld_flux

GAMMA = 5.0 / 3.0


def _pairs(n=3000, seed=11, gamma=GAMMA):
    S = sample_pairs(n, None, np.random.default_rng(seed), gamma)
    eos = make_eos(gamma)
    return S, eos, to_prims(S["L"], eos), to_prims(S["R"], eos)


def _bits_equal(a, b):
    return torch.equal(a.view(torch.int64), b.view(torch.int64))


@pytest.mark.parametrize("idir", [0, 1])
def test_row_diag_leaves_hlld_flux_bitwise_unchanged(idir):
    """The hook only reads: flux, state and p* are identical to the bit."""
    _, eos, sL, sR = _pairs(2000)
    sL["Bx"][:50] = 0.0
    sR["Bx"][:50] = 0.0                       # B_n = 0 rows: the fallback path
    if idir:
        rot = lambda s: {**s, "vx": s["vz"], "vy": s["vx"], "vz": s["vy"],
                         "Bx": s["Bz"], "By": s["Bx"], "Bz": s["By"]}
        sL, sR = rot(sL), rot(sR)
    f0, u0, p0 = hlld_flux(sL, sR, eos, idir=idir)
    rd = {}
    f1, u1, p1 = hlld_flux(sL, sR, eos, idir=idir, row_diag=rd)
    assert _bits_equal(p0, p1)
    for k in f0:
        assert _bits_equal(f0[k], f1[k]) and _bits_equal(u0[k], u1[k]), k
    # the diagnostics agree with what hlld_flux did
    failed = rd["failed"]
    assert torch.equal(failed, p1 < 0)
    assert torch.equal(rd["bn_zero"][:50], torch.ones(50, dtype=torch.bool))
    ok = ~failed
    assert _bits_equal(rd["p_root"][ok], p1[ok])
    assert bool((rd["resid"][ok & (rd["err"] == 0)].abs() < H._PSTAR_TOL).all())


def test_v3_features_are_rotation_and_field_flip_invariant():
    _, eos, sL, sR = _pairs(2000, seed=3)

    def feats(a, b, idir):
        uL, uR, fL, fR, cmax, cmin = compute_srmhd_fluxes(a, b, eos, idir)
        UL, FL = energy_form(uL, fL)
        UR, FR = energy_form(uR, fR)
        return ai.pstar_features(ai.FEATURE_VERSION_V3, a, b, uL, uR, UL, FL,
                                 UR, FR, cmin, cmax, idir)

    X0 = feats(sL, sR, 0)
    assert X0.shape == (2000, ai.n_features(ai.FEATURE_VERSION_V3)) == (2000, 17)
    # a rotation about the normal by an arbitrary angle
    th = 0.7
    c, s = np.cos(th), np.sin(th)
    rot = lambda d: {**d, "vy": c * d["vy"] - s * d["vz"], "vz": s * d["vy"] + c * d["vz"],
                     "By": c * d["By"] - s * d["Bz"], "Bz": s * d["By"] + c * d["Bz"]}
    X1 = feats(rot(sL), rot(sR), 0)
    flip = lambda d: {**d, **{k: -d[k] for k in ("Bx", "By", "Bz")}}
    X2 = feats(flip(sL), flip(sR), 0)
    # a reflection of the transverse plane (v and B alike, see ai_features)
    refl = lambda d: {**d, "vz": -d["vz"], "Bz": -d["Bz"]}
    X3 = feats(refl(sL), refl(sR), 0)
    scale = X0.abs().max(dim=0).values.clamp(min=1.0)
    for X in (X1, X2, X3):
        assert float(((X - X0).abs() / scale).max()) < 1e-12


def test_v3_tells_a_field_reversal_from_a_uniform_field_and_v2_does_not():
    """The coplanar case that motivates v3 (every 2D face is coplanar)."""
    eos = make_eos(GAMMA)
    base = dict(rho=1.0, p=1.0, vx=0.0, vy=0.0, vz=0.0, Bx=0.5, By=1.0, Bz=0.0)
    one = lambda **kw: {k: torch.tensor([v], dtype=torch.float64)
                        for k, v in {**base, **kw}.items()}
    L = to_prims(one(), eos)
    # ST1's jump with |B_y,R| = 0.5 (at |B_y,R| = 1 and gamma 5/3 HLLD's root
    # for the reversal fails the wave order and the face falls back to HLLE)
    R_rev = to_prims(one(rho=0.125, p=0.1, By=-0.5), eos)
    R_same = to_prims(one(rho=0.125, p=0.1, By=0.5), eos)
    lab_rev = label(L, R_rev, eos)
    lab_same = label(L, R_same, eos)
    v2, v3 = "X_" + ai.FEATURE_VERSION, "X_" + ai.FEATURE_VERSION_V3
    assert np.array_equal(lab_rev[v2], lab_same[v2])
    assert not np.allclose(lab_rev[v3], lab_same[v3])
    assert lab_rev["valid"][0] and lab_same["valid"][0]
    # 0.4545 against 0.6073: v2 maps both to one p*, one of them >= 14% off
    assert abs(lab_rev["p_star"][0] / lab_same["p_star"][0] - 1) > 0.2


def test_sampler_mixture_caps_and_coplanarity():
    cfg = merged_cfg(None)
    n = 4000
    S = sample_pairs(n, cfg, np.random.default_rng(5), GAMMA)
    counts = np.bincount(S["comp"], minlength=len(COMPONENTS))
    assert np.array_equal(counts, component_counts(n, cfg["components"]))
    for t in "LR":
        s = S[t]
        lT = np.log10(s["p"] / s["rho"])
        assert (lT >= cfg["caps"]["log_T"][0]).all() and (lT <= cfg["caps"]["log_T"][1]).all()
        v2 = s["vx"] ** 2 + s["vy"] ** 2 + s["vz"] ** 2
        assert (v2 < 1).all()
        B2 = s["Bx"] ** 2 + s["By"] ** 2 + s["Bz"] ** 2
        vB = s["vx"] * s["Bx"] + s["vy"] * s["By"] + s["vz"] * s["Bz"]
        b2 = B2 * (1 - v2) + vB ** 2
        sigma = b2 / (s["rho"] + GAMMA / (GAMMA - 1) * s["p"])
        assert (sigma <= cfg["caps"]["sigma_max"] * (1 + 1e-12)).all()
    assert np.array_equal(S["L"]["Bx"], S["R"]["Bx"])
    planar = np.isin(S["comp"], [COMPONENTS.index("indep_planar"),
                                 COMPONENTS.index("jump_planar")])
    # all four transverse vectors of a planar row are parallel
    vecs = [(S[t][a], S[t][b]) for t in "LR" for a, b in (("vy", "vz"), ("By", "Bz"))]
    for (a1, a2) in vecs:
        for (b1, b2) in vecs:
            cross = (a1 * b2 - a2 * b1)[planar]
            size = (np.hypot(a1, a2) * np.hypot(b1, b2))[planar] + 1e-300
            assert (np.abs(cross) <= 1e-12 * size + 1e-300).all()


def test_labeller_matches_hlld_flux_and_batches_do_not_matter():
    _, eos, sL, sR = _pairs(3000, seed=8)
    _, _, p = hlld_flux(sL, sR, eos)
    lab = label(sL, sR, eos)
    assert np.array_equal(lab["p_star"], p.numpy())
    assert np.array_equal(lab["valid"], (p > 0).numpy() & lab["valid"])
    assert lab["valid"].sum() == int((p > 0).sum())
    lab_b = label_batched({k: v.numpy() for k, v in sL.items()},
                          {k: v.numpy() for k, v in sR.items()}, eos, 0, batch=700)
    for k in lab:
        assert np.array_equal(lab[k], lab_b[k]), k


@pytest.mark.parametrize("ref", ["none", "ptot_avg"])
@pytest.mark.parametrize("transform", ["none", "asinh"])
def test_hlld_ai_flux_sees_exactly_the_normalised_training_features(transform, ref):
    """Inference normalises with the same function as the split builder, and
    picks the feature set, transform and pressure reference from the stats."""
    _, eos, sL, sR = _pairs(500, seed=2)
    v = ai.FEATURE_VERSION_V3
    lab = label(sL, sR, eos, versions=(v,))
    X = torch.from_numpy(lab["X_" + v])
    stats = dict(feature_version=np.array(v), x_transform=np.array(transform),
                 ref=np.array(ref), y_mu=np.array(0.3), y_sigma=np.array(0.7))
    p_ref = ai.p_ref_from_prims(sL, sR, eos)
    Xs = ai.scale_by_ref(X, v, p_ref) if ref == "ptot_avg" else X
    if transform == "asinh":
        stats["x_scale"] = np.median(np.abs(Xs.numpy()), axis=0) + 1e-3
        Xt = torch.asinh(Xs / torch.as_tensor(stats["x_scale"]))
    else:
        Xt = Xs
    stats["mu"], stats["sigma"] = Xt.mean(0).numpy(), Xt.std(0).numpy() + 1e-12
    Xn = ai.normalise(X, stats, p_ref)
    assert torch.equal(Xn, ((Xt - torch.as_tensor(stats["mu"]))
                            / torch.as_tensor(stats["sigma"])).float())
    seen = {}

    class _Net(torch.nn.Module):
        def forward(self, x):
            seen["x"] = x
            return torch.zeros(x.shape[0], 1)

    rd = {}
    H.hlld_ai_flux(sL, sR, eos, _Net(), stats, row_diag=rd)
    assert torch.equal(seen["x"], Xn)
    assert rd["feature_version"] == v
    # y = 0 -> p = 10^y_mu (times p_ref with the reference)
    want = torch.full((500,), 10 ** 0.3, dtype=torch.float64)
    if ref == "ptot_avg":
        want = want * p_ref
    assert torch.allclose(rd["p_nn"], want, rtol=1e-12, atol=0)


def test_scale_reference_makes_features_scale_free():
    """rho, p, B^2 -> lambda * leaves the scaled features and p*/p_ref alone
    (the exact symmetry ai_features.X_REF_POWER relies on)."""
    _, eos, sL, sR = _pairs(1500, seed=4)
    lam = 37.0
    sc = lambda s: {**s, "rho": lam * s["rho"], "p": lam * s["p"],
                    "eps": s["eps"], **{k: lam ** 0.5 * s[k] for k in ("Bx", "By", "Bz")}}
    for v in (ai.FEATURE_VERSION, ai.FEATURE_VERSION_V3):
        a = label(sL, sR, eos, versions=(v,))
        b = label(sc(sL), sc(sR), eos, versions=(v,))
        pa = ai.p_ref_from_prims(sL, sR, eos)
        pb = ai.p_ref_from_prims(sc(sL), sc(sR), eos)
        assert torch.allclose(pb, lam * pa, rtol=1e-13, atol=0)
        Xa = ai.scale_by_ref(torch.from_numpy(a["X_" + v]), v, pa)
        Xb = ai.scale_by_ref(torch.from_numpy(b["X_" + v]), v, pb)
        assert float(((Xa - Xb).abs() / Xa.abs().clamp(min=1e-3)).max()) < 1e-10
        ok = a["valid"] & b["valid"]
        ra = a["p_star"][ok] / pa.numpy()[ok]
        rb = b["p_star"][ok] / pb.numpy()[ok]
        assert np.median(np.abs(ra / rb - 1)) < 1e-9
