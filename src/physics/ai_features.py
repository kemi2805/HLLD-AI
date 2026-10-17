"""
Feature vector for the ML surrogate that predicts the HLLD star pressure p*.

This module is the SINGLE definition of that feature vector.  It is imported
by both the inference path (``hlld.hlld_ai_flux``) and the training-data
generator (``src/data/generate.py``).  Those two previously each wrote the
vector out by hand, which is exactly why it stayed pinned to the x
direction: fixing one copy could not fix the other, and nothing detected a
mismatch between them.

Direction genericity
--------------------
The features are decomposed into normal / transverse components relative to
the sweep direction ``idir``:

    per side:  R_tau,  R_Sn,  |R_St|,  |R_Bt|,  v_n,  |v_t|
    shared:    |B_n|,  cmin,  cmax

Because the transverse quantities enter only as MAGNITUDES, the set is
already invariant under rotations about the normal.  Making the component
names follow ``idir`` is therefore sufficient to make the network usable for
y-sweeps with no retraining — the physics it learned is the same physics.

Why ``|B_n|`` rather than ``B_n``
---------------------------------
SRMHD is invariant under ``B -> -B``.  Under that map ``R_B`` flips sign, so
``|R_Bt|`` is unchanged; ``R_S`` and ``R_tau`` involve B only bilinearly, so
they are unchanged; the velocities are unchanged; and ``p*`` is unchanged.
Every feature except a raw signed ``B_n`` is therefore already ``B -> -B``
invariant, and so is the target.  Feeding ``|B_n|`` makes the feature set
respect that symmetry too.

This also matters practically: the training data was generated with
``Bx > 0`` only, but the normal field changes sign in y-sweeps and in the
magnetic rotor.  With ``abs()`` the existing checkpoint remains valid for
negative-normal-field interfaces instead of being extrapolated blindly.

Known limitation
----------------
The transverse velocity and field enter only through their magnitudes, not
their relative angle.  Two states with identical features but different
transverse v-B alignment can have different ``p*``, so the map
features -> target is not a function and there is an irreducible error
floor.  ``src/physics/envelope.py`` records the relative angle from real
runs so this can be quantified rather than assumed.

The v3 set at the end of this module (``FEATURE_VERSION_V3``) closes that
gap: it is the complete set of invariants HLLD's residual depends on.  v2
stays the default, which is what the archived checkpoint was trained on.
Which version a checkpoint expects is stamped into its ``norm_stats.npz``
(``feature_version``) and read back by ``hlld.hlld_ai_flux``.
"""

from __future__ import annotations

import torch

# Bump whenever the feature definition changes in a way that invalidates a
# trained checkpoint.  Written into norm_stats.npz at training time and
# checked at load time, so a stale checkpoint fails loudly instead of
# silently producing garbage predictions.
FEATURE_VERSION = "v2-idir-abs-bn"

FEATURE_NAMES = (
    "R_tau_L", "R_Sn_L", "R_St_L", "R_Bt_L", "v_n_L", "v_t_L",
    "R_tau_R", "R_Sn_R", "R_St_R", "R_Bt_R", "v_n_R", "v_t_R",
    "absBn", "cmin", "cmax",
)
N_FEATURES = len(FEATURE_NAMES)

_S = ("Sx", "Sy", "Sz")
_B = ("Bx", "By", "Bz")
_V = ("vx", "vy", "vz")


def r_vectors(U_init_L, F_init_L, U_init_R, F_init_R, cmin, cmax):
    """The HLL jump vectors the features are built from.

    ``R_L = cmin*U_L - F_L`` and ``R_R = cmax*U_R - F_R``.  Note the sign
    convention differs from ``HLLDComputation``'s internal R by the sign of
    the cmin term; it is kept as-is because generator and inference agree on
    it, so the trained checkpoint is consistent with it.
    """
    RL = {k: cmin * U_init_L[k] - F_init_L[k] for k in U_init_L}
    RR = {k: cmax * U_init_R[k] - F_init_R[k] for k in U_init_R}
    return RL, RR


def build_pstar_features(sL, sR, uL, uR, RL, RR, cmin, cmax, idir: int = 0):
    """Assemble the (N, 15) feature matrix for the p* network.

    Parameters
    ----------
    sL, sR : primitive state dicts (need ``vx``/``vy``/``vz``).
    uL, uR : conserved state dicts (used only for the normal field).
    RL, RR : the jump vectors from :func:`r_vectors`.
    idir   : sweep direction; 0/1/2 select the normal component.
    """
    n = idir
    t1, t2 = (idir + 1) % 3, (idir + 2) % 3

    def side(s, R):
        v_n = s[_V[n]]
        v_t = torch.sqrt(s[_V[t1]] ** 2 + s[_V[t2]] ** 2)
        R_St = torch.sqrt(R[_S[t1]] ** 2 + R[_S[t2]] ** 2)
        R_Bt = torch.sqrt(R[_B[t1]] ** 2 + R[_B[t2]] ** 2)
        return R["tau"], R[_S[n]], R_St, R_Bt, v_n, v_t

    L = side(sL, RL)
    R = side(sR, RR)
    absBn = uL[_B[n]].abs()

    return torch.stack([*L, *R, absBn, cmin, cmax], dim=1)


# ── v3: the complete invariants of HLLD's own jump vectors ───────────────────
#
# Added with the 2026-10 retrain (branch pstar-ml).  v2 above stays the
# default, so the archived checkpoint and its norm stats keep working.
#
# HLLD's p* is the root of HLLDComputation.residual, and that residual reads
# the left/right states ONLY through HLLD's jump vectors
#
#     R_L = -cmin*U_L - F_L,   R_R = cmax*U_R - F_R     (energy form)
#
# (normal components R_tau, R_Sn; transverse 2-vectors a = R_St, b = R_Bt),
# the normal field B_n and the two speeds.  Every transverse quantity in the
# residual is a linear combination of a_L, b_L, a_R, b_R with scalar
# coefficients, so the residual -- and its root -- depends on them only
# through their Gram matrix: four lengths and six angles.  It is also
# unchanged under B -> -B, which flips b and B_n but not a.
#
# v2 keeps the four lengths and drops the six angles, and it builds its left
# vector with +cmin (r_vectors' docstring), which is not the residual's.  So
# v2 cannot tell a field reversal from a uniform field: B_t,L = +1 against
# B_t,R = -1 or +1 give identical v2 vectors -- in 2D every face is
# coplanar, so it is exactly the sign information v2 loses.  v3 is that
# Gram matrix in scale-free form, which makes the map features -> p* a
# FUNCTION (up to which root the bracket picks): R_tau, R_Sn and the
# lengths per side, |B_n|, cmin, cmax, and the six cosines, the three that
# pair an a with a b multiplied by sign(B_n) to make them B -> -B invariant.
# Nothing in it depends on the EOS: the residual does not, only the
# states' R vectors and speeds do.  (Which root the bracket selects still
# can, through the c2p of the HLL state.)
FEATURE_VERSION_V3 = "v3-hlld-r-invariants"

FEATURE_NAMES_V3 = (
    "R_tau_L", "R_Sn_L", "a_L", "b_L",
    "R_tau_R", "R_Sn_R", "a_R", "b_R",
    "absBn", "cmin", "cmax",
    "cos_aL_bL", "cos_aR_bR", "cos_aL_aR", "cos_bL_bR", "cos_aL_bR", "cos_bL_aR",
)

FEATURE_SETS = {FEATURE_VERSION: FEATURE_NAMES,
                FEATURE_VERSION_V3: FEATURE_NAMES_V3}


def n_features(version: str = FEATURE_VERSION) -> int:
    try:
        return len(FEATURE_SETS[version])
    except KeyError:
        raise ValueError(f"unknown feature version {version!r}; this code "
                         f"knows {sorted(FEATURE_SETS)}") from None


def hlld_r_vectors(U_init_L, F_init_L, U_init_R, F_init_R, cmin, cmax):
    """HLLDComputation's own jump vectors (note the -cmin on the left)."""
    RL = {k: -cmin * U_init_L[k] - F_init_L[k] for k in U_init_L}
    RR = {k: cmax * U_init_R[k] - F_init_R[k] for k in U_init_R}
    return RL, RR


def build_pstar_features_v3(uL, uR, RL, RR, cmin, cmax, idir: int = 0):
    """Assemble the (N, 17) v3 feature matrix from :func:`hlld_r_vectors`."""
    n = idir
    t1, t2 = (idir + 1) % 3, (idir + 2) % 3
    Bn = uL[_B[n]]
    sgn = torch.where(Bn < 0, -torch.ones_like(Bn), torch.ones_like(Bn))

    def vec(R, comp):
        return R[comp[t1]], R[comp[t2]]

    aL, bL = vec(RL, _S), vec(RL, _B)
    aR, bR = vec(RR, _S), vec(RR, _B)
    norm = lambda x: torch.sqrt(x[0] ** 2 + x[1] ** 2)
    naL, nbL, naR, nbR = norm(aL), norm(bL), norm(aR), norm(bR)

    def cos(x, nx, y, ny):
        den = nx * ny
        c = (x[0] * y[0] + x[1] * y[1]) / torch.where(den > 0, den,
                                                      torch.ones_like(den))
        return torch.where(den > 0, c.clamp(-1.0, 1.0), torch.zeros_like(c))

    return torch.stack([
        RL["tau"], RL[_S[n]], naL, nbL,
        RR["tau"], RR[_S[n]], naR, nbR,
        Bn.abs(), cmin, cmax,
        sgn * cos(aL, naL, bL, nbL), sgn * cos(aR, naR, bR, nbR),
        cos(aL, naL, aR, naR), cos(bL, nbL, bR, nbR),
        sgn * cos(aL, naL, bR, nbR), sgn * cos(bL, nbL, aR, naR),
    ], dim=1)


def pstar_features(version, sL, sR, uL, uR, U_init_L, F_init_L, U_init_R,
                   F_init_R, cmin, cmax, idir: int = 0):
    """The raw feature matrix of `version` -- the ONE call site shared by the
    data labeller (src/data/labelling.py) and hlld.hlld_ai_flux."""
    if version == FEATURE_VERSION:
        RL, RR = r_vectors(U_init_L, F_init_L, U_init_R, F_init_R, cmin, cmax)
        return build_pstar_features(sL, sR, uL, uR, RL, RR, cmin, cmax, idir)
    if version == FEATURE_VERSION_V3:
        RL, RR = hlld_r_vectors(U_init_L, F_init_L, U_init_R, F_init_R,
                                cmin, cmax)
        return build_pstar_features_v3(uL, uR, RL, RR, cmin, cmax, idir)
    n_features(version)            # raises with the list of known versions


def stats_str(norm_stats: dict, key: str, default=None):
    """A string stored in an npz (0-d array) or a plain dict, else default."""
    v = norm_stats.get(key)
    if v is None:
        return default
    return str(v.item()) if hasattr(v, "item") else str(v)


def stats_feature_version(norm_stats: dict):
    """The stamped feature version, or None for pre-versioning stats (the
    archived ones), which were built by v2's definition."""
    return stats_str(norm_stats, "feature_version")


# ── The pressure scale: an exact symmetry, used by the "ptot_avg" reference ──
#
# SRMHD is invariant under rho, p, B^2 -> lambda * (rho, p, B^2) at fixed
# velocities: every wave speed is unchanged, U, F and HLLD's jump vectors R
# scale by lambda (B and R_B by sqrt(lambda)), and so does p*.  HLLD's
# residual is homogeneous in the same way, so p*/p_ref is a function of
# scale-free inputs for any p_ref that scales like a pressure.  With
#
#     p_ref = (p_L + b2_L/2 + p_R + b2_R/2) / 2      (comoving b^2)
#
# -- the quantity hlld_ai_flux already computes for its HLLE fallback -- the
# reference also carries the answer for small jumps: L = R gives
# p* = p_ref exactly.  Norm stats with ``ref = "ptot_avg"`` divide each
# feature by p_ref ** X_REF_POWER (1 for pressure-like R components, 1/2 for
# field-like ones, 0 for speeds and cosines) and train on log10(p*/p_ref).
X_REF_POWER = {
    FEATURE_VERSION: (1, 1, 1, 0.5, 0, 0, 1, 1, 1, 0.5, 0, 0, 0.5, 0, 0),
    FEATURE_VERSION_V3: (1, 1, 1, 0.5, 1, 1, 1, 0.5, 0.5, 0, 0,
                         0, 0, 0, 0, 0, 0),
}


def p_ref_from_prims(sL, sR, eos):
    """(p + b^2/2) averaged over the two states, as hlld_ai_flux's fallback."""
    from .hlld import compute_b2, lorentz
    pL, _ = eos.press_and_cs2(sL["eps"], sL["rho"])
    pR, _ = eos.press_and_cs2(sR["eps"], sR["rho"])
    WL, _ = lorentz(sL)
    WR, _ = lorentz(sR)
    b2L = compute_b2((sL["vx"], sL["vy"], sL["vz"]), (sL["Bx"], sL["By"], sL["Bz"]), WL)
    b2R = compute_b2((sR["vx"], sR["vy"], sR["vz"]), (sR["Bx"], sR["By"], sR["Bz"]), WR)
    return torch.clamp(0.5 * (pL + 0.5 * b2L + pR + 0.5 * b2R), min=1e-30)


def stats_ref(norm_stats: dict) -> str:
    """"none" (the archived stats) or "ptot_avg"."""
    return stats_str(norm_stats, "ref", "none")


def scale_by_ref(X_raw: torch.Tensor, version: str, p_ref: torch.Tensor):
    """Raw features made scale-free: column j divided by p_ref**X_REF_POWER[j]."""
    pw = torch.as_tensor(X_REF_POWER[version], dtype=X_raw.dtype,
                         device=X_raw.device)
    return X_raw / p_ref.to(X_raw.dtype)[:, None] ** pw


def normalise(X_raw: torch.Tensor, norm_stats: dict,
              p_ref: torch.Tensor = None) -> torch.Tensor:
    """Raw features -> the network's float32 input, as at training time.

    ``ref`` (absent in the archived stats = "none") may be "ptot_avg": the
    features are first made scale-free with ``p_ref`` (see X_REF_POWER).
    ``x_transform`` (absent = "none") may be "asinh": x -> asinh(x/x_scale)
    per feature before standardising.  The split builder
    (scripts/pstar_build_splits.py) calls this same function on the
    training rows, so training and inference cannot drift apart.
    """
    X = X_raw
    if stats_ref(norm_stats) == "ptot_avg":
        if p_ref is None:
            raise ValueError("these norm stats need p_ref (ref = ptot_avg)")
        version = stats_feature_version(norm_stats) or FEATURE_VERSION
        X = scale_by_ref(X, version, p_ref)
    elif stats_ref(norm_stats) != "none":
        raise ValueError(f"unknown ref {stats_ref(norm_stats)!r}")
    tr = stats_str(norm_stats, "x_transform", "none")
    as_t = lambda k: torch.as_tensor(norm_stats[k], dtype=X.dtype,
                                     device=X.device)
    if tr == "asinh":
        X = torch.asinh(X / as_t("x_scale"))
    elif tr != "none":
        raise ValueError(f"unknown x_transform {tr!r}")
    return ((X - as_t("mu")) / as_t("sigma")).float()


def denormalise_target(y_pred: torch.Tensor, norm_stats: dict,
                       p_ref: torch.Tensor = None) -> torch.Tensor:
    """Network output -> p*.  The target is standardised log10 p*, or
    log10(p*/p_ref) when the stats say ref = "ptot_avg"."""
    y_mu = float(norm_stats["y_mu"])
    y_sigma = float(norm_stats["y_sigma"])
    p = 10.0 ** (y_pred.double() * y_sigma + y_mu)
    if stats_ref(norm_stats) == "ptot_avg":
        p = p * p_ref.double()
    return torch.clamp(p, min=1e-30)
