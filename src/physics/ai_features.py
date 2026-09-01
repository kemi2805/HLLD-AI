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
