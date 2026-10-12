"""Solve every face in one canonical orientation (``RMHD_CANONICAL``).

A one-dimensional Riemann problem has exact symmetries: the reflection
x -> -x (left and right exchanged, the normal components of v and B negated),
any orthogonal map of the two tangential axes, and B -> -B.  The exact
solution respects them; an iterative solver does not -- its seeds, its
unknowns and its rescue rungs see the components in a fixed order.  On
Orszag-Tang that breaks the pi-rotation symmetry: on 0.4-0.7% of the face
pairs one face is solved and its mirror image falls back to HLLD (calea job
40966, docs/what_we_solve.md section 7).  And x- and y-faces of a 2D problem
put the in-plane tangential field in different slots of the solver frame
((n, t1, t2) is cyclic: (x, y, z) and (y, z, x)), so the two directions are
answered by different solvers.

The discrete part of the symmetry group is 32 elements: the reflection (2)
times the signed permutations of (t1, t2) (8) times the sign of B (2).  All
of them act by exchanges and sign flips, which are exact in floating point.
Each face is replaced by the smallest member of its orbit by a fixed key
(B_n first, so that it is non-negative, then the two states' components;
see KEY and TOL), solved in the x-frame (``idir = 0``), and the answer is
mapped back.  Two faces related by any element reduce to the same problem,
bit for bit, so they get the same answer: the scheme's symmetry is then
limited by its rounding, not by the solver.

The answer maps back as follows, with A the tangential map and the sign of B,
P the negation of the normal components:

    no reflection    L' = A L,    R' = A R      F = A^-1 F',    U = A^-1 U'
    reflection       L' = A P R,  R' = A P L    F = -P A^-1 F', U = P A^-1 U'

(the normal flux of a reflected state is -P times the flux of the state:
the D, tau and tangential fluxes change sign, the normal-momentum flux does
not).  ``p_star`` is a scalar.  The diagnostics (LAST_DIAG) and a harvest
describe the canonical problems; LAST_DIAG gains the element per face
(``canonical_g``, 0 = the identity) and ``n_canonical_moved``.
"""
from __future__ import annotations

import numpy as np
import torch

# (reflection, exchange t1/t2, sign of t1, sign of t2, sign of B); the
# identity first, so a tie (a face symmetric under some element) keeps it
GROUP = [(sw, pm, s1, s2, b) for sw in (0, 1) for pm in (0, 1)
         for s1 in (1, -1) for s2 in (1, -1) for b in (1, -1)]
_V = ("vx", "vy", "vz")
_B = ("Bx", "By", "Bz")
_S = ("Sx", "Sy", "Sz")


def _axes(idir):
    """Lab axes of the solver frame (n, t1, t2): cyclic, as to_solver_frame."""
    return idir, (idir + 1) % 3, (idir + 2) % 3


def _columns(s, idir):
    """The eight key columns of one state, in the solver frame."""
    n, t1, t2 = _axes(idir)
    return [s["rho"], s["p"], s[_V[n]], s[_V[t1]], s[_V[t2]],
            s[_B[n]], s[_B[t1]], s[_B[t2]]]


def _table():
    """For each element and each of the 16 key columns of the image: which
    input column it is and with which sign.  Input columns 0-7 are the left
    state's, 8-15 the right's, each (rho, p, vn, vt1, vt2, Bn, Bt1, Bt2)."""
    src = np.zeros((len(GROUP), 16), np.int64)
    sgn = np.ones((len(GROUP), 16))
    for g, (sw, pm, s1, s2, b) in enumerate(GROUP):
        for side in (0, 1):
            base = 8 * ((side + sw) % 2)          # the reflection exchanges L, R
            o = 8 * side
            ta, tb = (4, 3) if pm else (3, 4)     # which input feeds t1, t2
            src[g, o:o + 8] = base + np.array([0, 1, 2, ta, tb, 5, ta + 3, tb + 3])
            sgn[g, o + 2] = -1.0 if sw else 1.0                    # v_n
            sgn[g, o + 3], sgn[g, o + 4] = s1, s2                  # v_t
            sgn[g, o + 5] = (-1.0 if sw else 1.0) * b              # B_n
            sgn[g, o + 6], sgn[g, o + 7] = s1 * b, s2 * b          # B_t
    return src, sgn


_SRC, _SGN = _table()


# The key, in the image's 16 columns (rho, p, vn, vt1, vt2, Bn, Bt1, Bt2 of
# the left state, then the right), and the sign it is compared with: B_n
# FIRST and negated, so every face is solved with B_n >= 0 -- the orientation
# the seed networks were trained on (generate_dataset samples B_x > 0; on the
# test fixture the plain solver solves 29 of 32 faces as generated and 23 of
# their mirror images, which have B_n < 0) -- then the left state, then the
# right state's remaining columns.
KEY = [(5, -1.0), (0, 1.0), (1, 1.0), (2, 1.0), (3, 1.0), (4, 1.0), (6, 1.0),
       (7, 1.0), (8, 1.0), (9, 1.0), (10, 1.0), (11, 1.0), (12, 1.0),
       (14, 1.0), (15, 1.0), (13, 1.0)]
# Values within TOL of each other, relative to the face's own scale of that
# kind of variable, compare as equal when the element is chosen.  Without
# it rounding noise decides: on the four-quadrant problem at t = 0.05 the
# two faces of 2,927 of 4,032 diagonal mirror pairs (inputs equal to 4e-11)
# took different orientations, wherever two values were nearly tied (the
# uniform quadrants: rho_L ~ rho_R, v_t ~ 0), and the run's symmetry was
# seeded at 8e-8 instead of 3e-11.  The surviving ties are then broken
# exactly, which keeps exact images bit for bit equivariant.
TOL = 1e-7
_KIND = [0, 1, 2, 2, 2, 3, 3, 3] * 2          # rho, p, v, B per column


def choose(sL, sR, idir):
    """Per face, the index into GROUP of the element that maps it to its
    canonical image: the smallest by KEY, first with values within TOL of
    the face's scale counted as equal, then exactly among what is left."""
    X = torch.stack([c.reshape(-1) for c in _columns(sL, idir) + _columns(sR, idir)])
    X = X.detach().cpu().numpy()
    n = X.shape[1]
    # the face's scales, invariant under every element (maxima over both
    # sides and all components): rho, p, |v| (at least 1: v is in units of
    # c), |B| (at least sqrt(p))
    s_rho = np.maximum(np.abs(X[0]), np.abs(X[8]))
    s_p = np.maximum(np.abs(X[1]), np.abs(X[9]))
    s_v = np.ones(n)
    s_B = np.maximum(np.abs(X[[5, 6, 7, 13, 14, 15]]).max(axis=0), np.sqrt(s_p))
    scale = np.stack([s_rho, s_p, s_v, s_B])
    alive = np.ones((len(GROUP), n), bool)
    with np.errstate(invalid="ignore"):
        for tol in (TOL, 0.0):
            for col, sign in KEY:
                vals = sign * _SGN[:, col, None] * X[_SRC[:, col]]
                vals = np.where(alive, vals, np.inf)
                alive &= vals <= vals.min(axis=0) + tol * scale[_KIND[col]]
    # a face with a NaN keeps every comparison false: leave it as it is
    return np.where(alive.any(axis=0), alive.argmax(axis=0), 0)


def _per_face(g, like):
    """The element's parts as per-face tensors (exact +-1 and booleans)."""
    G = np.asarray(GROUP)[g]
    t = lambda a: torch.as_tensor(a, dtype=like.dtype, device=like.device)
    sw = torch.as_tensor(G[:, 0].astype(bool), device=like.device)
    pm = torch.as_tensor(G[:, 1].astype(bool), device=like.device)
    return sw, pm, t(G[:, 2]), t(G[:, 3]), t(G[:, 4])


def to_canonical(sL, sR, idir):
    """The canonical problems, as primitive dicts in the x-frame, and the
    element per face."""
    g = choose(sL, sR, idir)
    cL, cR = transform(sL, sR, idir, g)
    return cL, cR, g


def transform(sL, sR, idir, g):
    """The image of each face under the element ``g`` (per face, an index
    into GROUP), as primitive dicts in the x-frame."""
    flat = lambda s: {k: v.reshape(-1) for k, v in s.items()}
    sL, sR = flat(sL), flat(sR)
    sw, pm, s1, s2, b = _per_face(g, sL["rho"])
    n, t1, t2 = _axes(idir)
    out = []
    for first, second in ((sL, sR), (sR, sL)):
        # the reflection takes the other side's state
        src = {k: torch.where(sw, second[k], first[k]) for k in first}
        d = {k: v.clone() for k, v in src.items() if k not in _V + _B}
        for vec, scale in ((_V, 1.0), (_B, b)):
            vn, va, vb = src[vec[n]], src[vec[t1]], src[vec[t2]]
            ta = torch.where(pm, vb, va)
            tb = torch.where(pm, va, vb)
            d[vec[0]] = torch.where(sw, -vn, vn) * scale
            d[vec[1]] = s1 * ta * scale
            d[vec[2]] = s2 * tb * scale
        out.append(d)
    return out[0], out[1]


def _back(Q, g, idir, flux):
    """Map an answer of the canonical problems (x-frame dict: D, tau, the
    momentum S and the field B) back to the faces' own frame."""
    sw, pm, s1, s2, b = _per_face(g, Q["D"])
    n, t1, t2 = _axes(idir)
    out = {}
    for k in Q:
        if k not in _S + _B:
            # D and tau: scalars, but the reflection reverses their flux
            out[k] = torch.where(sw, -Q[k], Q[k]) if flux else Q[k].clone()
    for vec, scale in ((_S, 1.0), (_B, b)):
        cn, ca, cb = Q[vec[0]] * scale, s1 * Q[vec[1]] * scale, s2 * Q[vec[2]] * scale
        va = torch.where(pm, cb, ca)              # undo the exchange
        vb = torch.where(pm, ca, cb)
        if flux:     # F = -P A^-1 F': the normal component keeps its sign
            vn, va, vb = cn, torch.where(sw, -va, va), torch.where(sw, -vb, vb)
        else:        # U = P A^-1 U'
            vn = torch.where(sw, -cn, cn)
        out[vec[n]], out[vec[t1]], out[vec[t2]] = vn, va, vb
    return out


def canonical_flux(flux_fn, sL, sR, eos, idir, **kw):
    """``flux_fn`` (exact_flux_batched) on the canonical problems, mapped
    back.  ``flux_fn`` must take ``_record_idir`` (the face direction for the
    harvest's coverage map) and must not canonicalise again."""
    from .hlld import LAST_DIAG
    shape = sL["rho"].shape
    cL, cR, g = to_canonical(sL, sR, idir)
    F, U, p_star = flux_fn(cL, cR, eos, idir=0, _record_idir=idir, **kw)
    # A face that is itself symmetric -- in a 2D run every face has v_z = B_z
    # = 0 -- reaches its canonical problem through more than one element, and
    # mapping back through a different one flips the sign of whatever the
    # solver left in the components that symmetry forces to zero.  Those
    # components ARE zero in the exact solution (no tangential velocity or
    # field along an axis on either side: the flow stays out of it), so they
    # are set to zero, which makes the 2D case exactly equivariant.  Faces
    # symmetric under an exchange or the reflection (equal tangential
    # components, mirror-image sides) are not treated: measure zero.
    for a, keys in (("y", ("Sy", "By")), ("z", ("Sz", "Bz"))):
        flat = ((cL["v" + a] == 0) & (cL["B" + a] == 0)
                & (cR["v" + a] == 0) & (cR["B" + a] == 0))
        if bool(flat.any()):
            for Q in (F, U):
                for k in keys:
                    Q[k] = torch.where(flat, torch.zeros_like(Q[k]), Q[k])
    F, U = _back(F, g, idir, flux=True), _back(U, g, idir, flux=False)
    LAST_DIAG.update(idir=idir, canonical_g=g.astype(np.int8),
                     n_canonical_moved=int((g != 0).sum()))
    shp = lambda d: {k: v.reshape(shape) for k, v in d.items()}
    return shp(F), shp(U), p_star.reshape(shape)
