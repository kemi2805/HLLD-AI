"""The linearised (Roe-type) Riemann solver: the exact solution to second
order in the jump, at the cost of one eigen-decomposition per face.

WHY
---
Production attempts the exact solver only on faces whose relative jump
exceeds ``tau_weak`` (1e-2); the other ~86% take HLLD.  There the exact flux
and HLLD's differ by ~2.5e-5 of the flux at the threshold and the difference
falls off linearly below it (docs/what_we_solve.md 4b) -- HLLD's error is
FIRST order in the jump.  A weak Riemann problem is a small perturbation
about the mean state, and its exact solution is the LINEARISED one to second
order: the jump decomposed into the characteristic fields of the mean state,
each wave a plane discontinuity travelling at its eigenvalue.  So on the
faces below the gate this solver's error scales as the square of the jump,
against HLLD's first power, for 14 flux evaluations and a 7x7 eigenproblem
per face and no root-finding.

WHAT
----
In primitive variables V = (rho, p, vn, vt1, vt2, Bt1, Bt2) with the normal
field fixed (it is single-valued at a face), the quasi-linear form of the
normal flux is

    dV/dt + A dV/dx = 0,      A = (dU/dV)^-1 (dF/dV)

with U the conserved variables and F the normal flux, both from the same
code that builds HLLD's (`compute_srmhd_fluxes`); the two Jacobians are
central differences in V.  At the arithmetic mean of the two states, A is
decomposed as A R = R diag(lambda), the jump as V_R - V_L = R alpha, and the
state on the ray xi = 0 is

    V* = V_L + sum over lambda_k < 0 of alpha_k r_k

(equivalently V_R minus the waves that run to the right).  The flux is the
physical flux of V* -- the same convention as the exact flux, which also
returns F(state at xi = 0).

Faces where this cannot be trusted keep the fallback: an eigenvalue with a
non-negligible imaginary part (the system is hyperbolic, so that is
round-off at a degeneracy), an eigenvector matrix that cannot be inverted,
or a V* that is not physical.  The mean state's Jacobian is what makes this
a linearisation rather than a Roe solver proper -- SRMHD has no closed-form
Roe average -- which is fine to second order and is all that is claimed.

This is NOT an exact solver: nothing is verified against jump conditions,
and its answers are never counted as exact.  It is the candidate for the
faces the gate hands to HLLD (``RMHD_WEAK_FLUX=linear`` in the exact flux),
to be judged by measurement.
"""
from __future__ import annotations

import torch

from .eos import hybrid_eos
from .hlld import Cons, Prim, compute_srmhd_fluxes, hlld_flux

_VARS = ("rho", "p", "vn", "vt1", "vt2", "Bt1", "Bt2")
_CONS = ("D", "tau", "Sn", "St1", "St2", "Bt1", "Bt2")
LAST_LINEAR: dict = {}


def _axes(idir: int):
    v = ["vx", "vy", "vz"]
    b = ["Bx", "By", "Bz"]
    t1, t2 = [(1, 2), (0, 2), (0, 1)][idir]
    return (v[idir], v[t1], v[t2]), (b[idir], b[t1], b[t2]), \
        ("S" + "xyz"[idir], "S" + "xyz"[t1], "S" + "xyz"[t2])


def _prim(V, Bn, eos, vk, bk) -> Prim:
    rho, p, vn, vt1, vt2, Bt1, Bt2 = V
    d = {"rho": rho, "p": p, vk[0]: vn, vk[1]: vt1, vk[2]: vt2,
         bk[0]: Bn, bk[1]: Bt1, bk[2]: Bt2}
    d["eps"] = eos.eps__press_rho(p, rho)
    return d


def _UF(V, Bn, eos, idir, vk, bk, sk):
    """Conserved (7, N) and normal flux (7, N) of the primitive columns."""
    prim = _prim(V, Bn, eos, vk, bk)
    u, _, f, _, _, _ = compute_srmhd_fluxes(prim, prim, eos, idir)
    U = torch.stack([u["D"], u["tau"], u[sk[0]], u[sk[1]], u[sk[2]],
                     u[bk[1]], u[bk[2]]])
    F = torch.stack([f["D"], f["tau"], f[sk[0]], f[sk[1]], f[sk[2]],
                     f[bk[1]], f[bk[2]]])
    return U, F


def _solve_batched(M, B):
    """``M^-1 B`` per element; an element whose solve fails is flagged
    instead of failing the batch."""
    try:
        return torch.linalg.solve(M, B), torch.ones(M.shape[0], dtype=torch.bool)
    except RuntimeError:
        pass
    X = torch.zeros_like(B)
    ok = torch.zeros(M.shape[0], dtype=torch.bool)
    for n in range(M.shape[0]):
        try:
            X[n] = torch.linalg.solve(M[n], B[n])
            ok[n] = True
        except RuntimeError:
            pass
    return X, ok


def _eig_batched(A):
    """Eigen-decomposition per element.  LAPACK refuses the WHOLE batch when
    one element does not converge (measured on the rotor: one face in
    ~10^5), so that element is retried alone and flagged if it fails."""
    n = A.shape[0]
    try:
        w, R = torch.linalg.eig(A)
        return w, R, torch.ones(n, dtype=torch.bool)
    except RuntimeError:
        pass
    w = torch.zeros(n, A.shape[1], dtype=torch.complex128)
    R = torch.zeros(n, A.shape[1], A.shape[2], dtype=torch.complex128)
    ok = torch.zeros(n, dtype=torch.bool)
    for k in range(n):
        try:
            w[k], R[k] = torch.linalg.eig(A[k])
            ok[k] = True
        except RuntimeError:
            pass
    return w, R, ok


def linearised_flux(sL: Prim, sR: Prim, eos: hybrid_eos, idir: int = 0,
                    fallback=hlld_flux, fd_rel: float = 1e-5,
                    imag_tol: float = 1e-6, cond_max: float = 1e12,
                    return_mask: bool = False):
    """Same contract as ``hlld_flux``: ``(F, U, p_star)`` for every face;
    with ``return_mask`` also the per-face mask of faces this solver
    answered (the rest carry the fallback).  ``p_star`` is the total
    pressure of the state at xi = 0."""
    vk, bk, sk = _axes(idir)
    F0, U0, p0 = fallback(sL, sR, eos, idir=idir)
    dt = sL["rho"].dtype
    N = sL["rho"].numel()
    shape = sL["rho"].shape
    col = lambda s, k: s[k].reshape(-1).to(dt)
    VL = torch.stack([col(sL, "rho"), col(sL, "p"), col(sL, vk[0]),
                      col(sL, vk[1]), col(sL, vk[2]), col(sL, bk[1]),
                      col(sL, bk[2])])
    VR = torch.stack([col(sR, "rho"), col(sR, "p"), col(sR, vk[0]),
                      col(sR, vk[1]), col(sR, vk[2]), col(sR, bk[1]),
                      col(sR, bk[2])])
    Bn = 0.5 * (col(sL, bk[0]) + col(sR, bk[0]))
    Vm = 0.5 * (VL + VR)

    # ── the two Jacobians by central differences ────────────────────────
    scale = torch.stack([Vm[0].abs(), Vm[1].abs(), torch.ones_like(Vm[2]),
                         torch.ones_like(Vm[3]), torch.ones_like(Vm[4]),
                         torch.sqrt(Vm[1].abs()) + Vm[5].abs(),
                         torch.sqrt(Vm[1].abs()) + Vm[6].abs()])
    h = fd_rel * scale.clamp(min=1e-300)
    dU = torch.zeros(N, 7, 7, dtype=dt)
    dF = torch.zeros(N, 7, 7, dtype=dt)
    for j in range(7):
        Vp = Vm.clone()
        Vp[j] = Vm[j] + h[j]
        Vq = Vm.clone()
        Vq[j] = Vm[j] - h[j]
        Up, Fp = _UF(Vp, Bn, eos, idir, vk, bk, sk)
        Uq, Fq = _UF(Vq, Bn, eos, idir, vk, bk, sk)
        dU[:, :, j] = ((Up - Uq) / (2.0 * h[j])).T
        dF[:, :, j] = ((Fp - Fq) / (2.0 * h[j])).T

    ok = torch.isfinite(dU).all(dim=(1, 2)) & torch.isfinite(dF).all(dim=(1, 2))
    ok &= (VL[0] > 0) & (VR[0] > 0) & (VL[1] > 0) & (VR[1] > 0)
    A = torch.zeros(N, 7, 7, dtype=dt)
    if ok.any():
        i = torch.nonzero(ok).reshape(-1)
        A[i], solved = _solve_batched(dU[i], dF[i])
        ok[i] = solved

    # ── eigen-decomposition at the mean state ──────────────────────────
    lam = torch.zeros(N, 7, dtype=dt)
    Rm = torch.zeros(N, 7, 7, dtype=dt)
    if ok.any():
        i = torch.nonzero(ok).reshape(-1)
        w, Rc, eig_ok = _eig_batched(A[i])
        real_ok = eig_ok & (w.imag.abs() <= imag_tol * (1.0 + w.real.abs())).all(dim=1)
        # a real eigenvector comes back times an arbitrary complex phase:
        # take the phase off (by the largest component) before the real part
        big = torch.argmax(Rc.abs(), dim=1, keepdim=True)          # (n, 1, 7)
        ph = torch.gather(Rc, 1, big)                              # (n, 1, 7)
        Rc = Rc * (ph.conj() / ph.abs().clamp(min=1e-300))
        real_ok &= (Rc.imag.abs() <= imag_tol * (1.0 + Rc.real.abs())).all(dim=(1, 2))
        lam[i] = w.real
        Rm[i] = Rc.real
        # normalise the eigenvectors, then judge their conditioning
        nrm = torch.linalg.norm(Rm[i], dim=1, keepdim=True).clamp(min=1e-300)
        Rm[i] = Rm[i] / nrm
        cond = torch.linalg.cond(Rm[i])
        good = real_ok & torch.isfinite(cond) & (cond < cond_max)
        ok[i] = good

    # ── the state on the ray ────────────────────────────────────────────
    Vs = VL.clone()
    if ok.any():
        i = torch.nonzero(ok).reshape(-1)
        dV = (VR - VL)[:, i].T.unsqueeze(-1)                    # (n, 7, 1)
        alpha, solved = _solve_batched(Rm[i], dV)
        alpha = alpha.squeeze(-1)                               # (n, 7)
        left = (lam[i] < 0.0).to(dt)                            # waves at xi < 0
        step = torch.einsum("nkj,nj->nk", Rm[i], alpha * left)  # (n, 7)
        Vs[:, i] = torch.where(solved[None, :], VL[:, i] + step.T, VL[:, i])
        ok[i] &= solved
    v2 = Vs[2] ** 2 + Vs[3] ** 2 + Vs[4] ** 2
    ok &= (Vs[0] > 0) & (Vs[1] > 0) & (v2 < 1.0) & torch.isfinite(Vs).all(dim=0)

    # ── its flux ────────────────────────────────────────────────────────
    F = {k: v.clone() for k, v in F0.items()}
    U = {k: v.clone() for k, v in U0.items()}
    p_star = p0.clone()
    if ok.any():
        i = torch.nonzero(ok).reshape(-1)
        Vi = Vs[:, i]
        prim = _prim(Vi, Bn[i], eos, vk, bk)
        u, _, f, _, _, _ = compute_srmhd_fluxes(prim, prim, eos, idir)
        for k in F:
            F[k].reshape(-1)[i] = f[k]
            U[k].reshape(-1)[i] = u[k]
        # total pressure of the state on the ray
        W2 = 1.0 / (1.0 - (Vi[2] ** 2 + Vi[3] ** 2 + Vi[4] ** 2))
        eta = Bn[i] * Vi[2] + Vi[5] * Vi[3] + Vi[6] * Vi[4]
        b2 = (Bn[i] ** 2 + Vi[5] ** 2 + Vi[6] ** 2) / W2 + eta ** 2
        p_star.reshape(-1)[i] = Vi[1] + 0.5 * b2
    LAST_LINEAR.clear()
    LAST_LINEAR.update(n=N, n_linear=int(ok.sum()),
                       frac_fallback=1.0 - float(ok.sum()) / max(N, 1))
    if return_mask:
        return F, U, p_star, ok.reshape(shape)
    return F, U, p_star

