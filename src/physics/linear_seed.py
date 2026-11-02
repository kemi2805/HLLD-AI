"""The linearised fan as the START of the exact solvers on weak faces.

WHY
---
With the weak-jump gate open (the GR06 shock tubes, 2026-10-30, every face
attempted) the exact flux loses its faces where the jump is small: on
Balsara 5 at 200 cells 7,360 of the 8,146 faces it lost had a relative jump
below 1e-2, where only 38-64% of the faces between 1e-7 and 1e-4 came out
exact, against 95-97% above 1e-2; Balsara 2 and 3 lose the same way.  The
same faces are what those runs pay for: the seven-wave Newton fails on them
from the network's seed, every rescue rung runs, and on Balsara 5 the
continuation rung ends up answering 21% of all faces (40% at 800 cells).

A weak Riemann problem is a small perturbation of the mean state, and its
exact solution is the linearised fan to second order in the jump
(`linearised.py`): every zone of the fan is within O(jump^2) of the exact
one.  So the fan is the natural start for the exact solvers there -- far
closer than a network trained on strong jumps can place one.

WHAT
----
`linear_fan` is the decomposition `linearised_flux` makes (the same
Jacobians by central differences at the mean state, the same eigenvectors,
the same trust tests), returning the eight zones R1..R8 in wave order instead
of the state on the ray.  It is kept apart from `linearised_flux` so that
production's weak-face flux stays bit for bit what it was.  `seed6` reads the
seven-wave unknowns ``[ln p_LF, ln|Bt_CD|, psi_CD, ln p_RF, phi_L, phi_R]``
off the zones, `seed_planar` the five-wave planar ones ``[ln p_LF, Bt_CD,
ln p_RF]`` (signed in the planar frame of `planar5_b.to_planar`).

The fan is only a start: whatever the Newton returns from it passes exactly
the acceptance of the answers it replaces (`exact_flux`, RMHD_LINEAR_SEED).
"""
from __future__ import annotations

import numpy as np
import torch

from .linearised import _UF, _axes, _eig_batched, _solve_batched


def _pmag(Z, Bn):
    """Magnetic pressure b^2/2 of a solver-frame zone ``(rho, p, vn, vt1,
    vt2, Bt1, Bt2)`` (either pressure: it does not enter)."""
    _, _, vn, vt1, vt2, Bt1, Bt2 = Z
    W2 = 1.0 / np.maximum(1.0 - (vn * vn + vt1 * vt1 + vt2 * vt2), 1e-300)
    eta = Bn * vn + Bt1 * vt1 + Bt2 * vt2
    return 0.5 * ((Bn * Bn + Bt1 * Bt1 + Bt2 * Bt2) / W2 + eta * eta)


def _ptot(Z, Bn):
    """Total pressure of a zone carrying the gas pressure."""
    return Z[1] + _pmag(Z, Bn)


def linear_fan(left, right, Bn, eos, fd_rel: float = 1e-5,
               imag_tol: float = 1e-6, cond_max: float = 1e12):
    """The linearised fan of solver-frame states.

    ``left``/``right`` are 7-lists ``(rho, P_tot, v_n, v_t1, v_t2, B_t1,
    B_t2)`` of per-lane arrays (the exact solver's frame, total pressure),
    ``Bn`` the normal field.  Returns a dict:

      ok     the lanes whose fan can be trusted: a real spectrum, a
             well-conditioned eigenvector matrix, the eight zones physical
      lam    (N, 7) eigenvalues, ascending: LF LA LS CD RS RA RF
      zones  [R1..R8], each a 7-list of (N,) arrays ``(rho, p, ...)`` with
             the GAS pressure; R1 is the left state, R8 the right one to
             rounding
    """
    vk, bk, sk = _axes(0)
    dt = torch.float64
    L = [np.asarray(c, dtype=float) for c in left]
    R = [np.asarray(c, dtype=float) for c in right]
    Bn = np.asarray(Bn, dtype=float)
    N = Bn.shape[0]

    VL = torch.tensor(np.stack([L[0], L[1] - _pmag(L, Bn)] + L[2:]), dtype=dt)
    VR = torch.tensor(np.stack([R[0], R[1] - _pmag(R, Bn)] + R[2:]), dtype=dt)
    Bt = torch.tensor(Bn, dtype=dt)
    Vm = 0.5 * (VL + VR)

    # ── the two Jacobians by central differences (linearised_flux's) ─────
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
        Up, Fp = _UF(Vp, Bt, eos, 0, vk, bk, sk)
        Uq, Fq = _UF(Vq, Bt, eos, 0, vk, bk, sk)
        dU[:, :, j] = ((Up - Uq) / (2.0 * h[j])).T
        dF[:, :, j] = ((Fp - Fq) / (2.0 * h[j])).T

    ok = torch.isfinite(dU).all(dim=(1, 2)) & torch.isfinite(dF).all(dim=(1, 2))
    ok &= (VL[0] > 0) & (VR[0] > 0) & (VL[1] > 0) & (VR[1] > 0)
    A = torch.zeros(N, 7, 7, dtype=dt)
    if ok.any():
        i = torch.nonzero(ok).reshape(-1)
        A[i], solved = _solve_batched(dU[i], dF[i])
        ok[i] = solved

    # ── eigen-decomposition at the mean state, waves in ascending order ──
    lam = torch.zeros(N, 7, dtype=dt)
    Rm = torch.zeros(N, 7, 7, dtype=dt)
    if ok.any():
        i = torch.nonzero(ok).reshape(-1)
        w, Rc, eig_ok = _eig_batched(A[i])
        real_ok = eig_ok & (w.imag.abs() <= imag_tol * (1.0 + w.real.abs())).all(dim=1)
        big = torch.argmax(Rc.abs(), dim=1, keepdim=True)
        ph = torch.gather(Rc, 1, big)
        Rc = Rc * (ph.conj() / ph.abs().clamp(min=1e-300))
        real_ok &= (Rc.imag.abs() <= imag_tol * (1.0 + Rc.real.abs())).all(dim=(1, 2))
        order = torch.argsort(w.real, dim=1)
        lam[i] = torch.gather(w.real, 1, order)
        Rr = torch.gather(Rc.real, 2, order.unsqueeze(1).expand(-1, 7, -1))
        Rr = Rr / torch.linalg.norm(Rr, dim=1, keepdim=True).clamp(min=1e-300)
        Rm[i] = Rr
        cond = torch.linalg.cond(Rr)
        ok[i] = real_ok & torch.isfinite(cond) & (cond < cond_max)

    # ── the strengths and the zones ──────────────────────────────────────
    Z = [VL.clone() for _ in range(8)]
    if ok.any():
        i = torch.nonzero(ok).reshape(-1)
        dV = (VR - VL)[:, i].T.unsqueeze(-1)
        alpha, solved = _solve_batched(Rm[i], dV)
        alpha = alpha.squeeze(-1)                               # (n, 7)
        cur = VL[:, i].clone()
        for k in range(7):
            cur = cur + (Rm[i][:, :, k] * alpha[:, k:k + 1]).T
            Z[k + 1][:, i] = cur
        ok[i] &= solved
    zones = [[z[c].numpy().copy() for c in range(7)] for z in Z]
    okn = ok.numpy().copy()
    for z in zones:
        v2 = z[2] ** 2 + z[3] ** 2 + z[4] ** 2
        okn &= (z[0] > 0) & (z[1] > 0) & (v2 < 1.0)
        for c in z:
            okn &= np.isfinite(c)
    return dict(ok=okn, lam=lam.numpy(), zones=zones)


def _psi(Z):
    return np.arctan2(Z[6], Z[5])


def _wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def seed6(fan, Bn, lanes=None):
    """The seven-wave unknowns ``[ln p_LF, ln|Bt_CD|, psi_CD, ln p_RF,
    phi_L, phi_R]`` of the fan (fullcontact_b's layout: p the TOTAL pressure
    behind each fast wave, Bt_CD the field at the contact, phi_L = psi(R3) -
    psi(R2) and phi_R = psi(R7) - psi(R6) the two rotations), on ``lanes``
    (default: all)."""
    Bn = np.asarray(Bn, dtype=float)
    idx = np.arange(Bn.shape[0]) if lanes is None else np.asarray(lanes)
    Z = [[c[idx] for c in z] for z in fan["zones"]]
    b = Bn[idx]
    R2, R3, R4, R6, R7 = Z[1], Z[2], Z[3], Z[5], Z[6]
    return [np.log(np.maximum(_ptot(R2, b), 1e-300)),
            np.log(np.maximum(np.hypot(R4[5], R4[6]), 1e-300)),
            _psi(R4),
            np.log(np.maximum(_ptot(R7, b), 1e-300)),
            _wrap(_psi(R3) - _psi(R2)),
            _wrap(_psi(R7) - _psi(R6))]


def seed_planar(fan, left, right, Bn, lanes=None):
    """The five-wave planar unknowns ``[ln p_LF, Bt_CD, ln p_RF]`` of the fan,
    Bt_CD signed in `planar5_b.to_planar`'s frame (the left tangential field
    along +y), on ``lanes`` (default: all)."""
    from rmhd.batched import planar5_b as P5B
    Bn = np.asarray(Bn, dtype=float)
    idx = np.arange(Bn.shape[0]) if lanes is None else np.asarray(lanes)
    alpha = P5B.to_planar([np.asarray(c)[idx] for c in left],
                          [np.asarray(c)[idx] for c in right], Bn[idx])[2]
    Z = [[c[idx] for c in z] for z in fan["zones"]]
    b = Bn[idx]
    R4 = Z[3]
    bt = np.cos(alpha) * R4[5] - np.sin(alpha) * R4[6]
    return [np.log(np.maximum(_ptot(Z[1], b), 1e-300)), bt,
            np.log(np.maximum(_ptot(Z[6], b), 1e-300))]
