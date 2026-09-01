"""
Exact Riemann solver as a Godunov flux function.

Drop-in replacement for ``hlld_flux``: same
``flux_fn(primL, primR, eos, idir) -> (F, U, p_star)`` contract, so
``sweep()`` and ``run_2d.py`` need only name it.

This is the SCALAR reference implementation (Phase 1).  It loops over
interfaces and costs ~6 s each, so it is not usable for a production 2D run
-- its purpose is to pin down the physics (frame rotation, the xi = 0 ray,
flux assembly, the fallback contract) and to be the answer the batched
kernels must reproduce.

How a flux is obtained
----------------------
1. Rotate into the sweep frame (normal component first) and convert the gas
   pressure the 2D code carries into the TOTAL pressure the exact solver
   uses.
2. Triage: unphysical or near-uniform interfaces, and every structure class
   the seven-wave solver cannot handle, go straight to HLLD.
3. ML warm start -> ``fullcontact6`` -> the resolved state on the xi = 0 ray.
4. Assemble the flux by calling ``compute_srmhd_fluxes(s*, s*, eos, idir)``
   with the SAME state on both sides: it then returns ``fL == fR == F(U*)``
   and ``uL == uR == U*``, i.e. the full eight-key conserved-flux dict that
   constrained transport reads, computed by the code the rest of the repo
   already trusts.
5. Anything that fails any gate falls back to HLLD, and the fraction is
   reported rather than hidden.

Why the weak-jump gate is not a cop-out
---------------------------------------
Reconstruction leaves ``|q_L - q_R| ~ 1e-8`` across most of a smooth region.
There the exact and linearised fluxes agree to ``O(jump^2)``, so switching is
accuracy-neutral -- while running the exact solver would be both wasteful and
ill-conditioned, since multi-root ranking by Rankine-Hugoniot residual is
meaningless when every residual is at round-off and the finite-difference
Jacobian (EPS = 1e-6) would difference a function whose variation is smaller
than its own discretisation defect.
"""

from __future__ import annotations

import os
import sys

import torch

from .hlld import LAST_DIAG, compute_srmhd_fluxes, hlld_flux

_RMHD_ROOT = "/Users/miler/Codes/rmhd_final"
if _RMHD_ROOT not in sys.path:
    # APPEND, never insert: rmhd_final uses flat top-level module names
    # (eos, solver, contact, hlle) and must never shadow this package's own.
    sys.path.append(_RMHD_ROOT)

_V = ("vx", "vy", "vz")
_B = ("Bx", "By", "Bz")
_S = ("Sx", "Sy", "Sz")


def _check_eos(eos, gamma_expected=None):
    """The two repos must agree on the equation of state.

    ``rmhd_final``'s ``enthalpy_scalar`` gives ``h = 1 + gamma p / ((gamma-1) rho)``;
    ``hybrid_eos`` gives ``h = 1 + gamma_th p / ((gamma_th-1) rho)`` plus a cold
    part scaled by ``K``.  They coincide only for ``K == 0`` and
    ``gamma_th == gamma`` -- true for the 2D runs, but solving a different
    problem than intended is exactly the kind of error that produces
    plausible output, so it is asserted rather than assumed.
    """
    K = float(getattr(eos, "K", 0.0))
    g = float(getattr(eos, "gamma"))
    gt = float(getattr(eos, "gamma_th", g))
    if K != 0.0 or abs(gt - g) > 1e-14:
        raise ValueError(
            f"exact_flux requires an ideal gas (K=0, gamma_th==gamma); got "
            f"K={K}, gamma={g}, gamma_th={gt}"
        )
    return g


def _b2(vx, vy, vz, Bx, By, Bz):
    v2 = vx * vx + vy * vy + vz * vz
    W2 = 1.0 / torch.clamp(1.0 - v2, min=1e-14)
    vB = vx * Bx + vy * By + vz * Bz
    return (Bx * Bx + By * By + Bz * Bz) / W2 + vB * vB


def to_solver_frame(s, eos, idir):
    """(rho, P_total, v_n, v_t1, v_t2, B_t1, B_t2) plus B_n, all ``(N,)``."""
    n, t1, t2 = idir, (idir + 1) % 3, (idir + 2) % 3
    p_gas, _ = eos.press_and_cs2(s["eps"], s["rho"])
    b2 = _b2(s["vx"], s["vy"], s["vz"], s["Bx"], s["By"], s["Bz"])
    P_tot = p_gas + 0.5 * b2
    return (s["rho"], P_tot, s[_V[n]], s[_V[t1]], s[_V[t2]],
            s[_B[t1]], s[_B[t2]]), s[_B[n]]


def from_solver_frame(state, Bn, eos, idir, like):
    """Inverse of :func:`to_solver_frame`, back to a primitive dict."""
    n, t1, t2 = idir, (idir + 1) % 3, (idir + 2) % 3
    rho, P_tot, vn, vt1, vt2, Bt1, Bt2 = state
    out = {"rho": rho}
    out[_V[n]], out[_V[t1]], out[_V[t2]] = vn, vt1, vt2
    out[_B[n]], out[_B[t1]], out[_B[t2]] = Bn, Bt1, Bt2
    b2 = _b2(out["vx"], out["vy"], out["vz"], out["Bx"], out["By"], out["Bz"])
    out["p"] = torch.clamp(P_tot - 0.5 * b2, min=1e-30)
    out["eps"] = eos.eps__press_rho(out["p"], out["rho"])
    return out


def relative_jump(sL, sR):
    """Scale-free measure of how discontinuous an interface is."""
    worst = None
    for k in ("rho", "p", "vx", "vy", "vz", "Bx", "By", "Bz"):
        a, b = sL[k], sR[k]
        d = (a - b).abs() / (a.abs() + b.abs() + 1e-30)
        worst = d if worst is None else torch.maximum(worst, d)
    return worst


def exact_flux(sL, sR, eos, idir: int = 0, *, model=None, scaler=None,
               fallback=hlld_flux, tau_weak: float = 1e-6,
               accuracy: float = 1e-8, max_iter: int = 40,
               n_retries: int = 0, nsamples: int = 400):
    """Godunov flux from the exact RMHD Riemann solution at ``xi = 0``."""
    from batched import classify as C
    from batched import ray as RAY
    from batched import ray_scalar as RS

    gamma = _check_eos(eos)
    N = sL["rho"].numel()

    # HLLD everywhere first; the exact solve then overwrites the lanes it owns
    F, U, p_star = fallback(sL, sR, eos, idir=idir)
    F = {k: v.clone() for k, v in F.items()}
    U = {k: v.clone() for k, v in U.items()}
    p_star = p_star.clone()
    p_hll_fb = p_star.abs().clone()

    (L7, BnL) = to_solver_frame(sL, eos, idir)
    (R7, BnR) = to_solver_frame(sR, eos, idir)

    # sweep() forces the normal field to be single-valued at a face; if that
    # contract is broken the whole wave structure is ill-posed, so check it.
    dBn = float((BnL - BnR).abs().max())
    if dBn > 1e-10 * float(BnL.abs().max().clamp(min=1e-30)):
        raise ValueError(f"normal field is not single-valued at the face "
                         f"(max |Bn_L - Bn_R| = {dBn:.3e})")

    jump = relative_jump(sL, sR)
    v2L = sL["vx"] ** 2 + sL["vy"] ** 2 + sL["vz"] ** 2
    v2R = sR["vx"] ** 2 + sR["vy"] ** 2 + sR["vz"] ** 2
    bad = (v2L >= 1.0) | (v2R >= 1.0) | (sL["rho"] <= 0) | (sR["rho"] <= 0)
    weak = (~bad) & (jump < tau_weak)

    # ── structure classification (Level 1) ────────────────────────────────
    la_L, ra_L = C.alfven_speeds(L7[0], L7[1], L7[2], L7[3], L7[4],
                                 L7[5], L7[6], BnL, gamma)
    la_R, ra_R = C.alfven_speeds(R7[0], R7[1], R7[2], R7[3], R7[4],
                                 R7[5], R7[6], BnR, gamma)
    from wave_speeds import xi as _xi_scalar

    def _eigs(st, Bn):
        out = torch.zeros(N, 4, dtype=st[0].dtype)
        for i in range(N):
            s = [float(c[i]) for c in st]
            try:
                allv, _, _ = _xi_scalar(s, "LF", float(Bn[i]), gamma,
                                        return_all=True)
                out[i] = torch.tensor([float(v) for v in allv])
            except Exception:
                out[i] = torch.tensor([-1.0, 0.0, 0.0, 1.0], dtype=out.dtype)
        return out

    eL, eR = _eigs(L7, BnL), _eigs(R7, BnR)
    cls, _info = C.classify(L7[2], R7[2], la_L, ra_L, la_R, ra_R, eL, eR,
                            L7[5], L7[6], R7[5], R7[6])

    # Phase 1 solves only the classes fullcontact6 is defined for.  The
    # reduced 3-wave path for the degenerate classes is Phase 5; until then
    # they take the HLLD fallback, which is reported, not hidden.
    solvable = (~bad) & (~weak) & ((cls == C.FULL7) | (cls == C.COPLANAR))

    n_conv = n_ray_fan = n_unphys = 0
    idx = torch.nonzero(solvable).reshape(-1).tolist()

    if idx:
        import ml_guess as mg
        from fullcontact import fullcontact6
        if model is None or scaler is None:
            model, scaler = mg.load(os.path.join(_RMHD_ROOT,
                                                 "data/ml_guess_gamma53.pt"))

    for i in idx:
        left = [float(c[i]) for c in L7]
        right = [float(c[i]) for c in R7]
        Bn = float(BnL[i])
        try:
            if n_retries:
                res, _ = mg.predict_unk6_retry(
                    model, scaler, left, right, Bn, gamma, accuracy=accuracy,
                    max_iter=max_iter, n_retries=n_retries)
            else:
                u6 = mg.predict_unk6(model, scaler, left, right, Bn)
                res = fullcontact6(left, right, u6, Bn, gamma,
                                   accuracy=accuracy, max_iter=max_iter)
            if res is None or not res["converged"]:
                continue
            st, reg = RS.state_at_xi(left, right, res["zones"], res["VsLv"],
                                     res["VsRv"], Bn, gamma, xi_target=0.0,
                                     nsamples=nsamples)
        except Exception:
            continue

        rho, P_tot, vn, vt1, vt2, Bt1, Bt2 = st
        v2 = vn * vn + vt1 * vt1 + vt2 * vt2
        if not (rho > 0.0 and v2 < 1.0):
            n_unphys += 1
            continue
        W2 = 1.0 / (1.0 - v2)
        eta = Bn * vn + Bt1 * vt1 + Bt2 * vt2
        b2 = (Bn * Bn + Bt1 * Bt1 + Bt2 * Bt2) / W2 + eta * eta
        if P_tot - 0.5 * b2 <= 0.0:
            n_unphys += 1
            continue

        one = {k: v[i:i + 1] for k, v in sL.items()}
        star = from_solver_frame(
            [torch.tensor([c], dtype=sL["rho"].dtype) for c in st],
            torch.tensor([Bn], dtype=sL["rho"].dtype), eos, idir, one)
        # with the SAME state on both sides this returns uL == uR == U*
        # and fL == fR == F(U*)  -- the full conserved-flux dict CT needs
        uS, _, fS, _, _, _ = compute_srmhd_fluxes(star, star, eos, idir)

        for k in F:
            F[k][i] = fS[k][0]
            U[k][i] = uS[k][0]
        p_star[i] = P_tot
        n_conv += 1
        if reg in RAY.FAN_REGIONS:
            n_ray_fan += 1

    fell_back = N - n_conv
    LAST_DIAG.update(
        solver="exact",
        idir=idir,
        n_interfaces=N,
        n_bad=int(bad.sum()),
        n_weak_gate=int(weak.sum()),
        **C.histogram(cls),
        n_attempted=len(idx),
        n_exact=n_conv,
        n_fan_interior=n_ray_fan,
        n_unphysical_star=n_unphys,
        n_hlld_fallback=fell_back,
        frac_hlld_fallback=fell_back / max(N, 1),
        frac_exact=n_conv / max(N, 1),
    )
    # p_star already carries hlld_flux's negative sentinel on every lane that
    # fell back, because those lanes were never overwritten -- so
    # tests/test_idir.py's branch-flip logic works unchanged.
    return F, U, p_star
