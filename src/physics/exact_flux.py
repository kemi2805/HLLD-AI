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

# Overridable: the cluster checkouts live elsewhere (Goethe:
# /work/astro/miler/codes/rmhd_final), and a git worktree of rmhd_final is how
# a bit-identity reference is isolated from edits on main.
_RMHD_ROOT = os.environ.get("RMHD_ROOT", "/Users/miler/Codes/rmhd_final")

# Which warm-start checkpoint to auto-load.  Override with RMHD_ML_CKPT.
#
# Default is v5 because it is the measured best on the SHOCK-TUBE regime,
# which is what the 1D gate exercises: 90.7% converged with the 4x retry
# ladder, against 89.0% for ml_guess_gamma53.pt (the previous hardcoded
# default) on the same 300-problem eval set.
#
# It is deliberately NOT the right choice for the rotor.  The rotor-trained
# checkpoints win there and lose here, and the gap is significant in both
# directions (paired McNemar, evalset_rotor_600 / evalset_300):
#
#     checkpoint                  rotor    tubes
#     ml_guess_gamma53_v5.pt      83.8%    90.7%
#     ml_guess_rotor_ft.pt        88.8%    84.0%
#     ml_guess_rotor_scratch.pt   87.2%    82.3%
#
# So for rotor production runs, set
#     RMHD_ML_CKPT=data/ml_guess_rotor_ft.pt
# rather than changing this default.  Which one to promote outright is an
# open decision -- ft and scratch are statistically indistinguishable from
# each other on both regimes.
#
# The checkpoint only affects convergence RATE and speed, never correctness:
# interfaces the Newton cannot resolve fall back to HLLD and are counted in
# LAST_DIAG.
_DEFAULT_CKPT = os.environ.get("RMHD_ML_CKPT", "data/ml_guess_gamma53_v5.pt")
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
    """Scale-free, rotation-invariant measure of how discontinuous an interface is.

    The obvious form -- max over every variable of |a-b|/(|a|+|b|) -- is wrong
    for any quantity that changes SIGN.  The denominator vanishes at a zero
    crossing, so the ratio saturates at 1 no matter how smooth the data is.
    That is not a corner case here: the rotor is a rotating flow, so vx, vy and
    By cross zero across most of the domain.  Measured on a 128^2 HLLD rotor at
    t=0.4, the old form called 99.9% of interfaces discontinuous at a 1e-2
    threshold, while rho and p -- the only strictly positive variables -- put
    just 3.2% and 7.9% above 0.1.  The weak-jump gate could therefore never
    fire, and every interface went to the exact solver.

    Three fixes, one per kind of variable:

    * rho and p are strictly positive, so the ordinary relative difference is
      well behaved and is kept.
    * velocity is compared as an absolute VECTOR jump.  v is in units of c and
      bounded by 1, so |v_L - v_R| is already dimensionless and scale-free, and
      it cannot blow up where a component passes through zero.
    * the field is compared as a vector jump against a scale floored by the
      pressure scale, so a component crossing zero -- or |B| itself vanishing --
      cannot empty the denominator.

    Using vector norms rather than a per-component max also makes the measure
    rotation-invariant, which the old form was not: the same physical interface
    could gate differently in an x-sweep than in a y-sweep, which would quietly
    break `test_idir` and the rotor's pi-rotation symmetry diagnostic.
    """
    tiny = 1e-30
    worst = (sL["rho"] - sR["rho"]).abs() / (
        sL["rho"].abs() + sR["rho"].abs() + tiny)
    worst = torch.maximum(worst, (sL["p"] - sR["p"]).abs() / (
        sL["p"].abs() + sR["p"].abs() + tiny))

    dv = sum((sL[k] - sR[k]) ** 2 for k in ("vx", "vy", "vz")).sqrt()
    worst = torch.maximum(worst, dv)

    dB = sum((sL[k] - sR[k]) ** 2 for k in ("Bx", "By", "Bz")).sqrt()
    BL = sum(sL[k] ** 2 for k in ("Bx", "By", "Bz")).sqrt()
    BR = sum(sR[k] ** 2 for k in ("Bx", "By", "Bz")).sqrt()
    scale = BL + BR + (sL["p"].abs() + sR["p"].abs()).sqrt()
    return torch.maximum(worst, dB / (scale + tiny))


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
    solved_idx: list[int] = []
    idx = torch.nonzero(solvable).reshape(-1).tolist()

    if idx:
        import ml_guess as mg
        from fullcontact import fullcontact6
        if model is None or scaler is None:
            model, scaler = mg.load(os.path.join(_RMHD_ROOT, _DEFAULT_CKPT))

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
        solved_idx.append(i)
        n_conv += 1
        if reg in RAY.FAN_REGIONS:
            n_ray_fan += 1

    fell_back = N - n_conv
    exact_mask = torch.zeros(N, dtype=torch.bool)
    if solved_idx:
        exact_mask[torch.tensor(solved_idx, dtype=torch.long)] = True
    LAST_DIAG.update(
        exact_mask=exact_mask,
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


# ===========================================================================
# Batched path -- the one a production run uses
# ===========================================================================

_SOLVER_CACHE: dict = {}


def _batched_parts(gamma):
    """Build (and cache) the batched solvers and ray for one ``gamma``.

    Cached because ``make_solver`` source-recompiles the scalar kernels, which
    is not something to repeat per sweep.
    """
    if gamma in _SOLVER_CACHE:
        return _SOLVER_CACHE[gamma]
    from batched import api as API
    from batched import rarefaction_b as RB
    from batched import ray_b as RAY
    from batched import wave_speeds_b as WB
    import numpy as np

    idx = {"LF": 0, "LS": 1, "RS": 2, "RF": 3}

    def xi_fn(state, switch, Bn, g=gamma):
        # ok is per-ROOT, (N, 4), aligned with eig's [LF, LS, RS, RF].  Ask it
        # about the root being requested and no other: near B_n = 0 the
        # interior pair merges into a double root that double precision cannot
        # resolve, while the fast pair stays exact to ~1e-17.  Gating LF/RF on
        # the interior pair's failure refuses rarefactions that are perfectly
        # well determined -- and B_n = By IS the rotor's y-sweep, zero at t=0
        # and small for a long while after.  See rmhd_final commit 2392aa6.
        k = idx[switch]
        eig, _, _, ok = WB.xi_all(*state, Bn, g)
        return np.where(ok[:, k], eig[:, k], np.nan)

    fan_p, fan_n = RB.make_integrators(gamma, lambda s, sw, B, g:
                                       xi_fn(s, sw, B, g))
    parts = dict(solve=API.make_solver(gamma), ray=RAY, xi=xi_fn,
                 fan_p=fan_p, fan_n=fan_n, np=np)
    _SOLVER_CACHE[gamma] = parts
    return parts


def exact_flux_batched(sL, sR, eos, idir: int = 0, *, model=None, scaler=None,
                       fallback=hlld_flux, tau_weak: float = 1e-6,
                       tau_bt: float = 1e-9,
                       accuracy: float = 1e-8, max_iter: int = 40,
                       n_retries: int = 0, harvester=None):
    """Godunov flux from the exact solution at ``xi = 0``, batched.

    Same contract as :func:`exact_flux` and as ``hlld_flux``, so ``sweep()``
    needs only the name.  Every interface of a sweep is solved in one call.

    The structure of the routine is: HLLD everywhere first, then overwrite the
    lanes the exact solver actually owns.  That ordering is deliberate -- a
    lane that fails ANY gate keeps a valid flux rather than a hole, and the
    negative ``p_star`` sentinel ``hlld_flux`` wrote survives on exactly those
    lanes, which is what ``tests/test_idir.py``'s branch-flip logic reads.
    """
    P = _batched_parts(_check_eos(eos))
    np = P["np"]
    RAY = P["ray"]
    gamma = _check_eos(eos)
    N = sL["rho"].numel()
    dt = sL["rho"].dtype

    F, U, p_star = fallback(sL, sR, eos, idir=idir)
    F = {k: v.clone() for k, v in F.items()}
    U = {k: v.clone() for k, v in U.items()}
    p_star = p_star.clone()

    (L7, BnL), (R7, BnR) = to_solver_frame(sL, eos, idir), to_solver_frame(sR, eos, idir)
    dBn = float((BnL - BnR).abs().max())
    if dBn > 1e-10 * float(BnL.abs().max().clamp(min=1e-30)):
        raise ValueError(f"normal field is not single-valued at the face "
                         f"(max |Bn_L - Bn_R| = {dBn:.3e})")

    to_np = lambda t: t.detach().cpu().numpy().astype(float)
    left = [to_np(c) for c in L7]
    right = [to_np(c) for c in R7]
    Bn = to_np(BnL)

    jump = relative_jump(sL, sR)
    v2L = sL["vx"] ** 2 + sL["vy"] ** 2 + sL["vz"] ** 2
    v2R = sR["vx"] ** 2 + sR["vy"] ** 2 + sR["vz"] ** 2
    bad = (v2L >= 1.0) | (v2R >= 1.0) | (sL["rho"] <= 0) | (sR["rho"] <= 0)
    weak = (~bad) & (jump < tau_weak)
    live = to_np((~bad) & (~weak)).astype(bool)
    sel = np.flatnonzero(live)

    diag = dict(solver="exact-batched", idir=idir, n_interfaces=N,
                n_bad=int(bad.sum()), n_weak_gate=int(weak.sum()))

    if sel.size == 0:
        LAST_DIAG.update(**diag, n_attempted=0, n_exact=0,
                         n_hlld_fallback=N, frac_hlld_fallback=1.0,
                         frac_exact=0.0)
        return F, U, p_star

    subL = [c[sel] for c in left]
    subR = [c[sel] for c in right]
    subBn = Bn[sel]

    # The ML warm start is not optional here.  Without it the Newton starts
    # from a structure-free guess, and the difference is not marginal: on the
    # same 40 interfaces, neutral seeding solved 23 and took 1896 s where the
    # warm start solved 30 in 383 s.  The scalar path auto-loads for the same
    # reason, so the batched one must too or the comparison is meaningless.
    import ml_guess as mg
    if model is None or scaler is None:
        model, scaler = mg.load(os.path.join(_RMHD_ROOT, _DEFAULT_CKPT))
    from batched import ml_b as MB
    seed6, feats = MB.predict_unk6(
        model, scaler, np.stack(subL, axis=1), np.stack(subR, axis=1), subBn)
    # The net is an interpolator; outside its training box it extrapolates
    # without limit.  Measured on the t=0 rotor x-sweep, where |Bt| is exactly
    # zero on both sides: it is fed log(0 + 1e-30) = -69 against a training
    # minimum of -6.9 and answers ln|Bt_CD| ~ 110, seeding the Newton at 1e48
    # so its first residual is inf.  Bounding the seed does not rescue those
    # lanes -- they fall back to HLLD and are counted -- it keeps a nonsense
    # seed from putting non-finite numbers into a shared batch.
    seed6, n_seed_clamped = MB.clamp_seed_physical(seed6, subL, subR, subBn)
    # per-lane RNG keys folded from each lane's own canonical features, so a
    # retry jitter cannot depend on where the interface sits in the batch
    keys = MB.lane_keys(feats) if n_retries else None

    # tau_bt is applied INSIDE solve_batch rather than here: it must only
    # skip the seven-wave classes.  The reduced three-wave solver carries no
    # rotation unknowns and handles Bt = 0 correctly -- 8 of the rotor's 12
    # degenerate x-sweep lanes converge on it at t=0 -- so a tier-0 gate would
    # have thrown those answers away along with the unrepresentable ones.
    res, d = P["solve"](subL, subR, subBn, seed6=seed6, accuracy=accuracy,
                        max_iter=max_iter, n_retries=n_retries, keys=keys,
                        tau_bt=tau_bt)
    diag.update(d)
    diag["n_seed_clamped"] = n_seed_clamped

    from batched import classify as C
    cls = res["cls"]
    conv = res["converged"]

    # ── xi = 0, by structure class ────────────────────────────────────────
    star = [np.zeros(sel.size) for _ in range(7)]
    ray_ok = np.zeros(sel.size, dtype=bool)
    region = np.full(sel.size, -1, dtype=int)
    n_fan = 0

    m7 = np.flatnonzero(conv & np.isin(cls, (C.FULL7, C.COPLANAR)))
    if m7.size:
        st, reg, e = RAY.state_at_xi(
            [c[m7] for c in subL], [c[m7] for c in subR],
            [[z[j][m7] for j in range(7)] for z in res["zones"]],
            [v[m7] for v in res["VsLv"]], [v[m7] for v in res["VsRv"]],
            subBn[m7], gamma, P["xi"], P["fan_p"], P["fan_n"])
        for j in range(7):
            star[j][m7] = st[j]
        region[m7] = reg
        ray_ok[m7] = ~e
        n_fan += int(np.isin(reg, RAY.FAN_REGIONS).sum())

    m3 = np.flatnonzero(conv & ~np.isin(cls, (C.FULL7, C.COPLANAR)))
    if m3.size:
        st, reg, e = RAY.state_at_xi_3wave(
            [c[m3] for c in subL], [c[m3] for c in subR],
            [z[m3] for z in res["zones"][2]], [z[m3] for z in res["zones"][3]],
            res["VsL"][m3], res["VsR"][m3], subBn[m3], gamma, P["xi"],
            P["fan_p"])
        for j in range(7):
            star[j][m3] = st[j]
        region[m3] = reg
        ray_ok[m3] = ~e
        n_fan += int(np.isin(reg, RAY.FAN_REGIONS_3).sum())

    # ── the resolved state must itself be physical ────────────────────────
    rho, P_tot, vn, vt1, vt2, Bt1, Bt2 = star
    v2 = vn * vn + vt1 * vt1 + vt2 * vt2
    W2 = 1.0 / np.maximum(1.0 - v2, 1e-300)
    eta = subBn * vn + Bt1 * vt1 + Bt2 * vt2
    b2 = (subBn ** 2 + Bt1 ** 2 + Bt2 ** 2) / W2 + eta ** 2
    physical = (rho > 0.0) & (v2 < 1.0) & (P_tot - 0.5 * b2 > 0.0)
    for c in star:
        physical &= np.isfinite(c)
    take = conv & ray_ok & physical

    n_unphys = int((conv & ray_ok & ~physical).sum())

    # Harvest before the flux assembly, while the per-lane verdict is still
    # in hand: `take` is the only place that knows a lane cleared EVERY gate
    # (converged, ray resolved, state physical), and downstream only the
    # assembled flux survives.
    if harvester is not None:
        from src.physics.harvest import Harvester           # noqa: F401
        harvester.record(
            subL, subR, subBn, cls=cls, converged=conv, zones=res["zones"],
            VsLv=res["VsLv"], VsRv=res["VsRv"],
            attempts=res.get("attempts", np.ones(sel.size, dtype=int)),
            accepted=take,
            seven_wave=np.isin(cls, (C.FULL7, C.COPLANAR)))

    g = sel[take]
    if g.size:
        tt = lambda a: torch.tensor(a[take], dtype=dt)
        one = {k: v[g] for k, v in sL.items()}
        star_t = from_solver_frame([tt(c) for c in star], tt(subBn),
                                   eos, idir, one)
        uS, _, fS, _, _, _ = compute_srmhd_fluxes(star_t, star_t, eos, idir)
        for k in F:
            F[k][g] = fS[k]
            U[k][g] = uS[k]
        p_star[g] = torch.tensor(P_tot[take], dtype=dt)

    n_exact = int(g.size)
    fell_back = N - n_exact
    exact_mask = torch.zeros(N, dtype=torch.bool)
    if g.size:
        exact_mask[torch.as_tensor(g, dtype=torch.long)] = True
    LAST_DIAG.update(
        **diag,
        exact_mask=exact_mask,
        n_attempted=int(sel.size),
        n_exact=n_exact,
        n_fan_interior=n_fan,
        n_unphysical_star=n_unphys,
        n_ray_failed=int((conv & ~ray_ok).sum()),
        n_hlld_fallback=fell_back,
        frac_hlld_fallback=fell_back / max(N, 1),
        frac_exact=n_exact / max(N, 1),
    )
    return F, U, p_star
