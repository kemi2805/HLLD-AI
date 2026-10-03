"""The failure ledger: which of the interfaces production cannot solve does ANY
method we have solve?  (science plan H, step 1)

On the 64^2 production run 94.6% of the attempted interfaces are solved and
5.4% fall back to HLLD.  Before building anything to close that gap -- an ML
seed, another solution family -- this measures what closing it can possibly
mean: every failing lane of a fixed set of recorded sweeps is offered, offline
and with a generous budget, to each method this project has, and every answer
is held to PRODUCTION'S acceptance (verified against the full seven-wave
residual to 1e-8, the xi = 0 ray resolved, the state physical, the fan
ordered).  A method's own "converged" flag counts for nothing here.

    collect   replay the recordings through the production path, check that
              it reproduces them, and write the failing lanes (and a control
              sample of solved ones) in the solver frame
    run       offer one shard of that population to the methods named by
              --methods:
                a  seven-wave, 8 retries, diverse seeds (the ensemble, the
                   K-candidate checkpoint, jitter)
                b  planar, all four flip branches x six seeds
                c  the eps-ladder with its polish, and the tube-read seed
                d  planar with the slow-shock window widened: the
                   intermediate-shock branch.  NUMPY slow-shock kernel only --
                   run it in its own process with RMHD_SLOWSHOCK unset
                e  the tube test's label: elementary or compound
    --summarise   the ledger: the failures split into
                R1  recovered by a cheap method (a, b)
                R2  recovered only by an expensive search (c)
                R3  recovered only by the intermediate branch (d)
                R4  nothing found
              each against the failure reason and the compound label

The CONTROLS (lanes production solves) are there for one purpose: a method
that answers them DIFFERENTLY from production has found another root, and how
often that happens is the false-acceptance rate a rung built on it would
have.  They are reported beside every method.

Every method leaves a TRACE per interface -- each attempt, what it was
started from, how far it got and why it was refused -- and

    --report DIR   prints one card per interface: the two states and the
                   physics read off them, what production did, every attempt
                   of every method, what the tube saw, and the verdict.
                   --lanes picks interfaces, --group failing|control|all,
                   --out writes the cards to a file and a one-line-per-
                   interface table beside it (.csv)

    python scripts/failure_ledger.py collect REC_DIR pfa pfb pfc --out pop.npz
    python scripts/failure_ledger.py run pop.npz --methods a,b,c,e --out s.npz
    python scripts/failure_ledger.py --summarise DIR
    python scripts/failure_ledger.py --report DIR --out DIR/cards.txt
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time
import warnings

import numpy as np
import torch

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "scripts"))

from rmhd.eos import set_eos                                       # noqa: E402
set_eos("ideal")
from rmhd import ml_guess as mg, paths as rpaths                   # noqa: E402
from rmhd.batched import fullcontact_b as FB                       # noqa: E402
from rmhd.batched import ml_b as MB                                # noqa: E402
from rmhd.batched import planar5_b as P5                           # noqa: E402
from rmhd.batched import slow_shock_b as SSB                       # noqa: E402
from src.physics import exact_flux as EF                           # noqa: E402
from src.physics.eos import hybrid_eos                             # noqa: E402
from src.physics.hlld import LAST_DIAG                             # noqa: E402

warnings.filterwarnings("ignore")
np.seterr(all="ignore")
torch.set_num_threads(1)

GAMMA = 5.0 / 3.0
ACC, VERIFY = 1e-8, 1e-8
FLIPS = ((False, False), (True, False), (False, True), (True, True))
WIDE_MARGIN = 1.0
N_CONTROL = 1200
KCAND = "data/ml_guess_rotor_bok4.pt"
SEED3 = ("default", "sqrt(2P)", "+|BtL|", "-|BtL|", "+|BtR|", "-|BtR|")
FLIPN = ("--", "L-", "-R", "LR")
# why an answer was refused by production's acceptance (`accept`)
WHY = {0: "ACCEPTED", 1: "not converged", 2: "ray unresolved",
       3: "unphysical state", 4: "self-crossing fan", -1: "not tried"}
REASON7 = {0: "accepted", 1: "not converged", 2: "ray", 3: "unphysical",
           4: "self-crossing", 5: "degenerate refused"}
REASON5 = {0: "accepted", 1: "not converged", 2: "not planar",
           3: "fails full residual", 4: "ray", 5: "unphysical",
           6: "self-crossing", 7: "not offered"}


def sub(state, i):
    return [c[i] for c in state]


# ── production's acceptance, for an answer given as zones and speeds ──────

def accept(P, L, R, Bn, zones6, VsLv, VsRv, nrm, order=True):
    """(accepted, p_star, why): does the answer survive what production asks
    of one -- residual, ray, physical state and wave order?  ``why`` is a key
    of WHY: the FIRST test the lane failed, in production's own order."""
    n = Bn.size
    ok = np.isfinite(nrm) & (nrm <= VERIFY)
    for z in zones6:
        for c in z:
            ok &= np.isfinite(c)
    why = np.where(ok, 0, 1).astype(np.int8)
    pstar = np.full(n, np.nan)
    i = np.flatnonzero(ok)
    if i.size == 0:
        return ok, pstar, why
    Z = [[c[i] for c in z] for z in zones6]
    VL, VR = [v[i] for v in VsLv], [v[i] for v in VsRv]
    st, _, e = P["ray"].state_at_xi(sub(L, i), sub(R, i), Z, VL, VR, Bn[i],
                                    GAMMA, P["xi"], P["fan_p"], P["fan_n"])
    rho, Pt, vn, v1, v2, b1, b2 = st
    vv = vn * vn + v1 * v1 + v2 * v2
    W2 = 1.0 / np.maximum(1.0 - vv, 1e-300)
    eta = Bn[i] * vn + b1 * v1 + b2 * v2
    bb = (Bn[i] ** 2 + b1 ** 2 + b2 ** 2) / W2 + eta ** 2
    phys = (rho > 0.0) & (vv < 1.0) & (Pt - 0.5 * bb > 0.0)
    for c in st:
        phys &= np.isfinite(c)
    x = (EF._self_crossing(Z, VL, VR, EF._WAVE_ORDER_TOL) if order
         else np.zeros(i.size, bool))
    w = np.zeros(i.size, np.int8)
    w[x] = 4
    w[~phys] = 3
    w[e] = 2
    why[i] = w
    ok[i] = w == 0
    pstar[i[w == 0]] = Pt[w == 0]
    return ok, pstar, why


def first(best, new_ok, new_p, code, extra=None):
    """Keep the FIRST accepted answer per lane, and say which attempt it was."""
    take = new_ok & ~best["ok"]
    best["ok"] |= new_ok
    best["pstar"][take] = new_p[take]
    best["how"][take] = code
    if extra is not None:
        for k, v in extra.items():
            best[k][take] = v[take]
    return int(take.sum())


def blank(n):
    return dict(ok=np.zeros(n, bool), pstar=np.full(n, np.nan),
                how=np.full(n, -1, np.int16))


# ── the methods ───────────────────────────────────────────────────────────

def seeds7(L, R, Bn):
    """The diverse seven-wave seeds: every candidate of every checkpoint."""
    X = (np.stack(L, axis=1), np.stack(R, axis=1), Bn)
    out, feats = [], None
    for ck in [EF._DEFAULT_CKPT] + list(EF._EXTRA_CKPTS) + [KCAND]:
        try:
            m, s = mg.load(str(rpaths.resolve(ck)))
        except Exception as e:                       # a missing checkpoint
            print("  ! seeds: %s not loaded (%s)" % (ck, type(e).__name__))
            continue
        cands, f = MB.predict_unk6_k(m, s, *X)
        feats = f if feats is None else feats
        for u in cands:
            out.append(MB.clamp_seed_physical(u, L, R, Bn)[0])
    return out, feats


def method_a(P, F, L, R, Bn, log):
    """Seven-wave: every seed we can make, then jitter -- 8 retries."""
    sd, feats = seeds7(L, R, Bn)
    keys = MB.lane_keys(feats)
    r = MB.solve_with_retries(F["fullcontact6"], L, R, Bn, sd[0], keys,
                              accuracy=ACC, max_iter=40, n_retries=8,
                              seeds_extra=sd[1:])
    ok, p, why = accept(P, L, R, Bn, r["zones"], r["VsLv"], r["VsRv"],
                        np.where(r["converged"], r["nrm"], np.inf))
    log("a  seven-wave, %d seeds + jitter: %d/%d (converged %d, of them "
        "self-crossing %d)" % (len(sd), ok.sum(), Bn.size,
                               r["converged"].sum(), (why == 4).sum()))
    return dict(ok=ok, pstar=p, why=why, attempts=np.asarray(r["attempts"]),
                iters=np.asarray(r["n_iter_total"]),
                resid=np.asarray(r["nrm"], float),
                nseeds=np.full(Bn.size, len(sd), np.int16))


def seeds3(L0, R0):
    """Six planar seeds, in the planar frame: the default, sqrt(2P) with the
    sign of the mean field, and +-|B_t| of either side."""
    d = P5.default_seed(L0, R0)
    p0 = d[0]
    s2p = np.sqrt(2.0 * np.exp(p0)) * np.where(d[1] >= 0.0, 1.0, -1.0)
    out = [None, [p0.copy(), s2p, p0.copy()]]
    for bt in (np.abs(L0[5]), -np.abs(L0[5]), np.abs(R0[5]), -np.abs(R0[5])):
        out.append([p0.copy(), bt.copy(), p0.copy()])
    return out


def planar_sweep(P, L, R, Bn, L0, R0, log, tag, flips=FLIPS, thorough=None):
    """All flip branches x all seeds through production's own planar path.

    The trace keeps EVERY attempt: `t_why[lane, flip, seed]` is the planar
    reason code (REASON5; -1 = not tried), with the iterations and the two
    residuals beside it.  A lane in ``thorough`` is offered every
    combination even after one has succeeded -- for a failing interface the
    pattern of which starts work IS the information -- while the others stop
    at their first accepted answer.
    """
    n = Bn.size
    nf, ns = len(flips), len(SEED3)
    thorough = np.zeros(n, bool) if thorough is None else thorough
    best = blank(n)
    best["flip"] = np.full(n, -1, np.int8)
    best["seed"] = np.full(n, -1, np.int8)
    best["inter"] = np.zeros(n, bool)
    best["unk3"] = np.full((n, 3), np.nan)
    best["t_why"] = np.full((n, nf, ns), -1, np.int8)
    best["t_iter"] = np.full((n, nf, ns), -1, np.int16)
    best["t_resid"] = np.full((n, nf, ns), np.nan)
    best["t_full"] = np.full((n, nf, ns), np.nan)
    best["t_slack"] = np.full((n, nf, ns), np.nan)
    best["t_pstar"] = np.full((n, nf, ns), np.nan)
    best["t_inter"] = np.zeros((n, nf, ns), bool)
    for fi, fl in enumerate(flips):
        for si, sd in enumerate(seeds3(L0, R0)):
            i = np.flatnonzero(~best["ok"] | thorough)
            if i.size == 0:
                break
            r = EF._planar_and_ray(
                P, GAMMA, sub(L, i), sub(R, i), Bn[i], accuracy=ACC,
                max_iter=40, flip=fl,
                unk3_init=None if sd is None else [c[i] for c in sd])
            best["t_why"][i, fi, si] = r["reason"]
            best["t_iter"][i, fi, si] = r["n_iter"]
            best["t_resid"][i, fi, si] = r["resid"]
            best["t_full"][i, fi, si] = r["full_resid"]
            best["t_slack"][i, fi, si] = r["slack"]
            ok = np.zeros(n, bool); ok[i] = r["good"]
            p = np.full(n, np.nan)
            inter = np.zeros(n, bool)
            if r["star"] is not None:
                p[i] = r["star"][1]
                # a slow-family wave running ahead of its own Alfven wave
                inter[i] = ((r["VsLv"][2] < r["VsLv"][1] - 1e-9)
                            | (r["VsRv"][2] > r["VsRv"][1] + 1e-9))
                best["t_pstar"][i, fi, si] = np.where(r["good"], p[i], np.nan)
                best["t_inter"][i, fi, si] = inter[i] & r["good"]
            u3 = np.full((n, 3), np.nan)
            u3[i] = np.stack(r["unk"], axis=1)
            first(best, ok, p, 10 * fi + si,
                  dict(flip=np.full(n, fi, np.int8),
                       seed=np.full(n, si, np.int8), inter=inter, unk3=u3))
    log("%s planar, %d flips x %d seeds: %d/%d (the default seed on the "
        "unflipped branch alone: %d; intermediate %d)"
        % (tag, nf, ns, best["ok"].sum(), n,
           (best["t_why"][:, 0, 0] == 0).sum(),
           (best["ok"] & best["inter"]).sum()))
    return best


def method_c(P, F, S5, L, R, Bn, L0, R0, log):
    """The expensive searches: the eps-ladder, and the tube's own read."""
    import coplanar_limit as CLM
    from src.physics.tube_seed import read_zones, run_tubes
    n = Bn.size
    best = blank(n)
    # the ladder: tilt the tangential field out of the plane by eps, solve the
    # (regular) tilted problem, and walk eps down to zero
    best["lad_started"] = np.zeros(n, bool)
    best["lad_eps"] = np.full(n, np.nan)       # the smallest tilt reached
    best["lad_alive"] = np.zeros(n, bool)      # it reached the last rung
    best["lad_route"] = np.zeros(n, np.int8)   # polish: 0 none 1 as-is
    best["lad_why"] = np.full(n, -1, np.int8)  #   2 snapped 3 planar
    # the tube seed, as a seven-wave seed and as a planar one
    best["tube_read"] = np.zeros(n, bool)
    best["tube7_why"] = np.full(n, -1, np.int8)
    best["tube7_resid"] = np.full(n, np.nan)
    best["tube5_why"] = np.full(n, -1, np.int8)
    best["tube5_flip"] = np.full(n, -1, np.int8)
    models = CLM.load_models(EF._DEFAULT_CKPT, ",".join(EF._EXTRA_CKPTS))
    rec, alive, cur, eps_star, att = CLM.ladder(
        F, models, L0, R0, Bn, np.array(CLM.EPS_LADDER, float), 2, 60,
        lambda s: None)
    best["lad_started"] = rec.conv[:, 0].copy()
    best["lad_eps"] = np.asarray(eps_star, float)
    best["lad_alive"] = np.asarray(alive, bool)
    i = np.flatnonzero(rec.conv[:, 0])
    if i.size:
        route, unk, nrm, Z, VL, VR = CLM.polish(
            F, S5, sub(L0, i), sub(R0, i), Bn[i], sub(cur, i), 60)
        zones = [[Z[:, k, j] for j in range(7)] for k in range(6)]
        # the polish works in the planar frame; so does the acceptance here
        ok_i, p_i, w_i = accept(P, sub(L0, i), sub(R0, i), Bn[i], zones,
                                [VL[:, m] for m in range(3)],
                                [VR[:, m] for m in range(3)],
                                np.where(route > 0, nrm, np.inf))
        best["lad_route"][i] = route
        best["lad_why"][i] = w_i
        ok = np.zeros(n, bool); ok[i] = ok_i
        p = np.full(n, np.nan); p[i] = p_i
        first(best, ok, p, 100)
    n_lad = int(best["ok"].sum())

    # the tube: evolve the Riemann problem itself, read the fan, hand it over
    i = np.flatnonzero(~best["ok"])
    if i.size:
        eos = hybrid_eos(K=0.0, gamma=GAMMA, gamma_th=GAMMA)
        UL = np.stack(sub(L, i), axis=1); UR = np.stack(sub(R, i), axis=1)
        prof, t = run_tubes(UL, UR, Bn[i], eos, ncells=256, tend=0.25,
                            max_steps=20000)
        u6, okz = read_zones(prof, Bn[i], GAMMA, dx=1.0 / 256, t=t)[:2]
        best["tube_read"][i] = np.asarray(okz, bool)
        u6 = [np.nan_to_num(np.asarray(c, float)) for c in u6]
        u6, _ = MB.clamp_seed_physical(u6, sub(L, i), sub(R, i), Bn[i])
        r = F["fullcontact6"](sub(L, i), sub(R, i), u6, Bn[i], accuracy=ACC,
                              max_iter=60)
        ok_i, p_i, w_i = accept(P, sub(L, i), sub(R, i), Bn[i], r["zones"],
                                r["VsLv"], r["VsRv"],
                                np.where(r["converged"], r["nrm"], np.inf))
        best["tube7_why"][i] = w_i
        best["tube7_resid"][i] = r["nrm"]
        ok = np.zeros(n, bool); ok[i] = ok_i
        p = np.full(n, np.nan); p[i] = p_i
        first(best, ok, p, 101)
        # and the same read as a planar seed, on its own flip branch
        u3, fl = P5.unk6_to_unk3(u6, sub(L, i))
        for fi, f in enumerate(FLIPS):
            j = np.flatnonzero((fl[:, 0] == f[0]) & (fl[:, 1] == f[1])
                               & ~best["ok"][i])
            if j.size == 0:
                continue
            g = i[j]
            rp = EF._planar_and_ray(P, GAMMA, sub(L, g), sub(R, g), Bn[g],
                                    accuracy=ACC, max_iter=40, flip=f,
                                    unk3_init=[c[j] for c in u3])
            best["tube5_why"][g] = rp["reason"]
            best["tube5_flip"][g] = fi
            ok = np.zeros(n, bool); ok[g] = rp["good"]
            p = np.full(n, np.nan)
            if rp["star"] is not None:
                p[g] = rp["star"][1]
            first(best, ok, p, 102)
    log("c  eps-ladder %d, tube seed +%d: %d/%d"
        % (n_lad, best["ok"].sum() - n_lad, best["ok"].sum(), n))
    return best


def method_d(P, L, R, Bn, L0, R0, log, thorough=None):
    """The intermediate-shock branch: the slow-shock window opened."""
    if os.environ.get("RMHD_SLOWSHOCK", "") not in ("", "numpy"):
        raise SystemExit("method d needs the NUMPY slow-shock kernel: the "
                         "window patch does not reach the compiled one -- "
                         "unset RMHD_SLOWSHOCK")
    old = SSB.MARGIN
    SSB.MARGIN = WIDE_MARGIN
    try:
        best = planar_sweep(P, L, R, Bn, L0, R0, log, "d  wide-window",
                            thorough=thorough)
    finally:
        SSB.MARGIN = old
    return best


def method_e(L, R, Bn, log):
    """The tube test: what the PDE itself does with this interface."""
    import tube_features as TF
    UL = np.stack(L, axis=1); UR = np.stack(R, axis=1)
    tf, _ = TF.classify_tubes(UL, UR, Bn, 256, 0.25, log=lambda s: None)
    fk = np.asarray(tf["fkind"])
    compound = (fk == 2).any(axis=1)
    log("e  tube label: compound %d/%d" % (compound.sum(), Bn.size))
    out = dict(compound=compound)
    for k in ("nfeat", "fkind", "ffam", "fspeed", "fwidth", "fdpsi", "fdmag",
              "fbtmin", "fstr", "fnov"):
        out[k] = np.asarray(tf[k])
    return out


# ── collect ───────────────────────────────────────────────────────────────

def collect(a):
    # each recording carries the gamma it was made with (the rotor's predate
    # the field: 5/3); the stages after `collect` are the rotor's only
    eoses = {}
    rows = {k: [] for k in ("UL", "UR", "group", "tag", "sweep", "lane", "idir",
                            "t0", "reason7", "reason5", "src", "pstar",
                            "attempts", "iters7", "iters5")}
    rng = np.random.default_rng(0)
    n_att = n_ex = n_mis = 0
    n_gain = n_lost = n_both = n_moved = 0
    dp_max = 0.0
    for ti, tag in enumerate(a.tags):
        pat = re.compile(r"^sweep_%s_(\d+)\.pt$" % re.escape(tag))
        ks = sorted(int(pat.match(f).group(1)) for f in os.listdir(a.rec)
                    if pat.match(f))
        if not ks:
            raise SystemExit("no recordings sweep_%s_*.pt in %s" % (tag, a.rec))
        for k in ks:
            rec = torch.load(os.path.join(a.rec, "sweep_%s_%d.pt" % (tag, k)))
            gam = float(rec.get("gamma", GAMMA))
            eos = eoses.setdefault(gam, hybrid_eos(K=0.0, gamma=gam, gamma_th=gam))
            F, U, ps = EF.exact_flux_batched(
                rec["sL"], rec["sR"], eos, idir=rec["idir"], tau_weak=1e-2,
                tau_bt=1e-9, n_retries=2, max_iter=40)
            d = dict(LAST_DIAG)
            m_now, m_rec = d["exact_mask"].numpy(), rec["exact_mask"].numpy()
            n_mis += int((m_now != m_rec).sum())
            # paired against the recording (the switch under test is in the
            # environment of THIS run; the recording is production as it was)
            n_gain += int((m_now & ~m_rec).sum())
            n_lost += int((~m_now & m_rec).sum())
            both = m_now & m_rec
            dp = (ps.reshape(-1) - rec["p_star"].reshape(-1)).abs().numpy()[both]
            n_both += int(both.sum())
            n_moved += int((dp > 0).sum())
            dp_max = max(dp_max, float(dp.max()) if dp.size else 0.0)
            sel = np.asarray(d["sel"])
            ex = m_now[sel]
            n_att += sel.size; n_ex += int(ex.sum())
            (L7, BnL), (R7, _) = (EF.to_solver_frame(rec["sL"], eos, rec["idir"]),
                                  EF.to_solver_frame(rec["sR"], eos, rec["idir"]))
            g = lambda t: t.detach().cpu().numpy().astype(float)
            UL = np.stack([g(c) for c in L7] + [g(BnL)], axis=1)[sel]
            UR = np.stack([g(c) for c in R7] + [g(BnL)], axis=1)[sel]
            src = np.zeros(sel.size, np.int8)             # 0 fell back
            src[d["reason7"] == 0] = 1                    # seven-wave family
            src[np.asarray(d["planar5_mask"], bool)] = 2  # planar rescue
            if "planar4_mask" in d:
                src[np.asarray(d["planar4_mask"], bool)] = 3  # four-unknown
            src[~ex] = 0
            fail = np.flatnonzero(~ex)
            ctrl = np.flatnonzero(ex)
            quota = max(1, N_CONTROL // (len(a.tags) * len(ks)))
            if ctrl.size > quota:
                ctrl = np.sort(rng.choice(ctrl, quota, replace=False))
            for idx, grp in ((fail, 1), (ctrl, 0)):
                rows["UL"].append(UL[idx]); rows["UR"].append(UR[idx])
                rows["group"].append(np.full(idx.size, grp, np.int8))
                rows["tag"].append(np.full(idx.size, ti, np.int16))
                rows["sweep"].append(np.full(idx.size, k, np.int32))
                rows["lane"].append(sel[idx])
                rows["idir"].append(np.full(idx.size, rec["idir"], np.int8))
                rows["t0"].append(np.full(idx.size, float(rec.get("t0", 0.0))))
                rows["reason7"].append(np.asarray(d["reason7"])[idx])
                rows["reason5"].append(np.asarray(d["reason5"])[idx])
                rows["src"].append(src[idx])
                rows["pstar"].append(ps.reshape(-1).numpy()[sel][idx])
                rows["attempts"].append(np.asarray(d["seven_attempts"])[idx])
                rows["iters7"].append(np.asarray(d["seven_n_iter_total"])[idx])
                rows["iters5"].append(np.asarray(d["planar_n_iter"])[idx])
            print("  %s sweep %2d idir %d: attempted %4d exact %4d failed %3d"
                  % (tag, k, rec["idir"], sel.size, ex.sum(), fail.size),
                  flush=True)
    out = {k: np.concatenate(v) for k, v in rows.items()}
    out.update(tags=np.array(a.tags), n_attempted=n_att, n_exact=n_ex,
               n_mismatch=n_mis)
    np.savez_compressed(a.out, **out)
    nf = int((out["group"] == 1).sum())
    print("\ncollected %d sweeps: attempted %d, exact %d (%.2f%%), failing %d, "
          "controls %d" % (np.unique(np.stack([out["tag"], out["sweep"]]),
                                     axis=1).shape[1], n_att, n_ex,
                           100.0 * n_ex / max(n_att, 1), nf,
                           (out["group"] == 0).sum()))
    if a.paired:
        from math import comb
        nd = n_gain + n_lost
        pval = (min(1.0, 2.0 * sum(comb(nd, j) for j in
                                   range(min(n_gain, n_lost) + 1)) / 2.0 ** nd)
                if nd else 1.0)
        rec_ex = n_ex - n_gain + n_lost
        print("paired against the recording: exact %d -> %d of %d attempted "
              "(%.2f%% -> %.2f%%); gained %d, lost %d, McNemar p = %.2g"
              % (rec_ex, n_ex, n_att, 100.0 * rec_ex / max(n_att, 1),
                 100.0 * n_ex / max(n_att, 1), n_gain, n_lost, pval))
        print("  exact in both: %d lanes, star pressure moved on %d "
              "(largest change %.2e)" % (n_both, n_moved, dp_max))
        print("wrote", a.out)
        return
    print("G0: the replay differs from the recording on %d lanes -- %s"
          % (n_mis, "PASS" if n_mis == 0 else "FAIL"))
    print("wrote", a.out)
    if n_mis:
        sys.exit(1)


# ── run ───────────────────────────────────────────────────────────────────

def run(a):
    t0 = time.time()
    log = lambda s: print("[shard %d/%d] %6.0fs  %s"
                          % (a.shard, a.nshards, time.time() - t0, s), flush=True)
    Z = np.load(a.pop)
    sel = np.arange(Z["group"].size)[a.shard::a.nshards]
    if a.limit:
        sel = sel[:a.limit]
    UL, UR = Z["UL"][sel], Z["UR"][sel]
    L = [UL[:, j].copy() for j in range(7)]
    R = [UR[:, j].copy() for j in range(7)]
    Bn = UL[:, 7].copy()
    n = sel.size
    log("%d lanes (%d failing, %d controls); methods %s"
        % (n, (Z["group"][sel] == 1).sum(), (Z["group"][sel] == 0).sum(),
           a.methods))
    P = EF._batched_parts(GAMMA)
    F = FB.make_solver(GAMMA)
    S5 = P5.make_solver(GAMMA)
    L0, R0, alpha, presid = P5.to_planar(L, R, Bn)
    out = dict(idx=sel, methods=np.array(a.methods.split(",")))
    # the expensive methods are for the failures only; the controls exist to
    # catch a cheap rung answering a solved lane differently
    isf = Z["group"][sel] == 1
    f = np.flatnonzero(isf)
    Lf, Rf, Bf = sub(L, f), sub(R, f), Bn[f]
    L0f, R0f = sub(L0, f), sub(R0, f)

    def scatter(r):
        o = {}
        for k, v in r.items():
            v = np.asarray(v)
            full = (np.full((n,) + v.shape[1:], np.nan) if v.dtype.kind == "f"
                    else np.zeros((n,) + v.shape[1:], v.dtype))
            if v.dtype.kind == "i":
                full[...] = -1
            full[f] = v
            o[k] = full
        return o

    for m in a.methods.split(","):
        if m == "a":
            r = method_a(P, F, L, R, Bn, log)
        elif m == "b":
            r = planar_sweep(P, L, R, Bn, L0, R0, log, "b ", thorough=isf)
        elif m == "c":
            r = scatter(method_c(P, F, S5, Lf, Rf, Bf, L0f, R0f, log)) \
                if f.size else {}
        elif m == "d":
            r = method_d(P, L, R, Bn, L0, R0, log, thorough=isf)
        elif m == "e":
            r = scatter(method_e(Lf, Rf, Bf, log)) if f.size else {}
        else:
            raise SystemExit("unknown method %r" % m)
        for k, v in r.items():
            out["%s_%s" % (m, k)] = v
        # one line per failing interface, as it happens: the log is readable
        # while the job runs, not only after the ledger is assembled
        if "ok" in r:
            for j in f:
                extra = ""
                if m == "a":
                    extra = "%d attempts, %d iterations, residual %.1e, %s" % (
                        r["attempts"][j], r["iters"][j], r["resid"][j],
                        WHY[int(r["why"][j])])
                elif m in ("b", "d"):
                    w = r["t_why"][j]
                    extra = "%d of %d starts accepted%s" % (
                        (w == 0).sum(), (w >= 0).sum(),
                        "" if not r["ok"][j] else ", first: flip %s seed %s%s" % (
                            FLIPN[r["flip"][j]], SEED3[r["seed"][j]],
                            " (intermediate)" if r["inter"][j] else ""))
                elif m == "c":
                    extra = "ladder %s, tube seed %s" % (
                        "did not start" if not r["lad_started"][j]
                        else "reached eps %.0e" % r["lad_eps"][j],
                        WHY.get(int(r["tube7_why"][j]), "?"))
                log("  lane %5d  %s  %-8s %s" % (
                    sel[j], m, "SOLVED" if r["ok"][j] else "failed", extra))
    out["wall"] = time.time() - t0
    np.savez_compressed(a.out, **out)
    log("wrote %s" % a.out)


# ── the ledger ────────────────────────────────────────────────────────────

def load_all(d):
    """The population and every shard's columns, scattered back onto it."""
    pop = np.load(os.path.join(d, "population.npz"))
    n = pop["group"].size
    col = {}
    for f in sorted(glob.glob(os.path.join(d, "shard_*.npz"))):
        z = np.load(f)
        for k in z.files:
            if k in ("idx", "methods", "wall"):
                continue
            v = z[k]
            if k not in col:
                if v.dtype.kind == "f":
                    col[k] = np.full((n,) + v.shape[1:], np.nan)
                elif v.dtype.kind == "b":
                    col[k] = np.zeros((n,) + v.shape[1:], bool)
                else:
                    col[k] = np.full((n,) + v.shape[1:], -1, v.dtype)
                col[k + "__seen"] = np.zeros(n, bool)
            col[k][z["idx"]] = v
            col[k + "__seen"][z["idx"]] = True
    return pop, col


def verdict(pop, col):
    """R1..R4 per failing lane (0 for a control), as the ledger defines them."""
    n = pop["group"].size
    z = np.zeros(n, bool)
    ok = {m: col["%s_ok" % m].astype(bool) if "%s_ok" % m in col else z
          for m in "abcd"}
    fail = pop["group"] == 1
    cheap = ok["a"] | ok["b"]
    R = np.zeros(n, np.int8)
    R[fail] = 4
    R[fail & ok["d"]] = 3
    R[fail & ok["c"]] = 2
    R[fail & cheap] = 1
    return R, ok


def summarise(d):
    pop, col = load_all(d)
    n = pop["group"].size
    have = sorted({k.split("_")[0] for k in col if k.endswith("_ok")})
    fail, ctrl = pop["group"] == 1, pop["group"] == 0
    for m in have:
        seen = col["%s_ok__seen" % m]
        if not seen[fail].all():
            print("  ! method %s covers only %d of %d failing lanes -- the "
                  "numbers below are over an INCOMPLETE run"
                  % (m, seen[fail].sum(), fail.sum()))
    nA, nE = int(pop["n_attempted"]), int(pop["n_exact"])
    nF = int(fail.sum())
    ok = {m: col["%s_ok" % m].astype(bool) for m in have}
    print("failure ledger: %s" % d)
    print("  recordings: %d attempted, %d exact (%.2f%%), %d failing (%.2f%%); "
          "controls %d" % (nA, nE, 100.0 * nE / nA, nF, 100.0 * nF / nA,
                           ctrl.sum()))
    print("  methods run: %s\n" % ", ".join(have))

    print("  %-44s %9s %11s %14s" % ("method", "recovers", "of failures",
                                     "of attempted"))
    lab = dict(a="a  seven-wave, diverse seeds, 8 retries",
               b="b  planar, 4 flips x 6 seeds",
               c="c  eps-ladder and tube seed",
               d="d  intermediate branch (wide window)")
    for m in have:
        k = int((ok[m] & fail).sum())
        print("  %-44s %9d %10.1f%% %13.2f%%"
              % (lab.get(m, m), k, 100.0 * k / max(nF, 1), 100.0 * k / nA))

    z = np.zeros(n, bool)
    cheap = ok.get("a", z) | ok.get("b", z)
    R1 = fail & cheap
    R2 = fail & ~cheap & ok.get("c", z)
    R3 = fail & ~cheap & ~ok.get("c", z) & ok.get("d", z)
    R4 = fail & ~(R1 | R2 | R3)
    print("\n  the split of the failures:")
    for name, m in (("R1  a cheap method (a, b)", R1),
                    ("R2  only an expensive search (c)", R2),
                    ("R3  only the intermediate branch (d)", R3),
                    ("R4  nothing found", R4)):
        print("    %-40s %6d  %5.1f%% of failures  %5.2f%% of attempted"
              % (name, m.sum(), 100.0 * m.sum() / max(nF, 1),
                 100.0 * m.sum() / nA))
    print("    solved fraction if every recovery were a rung: "
          "elementary only %.2f%%, with the intermediate branch %.2f%%"
          % (100.0 * (nE + R1.sum() + R2.sum()) / nA,
             100.0 * (nE + R1.sum() + R2.sum() + R3.sum()) / nA))

    if "e_compound" in col:
        cp = col["e_compound"].astype(bool)
        print("\n  the tube's label on the failures: compound %d of %d (%.1f%%)"
              % ((cp & fail).sum(), nF, 100.0 * (cp & fail).sum() / max(nF, 1)))
        for name, m in (("R1", R1), ("R2", R2), ("R3", R3), ("R4", R4)):
            if m.any():
                print("    %s: compound %5.1f%%" % (name, 100.0 * cp[m].mean()))

    print("\n  why production lost them (seven-wave reason | planar reason):")
    for r7 in sorted(np.unique(pop["reason7"][fail])):
        for r5 in sorted(np.unique(pop["reason5"][fail])):
            m = fail & (pop["reason7"] == r7) & (pop["reason5"] == r5)
            if m.sum():
                print("    %-20s | %-20s %6d   R1 %4d  R2 %4d  R3 %4d  R4 %4d"
                      % (REASON7[int(r7)], REASON5[int(r5)], m.sum(),
                         (m & R1).sum(), (m & R2).sum(), (m & R3).sum(),
                         (m & R4).sum()))

    print("\n  controls -- lanes production solves; a different answer is "
          "another root:")
    for m in have:
        if "%s_pstar" % m not in col:
            continue
        p = col["%s_pstar" % m]
        both = ctrl & ok[m] & np.isfinite(p)
        rel = np.abs(p[both] - pop["pstar"][both]) / np.abs(pop["pstar"][both])
        if both.any():
            print("    %-40s answers %4d of %4d; p* differs by > 1e-3 on %d "
                  "(%.1f%%), median %.1e"
                  % (lab.get(m, m)[:40], both.sum(), ctrl.sum(),
                     (rel > 1e-3).sum(), 100.0 * (rel > 1e-3).mean(),
                     np.median(rel)))


# ── the report: one card per interface ────────────────────────────────────

# OK accepted | nc the reduced Newton did not converge | np the input is not
# planar | vf converged, but fails the FULL residual | ray | unph unphysical |
# cross self-crossing fan | . not tried
CODE5 = {0: "OK", 1: "nc", 2: "np", 3: "vf", 4: "ray", 5: "unph", 6: "cross",
         7: ".", -1: "."}
KINDS = {0: "magnetosonic jump", 1: "rotation", 2: "REVERSAL FUSED TO A JUMP",
         3: "mixed rotation", -1: "undecidable", -2: ""}
FAMS = ("F-", "A-", "S-", "CD", "S+", "A+", "F+")
ROUTE = {0: "nothing", 1: "the limit as it is", 2: "the limit, angles snapped",
         3: "the planar solver from the limit"}


def physics(UL, UR):
    """What can be read off the two states before any solve."""
    from rmhd.batched import api as API
    L = [UL[:, j].copy() for j in range(7)]
    R = [UR[:, j].copy() for j in range(7)]
    Bn = UL[:, 7].copy()
    out = {}
    for tag, U in (("L", L), ("R", R)):
        rho, Pt, vx, vy, vz, By, Bz = U
        v2 = vx * vx + vy * vy + vz * vz
        W = 1.0 / np.sqrt(np.maximum(1.0 - v2, 1e-300))
        vB = Bn * vx + By * vy + Bz * vz
        b2 = (Bn * Bn + By * By + Bz * Bz) / (W * W) + vB * vB
        pg = Pt - 0.5 * b2
        h = 1.0 + GAMMA / (GAMMA - 1.0) * pg / rho
        out["W_" + tag] = W
        out["p_" + tag] = pg
        out["sigma_" + tag] = b2 / (rho * h)
        out["beta_" + tag] = 2.0 * pg / np.maximum(b2, 1e-300)
        out["Bt_" + tag] = np.hypot(By, Bz)
    psiL = np.arctan2(L[6], L[5]); psiR = np.arctan2(R[6], R[5])
    out["dpsi"] = np.abs((psiR - psiL + np.pi) % (2 * np.pi) - np.pi)
    out["Bn"] = Bn
    out["ratio"] = np.minimum(out["Bt_L"], out["Bt_R"]) / np.maximum(np.abs(Bn), 1e-300)
    out["rule"] = (out["ratio"] < 0.3) & (out["dpsi"] > 0.5 * np.pi)
    out["planarity"] = P5.to_planar(L, R, Bn)[3]
    cls, info = API.classify_batch(L, R, Bn, GAMMA)
    out["cls"] = np.asarray(cls)
    for k in ("d_contact", "d_slow", "d_fast"):
        out[k] = np.asarray(info[k], float)
    return out


def grid(why, inter=None):
    """The 4 x 6 table of planar starts as text."""
    rows = ["         (OK accepted, nc not converged, vf converged but fails the "
            "full residual, np not planar,",
            "          ray / unph / cross refused by the ray, the state or the "
            "wave order; flip = a pi rotation on L or R)",
            "            seed:  " + "".join("%-9s" % s_ for s_ in SEED3)]
    for fi in range(why.shape[0]):
        cells = []
        for si in range(why.shape[1]):
            c = CODE5[int(why[fi, si])]
            if inter is not None and why[fi, si] == 0 and inter[fi, si]:
                c = "OK*"
            cells.append("%-9s" % c)
        rows.append("        flip %s      %s" % (FLIPN[fi], "".join(cells).rstrip()))
    return rows


def card(j, pop, col, ph, R, ok, tags):
    g = lambda k: col[k][j] if k in col else None
    seen = lambda m: ("%s_ok__seen" % m) in col and col["%s_ok__seen" % m][j]
    fail = pop["group"][j] == 1
    k = int(pop["sweep"][j])
    out = []
    out.append("=" * 100)
    out.append("interface %d   %s   window %s (pre-evolved to t = %.3f), sweep %d "
               "= step %d, RK stage %d, %s-faces, lane %d"
               % (j, "FAILING in production" if fail else
                  "control (production solves it: %s)"
                  % {1: "seven-wave", 2: "planar rescue"}.get(int(pop["src"][j]), "?"),
                  tags[int(pop["tag"][j])], float(pop["t0"][j]), k, (k - 1) // 6,
                  ((k - 1) % 6) // 2, "xy"[int(pop["idir"][j])], int(pop["lane"][j])))
    out.append("")
    out.append("  the two states, solver frame (x = the face normal)      B_n = %+.6f"
               % ph["Bn"][j])
    out.append("       %11s %11s %11s %11s %11s %11s %11s %9s"
               % ("rho", "P_tot", "vx", "vy", "vz", "By", "Bz", "p_gas"))
    for tag, U in (("L", pop["UL"][j]), ("R", pop["UR"][j])):
        out.append("    %s  " % tag + " ".join("%11.6f" % v for v in U[:7])
                   + " %9.5f" % ph["p_" + tag][j])
    out.append("")
    out.append("  what the states say")
    out.append("    magnetisation sigma   L %-9.3g R %-9.3g   plasma beta  L %-9.3g R %-9.3g"
               % (ph["sigma_L"][j], ph["sigma_R"][j], ph["beta_L"][j], ph["beta_R"][j]))
    out.append("    Lorentz factor        L %-9.4f R %-9.4f" % (ph["W_L"][j], ph["W_R"][j]))
    out.append("    tangential field |Bt| L %-9.4g R %-9.4g   weaker side / |B_n| = %.3g"
               % (ph["Bt_L"][j], ph["Bt_R"][j], ph["ratio"][j]))
    out.append("    the tangential field turns by %.4f rad (%s); out of plane by %.1e"
               % (ph["dpsi"][j], "REVERSES" if ph["dpsi"][j] > 0.5 * np.pi
                  else "same sense", ph["planarity"][j]))
    from rmhd.batched import classify as CL
    out.append("    structure class %s; speed gaps as a fraction of the fan: "
               "contact %.2e, slow %.2e, fast %.2e"
               % (CL.CLASS_NAMES.get(int(ph["cls"][j]), "?"), ph["d_contact"][j],
                  ph["d_slow"][j], ph["d_fast"][j]))
    out.append("    the compound rule (weak field that reverses): %s"
               % ("YES" if ph["rule"][j] else "no"))
    out.append("")
    out.append("  production")
    out.append("    seven-wave Newton : %s -- %d attempt(s), %d iterations in all"
               % (REASON7[int(pop["reason7"][j])].upper()
                  if pop["reason7"][j] else "accepted",
                  int(pop["attempts"][j]), int(pop["iters7"][j])))
    r5 = int(pop["reason5"][j])
    out.append("    planar rescue     : %s%s"
               % (REASON5[r5].upper() if r5 else "accepted",
                  "" if pop["iters5"][j] < 0 else " -- %d iterations" % int(pop["iters5"][j])))
    out.append("    => %s" % ("HLLD fallback" if fail else "exact, p* = %.8f" % pop["pstar"][j]))
    out.append("")
    out.append("  the ledger")
    if seen("a"):
        out.append("    a  seven-wave, %d ML seeds then jitter, 8 retries" % int(g("a_nseeds")))
        out.append("         %s -- %d attempts, %d iterations, best residual %.2e%s"
                   % (WHY[int(g("a_why"))], int(g("a_attempts")), int(g("a_iters")),
                      float(g("a_resid")),
                      ", p* = %.8f" % g("a_pstar") if g("a_ok") else ""))
    for m, title in (("b", "b  planar, every flip branch x every seed"),
                     ("d", "d  the same with the slow-shock window opened "
                           "(the intermediate branch)")):
        if not seen(m):
            continue
        w = g(m + "_t_why")
        out.append("    %s" % title)
        out.extend(grid(w, g(m + "_t_inter")))
        n_ok, n_try = int((w == 0).sum()), int((w >= 0).sum())
        if n_ok:
            ps = g(m + "_t_pstar")[w == 0]
            out.append("         %d of %d starts accepted; p* between %.8f and %.8f%s"
                       % (n_ok, n_try, ps.min(), ps.max(),
                          "  <-- MORE THAN ONE ANSWER" if
                          (ps.max() - ps.min()) > 1e-6 * abs(ps.mean()) else ""))
            if m == "d" and g("d_t_inter")[w == 0].any():
                out.append("         OK* = the accepted answer has a slow-family wave "
                           "AHEAD of its Alfven wave: an intermediate shock")
        else:
            it = g(m + "_t_iter")[w >= 0]
            rr = g(m + "_t_resid")[w >= 0]
            out.append("         none of %d starts accepted; best reduced residual "
                       "%.2e, iterations %d-%d"
                       % (n_try, np.nanmin(rr) if rr.size else np.nan,
                          it.min() if it.size else 0, it.max() if it.size else 0))
        vf = w == 3
        if vf.any():
            sl = g(m + "_t_slack")
            out.append("         vf: %d starts SOLVE the three planar equations "
                       "(residual <= %.1e) and still fail the full system by "
                       "%.2e;" % (vf.sum(), np.nanmax(g(m + "_t_resid")[vf]),
                                  np.nanmin(g(m + "_t_full")[vf])))
            out.append("             slow-wave slack %.2e -- %s"
                       % (np.nanmin(sl[vf]) if sl is not None else np.nan,
                          "the slow wave cannot reach the tangential field it "
                          "was asked for" if sl is not None
                          and np.nanmin(sl[vf]) > 1e-6 else
                          "the slack is small: the mismatch is elsewhere"))
    if seen("c"):
        out.append("    c  the expensive searches")
        if g("c_lad_started"):
            out.append("         eps-ladder: started at a tilt of 0.1 rad, walked down "
                       "to %.0e%s; the polish took %s -> %s"
                       % (float(g("c_lad_eps")),
                          " (the last rung)" if g("c_lad_alive") else " and lost it there",
                          ROUTE[int(g("c_lad_route"))],
                          WHY.get(int(g("c_lad_why")), "?")))
        else:
            out.append("         eps-ladder: did not start (the tilted problem "
                       "itself was not solved)")
        if g("c_tube7_why") >= 0:
            out.append("         tube seed (256 cells): %s; as a seven-wave seed %s "
                       "(residual %.2e); as a planar seed on flip %s: %s"
                       % ("fan read cleanly" if g("c_tube_read") else
                          "fan NOT read cleanly", WHY[int(g("c_tube7_why"))],
                          float(g("c_tube7_resid")),
                          FLIPN[int(g("c_tube5_flip"))] if g("c_tube5_flip") >= 0 else "-",
                          REASON5.get(int(g("c_tube5_why")), "not tried")))
    if "e_compound__seen" in col and col["e_compound__seen"][j] and fail:
        out.append("    e  the tube test: the PDE itself, 256 and 512 cells")
        fk, ff = g("e_fkind"), g("e_ffam")
        any_ = False
        for q in range(fk.size):
            if fk[q] < -1:
                continue
            any_ = True
            out.append("         feature at x/t = %+.4f (width %.4f), family %-3s: %s"
                       "  [turn %.3f rad, |Bt| jump %.3g, |Bt| minimum %.3g of its "
                       "neighbours]"
                       % (g("e_fspeed")[q], g("e_fwidth")[q],
                          FAMS[int(ff[q])] if ff[q] >= 0 else "?",
                          KINDS.get(int(fk[q]), "?"), abs(g("e_fdpsi")[q]),
                          g("e_fdmag")[q], g("e_fbtmin")[q]))
        if not any_:
            out.append("         no feature confirmed at both resolutions")
        out.append("         => %s" % ("COMPOUND: not an elementary-wave solution"
                                       if g("e_compound") else "elementary"))
    out.append("")
    if fail:
        out.append("  VERDICT: %s" % {
            1: "R1 -- a cheap method finds an exact answer production misses",
            2: "R2 -- only an expensive search finds one",
            3: "R3 -- only the intermediate branch answers it",
            4: "R4 -- nothing we have solves this interface"}[int(R[j])])
    else:
        diff = []
        for m in "abd":
            k2 = m + "_pstar"
            if k2 in col and ok[m][j] and np.isfinite(col[k2][j]):
                rel = abs(col[k2][j] - pop["pstar"][j]) / abs(pop["pstar"][j])
                if rel > 1e-3:
                    diff.append("%s (p* off by %.1e)" % (m, rel))
        out.append("  CONTROL: %s" % ("every method that answers agrees with production"
                                     if not diff else
                                     "ANOTHER ROOT from " + ", ".join(diff)))
    out.append("")
    return out


def report(a):
    pop, col = load_all(a.report)
    tags = [str(t) for t in pop["tags"]]
    ph = physics(pop["UL"], pop["UR"])
    R, ok = verdict(pop, col)
    n = pop["group"].size
    pick = np.arange(n)
    if a.lanes:
        pick = np.array([int(v) for v in a.lanes.split(",")])
    elif a.group == "failing":
        pick = np.flatnonzero(pop["group"] == 1)
    elif a.group == "control":
        pick = np.flatnonzero(pop["group"] == 0)
    lines = []
    for j in pick:
        lines.extend(card(int(j), pop, col, ph, R, ok, tags))
    text = "\n".join(lines)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text + "\n")
        import csv
        tab = os.path.splitext(a.out)[0] + ".csv"
        with open(tab, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["interface", "group", "window", "sweep", "dir", "lane",
                        "verdict", "reason7", "reason5", "attempts7", "iters7",
                        "Bn", "sigma_max", "W_max", "Bt_min_over_Bn", "dpsi",
                        "compound_rule", "tube_compound", "class", "d_contact",
                        "d_slow", "a", "b", "b_starts_ok", "c", "d",
                        "d_starts_ok", "d_intermediate"])
            for j in pick:
                c = lambda k, dflt="": (col[k][j] if k in col else dflt)
                w.writerow([
                    j, "failing" if pop["group"][j] else "control",
                    tags[int(pop["tag"][j])], int(pop["sweep"][j]),
                    "xy"[int(pop["idir"][j])], int(pop["lane"][j]),
                    "R%d" % R[j] if pop["group"][j] else "",
                    REASON7[int(pop["reason7"][j])], REASON5[int(pop["reason5"][j])],
                    int(pop["attempts"][j]), int(pop["iters7"][j]),
                    "%.6g" % ph["Bn"][j],
                    "%.4g" % max(ph["sigma_L"][j], ph["sigma_R"][j]),
                    "%.4g" % max(ph["W_L"][j], ph["W_R"][j]),
                    "%.4g" % ph["ratio"][j], "%.4f" % ph["dpsi"][j],
                    int(ph["rule"][j]), int(bool(c("e_compound", 0))),
                    int(ph["cls"][j]), "%.3e" % ph["d_contact"][j],
                    "%.3e" % ph["d_slow"][j], int(ok["a"][j]), int(ok["b"][j]),
                    int((c("b_t_why", np.array(-1)) == 0).sum()),
                    int(ok["c"][j]), int(ok["d"][j]),
                    int((c("d_t_why", np.array(-1)) == 0).sum()),
                    int(bool(np.any(c("d_t_inter", np.array(False)))))])
        print("wrote %s (%d cards) and %s" % (a.out, pick.size, tab))
    else:
        print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", nargs="?", choices=("collect", "run"))
    ap.add_argument("args", nargs="*")
    ap.add_argument("--out")
    ap.add_argument("--methods", default="a,b,c,e")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--paired", action="store_true",
                    help="collect: a switch is under test -- report gained / "
                         "lost against the recording instead of G0")
    ap.add_argument("--summarise", metavar="DIR")
    ap.add_argument("--report", metavar="DIR")
    ap.add_argument("--lanes", help="comma-separated interface numbers")
    ap.add_argument("--group", default="failing",
                    choices=("failing", "control", "all"))
    a = ap.parse_args()
    if a.summarise:
        return summarise(a.summarise)
    if a.report:
        return report(a)
    if a.mode == "collect":
        a.rec, a.tags = a.args[0], a.args[1:]
        return collect(a)
    if a.mode == "run":
        a.pop = a.args[0]
        return run(a)
    ap.error("collect, run or --summarise")


if __name__ == "__main__":
    main()
