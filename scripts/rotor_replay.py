"""Paired comparison of two solver configurations on IDENTICAL inputs.

    record <tag>          run NSTEP (default 1) RK3 steps of the 32^2 rotor
                          and save every sweep's interface inputs and outputs
                          to /tmp/sweep_<tag>_<k>.pt
    replay <any> <tag>    feed the SAME inputs to exact_flux_batched under
                          the current RMHD_* kernel selection and diff per
                          interface: convergence flips, flux and p* agreement
                          on the commonly solved lanes.  RETRIES=<n> sets the
                          retry ladder, REPLAY_OUT=<npz> saves the per-sweep
                          convergence masks for `compare`
    compare <A.npz> <B.npz>
                          paired per-lane convergence of two replays (McNemar)

Why: comparing two full runs conflates "different answer on the same input"
with "different input because an earlier sweep differed".  Recording the
sweeps separates the two.  Measured 2026-09-06 (four compiled kernels vs
numpy fan + Alfven): five of six sweeps identical to machine precision, two
marginal lanes in one x-sweep -- what a whole-run comparison had shown as
5e-4 field differences and 128 vs 131 solved.

Env: RMHD_ML_CKPT, RMHD_FAN/ALFVEN/SLOWSHOCK/SHOCK as for a run.
     record: N (32) the grid; T0 (0) pre-evolve with HLLD to this time, then
             record NSTEP exact steps -- sweeps from the developed flow
             instead of the initial disc; REC_DIR (/tmp) where they go;
             PROBLEM (rotor) or orszag_tang -- box, gamma and boundaries as
             in run_2d.py; each recording carries its problem and gamma,
             and replay solves it with that gamma.
     replay: REC_DIR; WARMUP=k replays the first k sweeps silently first, so
             first-call compilation is not billed to sweep 1."""
import sys, os, re, time, functools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ^ the repo root, derived from this file -- not a hardcoded laptop path.
# The clusters run this too, and a literal /Users/... here means the cluster
# checkout silently loses `src` from sys.path after every `git pull`.
for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[v] = '1'
import numpy as np, torch
torch.set_num_threads(1); np.seterr(all='ignore')
from src.physics.grid import Grid2D
from src.physics.eos import hybrid_eos
from src.physics.initial_data2d import b_from_potential, magnetic_rotor, orszag_tang
from src.physics.driver2d import prims_to_cons_2d, sync_state, rk_step_ct, compute_dt_2d
from src.physics.state import EVOLVED_KEYS, State2D
from src.physics.hlld import LAST_DIAG, hlld_flux
from src.physics.exact_flux import exact_flux_batched
from rmhd.util.stats import mcnemar_exact


if __name__ == "__main__":   # guarded: RMHD_POOL workers import this module
    MODE, TAG = sys.argv[1], sys.argv[2]
    RETRIES = int(os.environ.get("RETRIES", "0"))
    REPLAY_OUT = os.environ.get("REPLAY_OUT")          # replay: save masks/p* here
    REC_DIR = os.environ.get("REC_DIR", "/tmp")
    rec_path = lambda tag, k: os.path.join(REC_DIR, f"sweep_{tag}_{k}.pt")
    base = functools.partial(exact_flux_batched, tau_weak=1e-2, tau_bt=1e-9,
                             n_retries=RETRIES, max_iter=40)
    # box, gamma and boundaries per problem, as run_2d.py has them
    PROBLEMS = {"rotor": ((-0.5, 0.5), 5.0 / 3.0, "outflow", magnetic_rotor),
                "orszag_tang": ((0.0, 1.0), 4.0 / 3.0, "periodic", orszag_tang)}
    PROBLEM = os.environ.get("PROBLEM", "rotor")
    if MODE == "record" and PROBLEM not in PROBLEMS:
        sys.exit(f"PROBLEM={PROBLEM}: not one of {', '.join(PROBLEMS)}")
    GAMMA = PROBLEMS.get(PROBLEM, PROBLEMS["rotor"])[1]
    if MODE == "replay":
        # the recording says which gamma it was made with (rotor ones predate
        # the field and are 5/3)
        first = os.path.join(REC_DIR, f"sweep_{sys.argv[3]}_1.pt")
        if os.path.exists(first):
            GAMMA = float(torch.load(first).get("gamma", 5.0 / 3.0))
    eos = hybrid_eos(K=0.0, gamma=GAMMA, gamma_th=GAMMA)
    kern = {k: os.environ.get(k, "numpy") for k in ("RMHD_FAN", "RMHD_ALFVEN", "RMHD_SLOWSHOCK", "RMHD_SHOCK")}
    kern["retries"] = RETRIES
    kern["ckpt"] = os.environ.get("RMHD_ML_CKPT", "(default)")
    kern["ckpts_extra"] = os.environ.get("RMHD_ML_CKPTS", "")


    if MODE == "compare":
        # compare <A.npz> <B.npz>: paired per-lane convergence of two replays of
        # the SAME recorded inputs (McNemar on the discordant lanes)
        A, B = np.load(sys.argv[2]), np.load(sys.argv[3])
        tb = tc = ta = 0
        n_sweeps = len([f for f in A.files if f.startswith("mask_")])
        print(f"{'sweep':>5} {'A':>4} {'B':>4} {'both':>5} {'A-only':>6} {'B-only':>6}")
        for k in range(1, n_sweeps + 1):
            ma, mb = A[f"mask_{k}"].astype(bool), B[f"mask_{k}"].astype(bool)
            both, ao, bo = int((ma & mb).sum()), int((ma & ~mb).sum()), int((~ma & mb).sum())
            tb += ao; tc += bo; ta += int(ma.sum())
            print(f"{k:>5} {int(ma.sum()):>4} {int(mb.sum()):>4} {both:>5} {ao:>6} {bo:>6}")
        print(f"total A={ta}  B={ta - tb + tc}  A-only={tb}  B-only={tc}  McNemar p={mcnemar_exact(tb, tc):.3g}")
        sys.exit(0)

    print(MODE, TAG, kern, flush=True)

    if MODE == "record":
        N = int(os.environ.get("N", "32"))
        T0 = float(os.environ.get("T0", "0"))
        os.makedirs(REC_DIR, exist_ok=True)
        (lo, hi), _, BC, init = PROBLEMS[PROBLEM]
        g = Grid2D(lo, hi, N, lo, hi, N, ng=2)
        prims, Az = init(g, eos)
        Bxf, Byf = b_from_potential(Az, g.dx, g.dy)
        c = prims_to_cons_2d(prims, g)
        st = sync_state(State2D(cons={k: c[k] for k in EVOLVED_KEYS}, Bxf=Bxf, Byf=Byf, prims=prims),
                        g, eos, BC, BC)
        # the developed flow: HLLD to T0 (cheap), then the recorded exact steps
        t_pre, n_pre = 0.0, 0
        while t_pre < T0 - 1e-14:
            dt = min(compute_dt_2d(st.prims, g, eos, 0.25), T0 - t_pre)
            st, _ = rk_step_ct(st, g, eos, dt, scheme="rk3", bc_x=BC,
                               bc_y=BC, flux_fn=hlld_flux, limiter="mc")
            t_pre += dt; n_pre += 1
        if n_pre:
            print(f"pre-evolved with HLLD to t={t_pre:.4f} in {n_pre} steps", flush=True)
        k = [0]

        def flux(sL, sR, eos_, idir=0):
            k[0] += 1
            t0 = time.time(); out = base(sL, sR, eos_, idir=idir); w = time.time() - t0
            d = dict(LAST_DIAG)
            torch.save({"sL": sL, "sR": sR, "idir": idir, "F": out[0], "U": out[1], "p_star": out[2],
                        "exact_mask": d["exact_mask"], "n_exact": d["n_exact"],
                        "n_attempted": d["n_attempted"], "t0": T0, "n": N,
                        "problem": PROBLEM, "gamma": GAMMA},
                       rec_path(TAG, k[0]))
            print(f"sweep {k[0]} idir={idir} attempted={d['n_attempted']} exact={d['n_exact']} {w:.1f}s", flush=True)
            return out

        for _step in range(int(os.environ.get("NSTEP", "1"))):
            dt = compute_dt_2d(st.prims, g, eos, 0.25)
            st, _ = rk_step_ct(st, g, eos, dt, scheme="rk3", bc_x=BC, bc_y=BC,
                               flux_fn=flux, limiter="mc")
        print("recorded", k[0], "sweeps", flush=True)
    else:
        REF = sys.argv[3]
        saved = {}
        # by exact name, not by prefix: REF=base used to count the
        # sweep_base_njit_* files as its own
        pat = re.compile(r"^sweep_%s_(\d+)\.pt$" % re.escape(REF))
        n_sweeps = len([f for f in os.listdir(REC_DIR) if pat.match(f)])
        for k in range(1, min(int(os.environ.get("WARMUP", "0")), n_sweeps) + 1):
            rec = torch.load(rec_path(REF, k))
            base(rec["sL"], rec["sR"], eos, idir=rec["idir"])
        for k in range(1, n_sweeps + 1):
            rec = torch.load(rec_path(REF, k))
            t0 = time.time(); F, U, p_star = base(rec["sL"], rec["sR"], eos, idir=rec["idir"]); w = time.time() - t0
            d = dict(LAST_DIAG); m2 = d["exact_mask"]; m1 = rec["exact_mask"]
            saved[f"mask_{k}"] = m2.numpy().copy(); saved[f"p_{k}"] = p_star.reshape(-1).numpy().copy()
            saved[f"rescued_{k}"] = d.get("n_retry_rescued", 0)
            saved[f"wall_{k}"] = w
            saved[f"skipped_{k}"] = d.get("n_compound_skip", 0)
            # lanes the compound rule routed to HLLD that the reference solved
            # exactly: the price of the routing, in fluxes and in how far the
            # flux moves (RMHD_COMPOUND_SKIP; plan F step 4)
            rt = d.get("compound_mask")
            rt = torch.as_tensor(np.asarray(rt), dtype=torch.bool) if rt is not None \
                else torch.zeros_like(m1)
            lost = m1 & rt
            saved[f"routed_{k}"] = rt.numpy().copy()
            both = m1 & m2
            only_rec, only_now = int((m1 & ~m2).sum()), int((~m1 & m2).sum())
            # Scale by the flux's OWN physical size, not each component's.
            # In a 2D planar sweep Sz and Bz are the out-of-plane components:
            # identically zero up to roundoff (|Sz| ~ 9e-9 against |Sx| ~ 38).
            # Dividing those by their own max turns a 2e-8 difference into a
            # reported "2.11 relative", which is how this diagnostic cried
            # wolf on 2026-09-16.  The old `or 1.0` guard only fired on an
            # exact 0.0, so roundoff-level components slipped through.
            gsc = max(float(rec["F"][k2].abs().max()) for k2 in F) or 1.0
            worst, wkey = 0.0, None
            for key in F:
                a = rec["F"][key].reshape(-1); b = F[key].reshape(-1)
                sc = max(float(a.abs().max()), 1e-6 * gsc)
                dd = ((a - b).abs() / sc)[both]
                if dd.numel() and float(dd.max()) > worst:
                    worst, wkey = float(dd.max()), key
            ps, ps2 = rec["p_star"].reshape(-1), p_star.reshape(-1)
            dps = ((ps - ps2).abs() / ps.abs().clamp(min=1e-30))[both]
            if int(lost.sum()):
                gsc2 = max(float(rec["F"][k2].abs().max()) for k2 in F) or 1.0
                w2, wk2 = 0.0, None
                for key in F:
                    a = rec["F"][key].reshape(-1); b = F[key].reshape(-1)
                    sc = max(float(a.abs().max()), 1e-6 * gsc2)
                    dd = ((a - b).abs() / sc)[lost]
                    if dd.numel() and float(dd.max()) > w2:
                        w2, wk2 = float(dd.max()), key
                print(f"   routed: {int(rt.sum())} lanes, {int(lost.sum())} of them were "
                      f"exact in the reference; flux rel diff on those max={w2:.2e} ({wk2})",
                      flush=True)
            print(f"sweep {k} idir={rec['idir']}: rec exact={int(m1.sum())} now exact={int(m2.sum())} "
                  f"flips: rec-only {only_rec}, now-only {only_now} | common-exact flux rel diff max={worst:.2e} ({wkey}) "
                  f"p* rel max={float(dps.max()) if dps.numel() else 0:.2e} | {w:.1f}s", flush=True)
            idx = torch.nonzero(m1 != m2).reshape(-1).tolist()[:10]
            if idx:
                print("   flipped lanes:", idx, flush=True)
            if dps.numel():
                top = torch.topk(dps, min(6, dps.numel()))
                print("   top p* rel diffs:", [f"{v:.1e}" for v in top.values.tolist()],
                      "lanes", torch.nonzero(both).reshape(-1)[top.indices].tolist(), flush=True)
        if REPLAY_OUT:
            np.savez(REPLAY_OUT, **saved)
            print("saved", REPLAY_OUT, flush=True)
