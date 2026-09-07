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

Env: RMHD_ROOT, RMHD_ML_CKPT, RMHD_FAN/ALFVEN/SLOWSHOCK/SHOCK as for a run."""
import sys, os, time, functools
sys.path.insert(0, '/Users/miler/Codes/HLLD')
for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[v] = '1'
import numpy as np, torch
torch.set_num_threads(1); np.seterr(all='ignore')
from src.physics.grid import Grid2D
from src.physics.eos import hybrid_eos
from src.physics.initial_data2d import b_from_potential, magnetic_rotor
from src.physics.driver2d import prims_to_cons_2d, sync_state, rk_step_ct, compute_dt_2d
from src.physics.state import EVOLVED_KEYS, State2D
from src.physics.hlld import LAST_DIAG
from src.physics.exact_flux import exact_flux_batched


if __name__ == "__main__":   # guarded: RMHD_POOL workers import this module
    MODE, TAG = sys.argv[1], sys.argv[2]
    RETRIES = int(os.environ.get("RETRIES", "0"))
    REPLAY_OUT = os.environ.get("REPLAY_OUT")          # replay: save masks/p* here
    base = functools.partial(exact_flux_batched, tau_weak=1e-2, tau_bt=1e-9,
                             n_retries=RETRIES, max_iter=40)
    eos = hybrid_eos(K=0.0, gamma=5 / 3, gamma_th=5 / 3)
    kern = {k: os.environ.get(k, "numpy") for k in ("RMHD_FAN", "RMHD_ALFVEN", "RMHD_SLOWSHOCK", "RMHD_SHOCK")}
    kern["retries"] = RETRIES
    kern["ckpt"] = os.environ.get("RMHD_ML_CKPT", "(default)")
    kern["ckpts_extra"] = os.environ.get("RMHD_ML_CKPTS", "")


    def _mcnemar(b, c):
        """Exact two-sided McNemar p-value from the discordant counts."""
        import math
        n = b + c
        if n == 0:
            return 1.0
        k = min(b, c)
        tail = sum(math.comb(n, j) for j in range(0, k + 1)) / 2.0 ** n
        return min(1.0, 2.0 * tail)


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
        print(f"total A={ta}  B={ta - tb + tc}  A-only={tb}  B-only={tc}  McNemar p={_mcnemar(tb, tc):.3g}")
        sys.exit(0)

    print(MODE, TAG, kern, flush=True)

    if MODE == "record":
        N = 32
        g = Grid2D(-0.5, 0.5, N, -0.5, 0.5, N, ng=2)
        prims, Az = magnetic_rotor(g, eos)
        Bxf, Byf = b_from_potential(Az, g.dx, g.dy)
        c = prims_to_cons_2d(prims, g)
        st = sync_state(State2D(cons={k: c[k] for k in EVOLVED_KEYS}, Bxf=Bxf, Byf=Byf, prims=prims),
                        g, eos, "outflow", "outflow")
        k = [0]

        def flux(sL, sR, eos_, idir=0):
            k[0] += 1
            t0 = time.time(); out = base(sL, sR, eos_, idir=idir); w = time.time() - t0
            d = dict(LAST_DIAG)
            torch.save({"sL": sL, "sR": sR, "idir": idir, "F": out[0], "U": out[1], "p_star": out[2],
                        "exact_mask": d["exact_mask"], "n_exact": d["n_exact"],
                        "n_attempted": d["n_attempted"]}, f"/tmp/sweep_{TAG}_{k[0]}.pt")
            print(f"sweep {k[0]} idir={idir} attempted={d['n_attempted']} exact={d['n_exact']} {w:.1f}s", flush=True)
            return out

        for _step in range(int(os.environ.get("NSTEP", "1"))):
            dt = compute_dt_2d(st.prims, g, eos, 0.25)
            st, _ = rk_step_ct(st, g, eos, dt, scheme="rk3", bc_x="outflow", bc_y="outflow",
                               flux_fn=flux, limiter="mc")
        print("recorded", k[0], "sweeps", flush=True)
    else:
        REF = sys.argv[3]
        saved = {}
        n_sweeps = len([f for f in os.listdir("/tmp") if f.startswith(f"sweep_{REF}_") and f.endswith(".pt")])
        for k in range(1, n_sweeps + 1):
            rec = torch.load(f"/tmp/sweep_{REF}_{k}.pt")
            t0 = time.time(); F, U, p_star = base(rec["sL"], rec["sR"], eos, idir=rec["idir"]); w = time.time() - t0
            d = dict(LAST_DIAG); m2 = d["exact_mask"]; m1 = rec["exact_mask"]
            saved[f"mask_{k}"] = m2.numpy().copy(); saved[f"p_{k}"] = p_star.reshape(-1).numpy().copy()
            saved[f"rescued_{k}"] = d.get("n_retry_rescued", 0)
            both = m1 & m2
            only_rec, only_now = int((m1 & ~m2).sum()), int((~m1 & m2).sum())
            worst, wkey = 0.0, None
            for key in F:
                a = rec["F"][key].reshape(-1); b = F[key].reshape(-1)
                sc = float(a.abs().max()) or 1.0
                dd = ((a - b).abs() / sc)[both]
                if dd.numel() and float(dd.max()) > worst:
                    worst, wkey = float(dd.max()), key
            ps, ps2 = rec["p_star"].reshape(-1), p_star.reshape(-1)
            dps = ((ps - ps2).abs() / ps.abs().clamp(min=1e-30))[both]
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
