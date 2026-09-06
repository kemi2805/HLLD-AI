"""Run a 2D SRMHD problem (magnetic rotor / Orszag-Tang) and save snapshots."""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
          "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np, torch

from src.physics.eos import hybrid_eos
from src.physics.grid import Grid2D
from src.physics.ct import div_b
from src.physics.driver2d import (compute_dt_2d, prims_to_cons_2d, rk_step_ct,
                                  sync_state)
from src.physics.initial_data2d import (b_from_potential, magnetic_rotor,
                                        orszag_tang)
from src.physics.state import EVOLVED_KEYS, State2D
from src.physics.envelope import EnvelopeRecorder, format_summary
from src.physics.hlld import (hlld_flux, hlle_flux, hllc_flux, hlld_ai_flux,
                              LAST_DIAG)

# `exact` is built lazily in main(): it needs tau_weak / n_retries bound in,
# and importing exact_flux drags in the rmhd_final solver, which should not
# happen for an ordinary HLLD run.
SOLVERS = {"hlld": hlld_flux, "hlle": hlle_flux, "hllc": hllc_flux,
           "hlld_ai": hlld_ai_flux, "exact": None}

# problem -> (domain, gamma, default t_end, bc_x, bc_y, has pi-rotation symmetry)
PROBLEMS = {
    "rotor":       dict(box=(-0.5, 0.5), gamma=5.0 / 3.0, tend=0.4,
                        bc=("outflow", "outflow"), sym=True),
    "orszag_tang": dict(box=(0.0, 1.0), gamma=4.0 / 3.0, tend=1.0,
                        bc=("periodic", "periodic"), sym=False),
}


def symmetry_error(st, g):
    """Rotor invariance under a pi-rotation about z combined with B -> -B.

    Exact for the continuum solution at all times:
        rho(-r)=rho(r), p(-r)=p(r), v(-r)=-v(r), Bx(-r)=+Bx(r), By(-r)=+By(r)

    Sensitive enough to catch a handedness in the upwind EMF selector, but
    note it is ultimately limited by the solver's DISCRETE fallbacks: a
    single HLLD->HLLE flip that differs between rotated partners injects an
    O(1) local difference.  Interpret it as a solver-robustness metric, not
    purely a constrained-transport one.
    """
    ph = g.phys
    worst = 0.0
    for k, parity in (("rho", +1), ("p", +1), ("vx", -1), ("vy", -1),
                      ("Bx", +1), ("By", +1)):
        a = st.prims[k][ph]
        b = torch.flip(a, dims=(0, 1))
        worst = max(worst, float((a - parity * b).abs().max())
                    / max(float(a.abs().max()), 1e-30))
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="rotor", choices=list(PROBLEMS))
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--tend", type=float, default=None)
    ap.add_argument("--cfl", type=float, default=0.25)
    ap.add_argument("--solver", default="hlld", choices=list(SOLVERS))
    # --- exact-flux options (ignored unless --solver exact) ---
    ap.add_argument("--tau-weak", type=float, default=1e-2,
                    help="interfaces whose relative jump is below this go to "
                         "HLLD.  The error is O(jump^2), so 1e-2 costs ~1e-4 "
                         "relative -- far below scheme truncation -- and it "
                         "is what makes the exact flux affordable at all")
    ap.add_argument("--tau-bt", type=float, default=1e-9,
                    help="interfaces whose tangential field, relative to "
                         "|Bn| + sqrt(P_L+P_R), is below this go straight to "
                         "HLLD.  With Bt = 0 on both sides three of the six "
                         "unknowns are undefined (ln|Bt_CD| and the two "
                         "rotations), so the solver cannot represent an "
                         "answer -- the rotor at t=0 is exactly this case.  "
                         "Kept tight: it should catch the unrepresentable, "
                         "not the merely hard")
    ap.add_argument("--exact-retries", type=int, default=0,
                    help="retry ladder rounds per unconverged interface")
    ap.add_argument("--exact-max-iter", type=int, default=40)
    ap.add_argument("--harvest", type=str, default=None,
                    help="directory to record solved-after-retry and "
                         "unsolved Riemann problems into")
    ap.add_argument("--harvest-stride", type=int, default=1)
    ap.add_argument("--harvest-all", action="store_true",
                    help="record EVERY solved seven-wave lane (not only the "
                         "retry-rescued ones) with high caps; the per-sweep "
                         "coverage map is always recorded")
    ap.add_argument("--emf-mode", default="solver", choices=["solver", "hll"])
    ap.add_argument("--limiter", default="mc")
    ap.add_argument("--no-upwind-emf", action="store_true",
                    help="use plain flux-CT (Balsara-Spicer) instead of "
                         "Gardiner-Stone upwinding")
    ap.add_argument("--out", default=None)
    ap.add_argument("--nsnap", type=int, default=5)
    ap.add_argument("--envelope", default=None,
                    help="record the interface states fed to the Riemann "
                         "solver to this .npz (measured ML training ranges)")
    ap.add_argument("--envelope-every", type=int, default=20,
                    help="record every Nth sweep")
    ap.add_argument("--envelope-samples", type=int, default=4000,
                    help="interfaces subsampled per recorded sweep")
    a = ap.parse_args()

    P = PROBLEMS[a.problem]
    tend = a.tend if a.tend is not None else P["tend"]
    out = a.out or f"results/{a.problem}_{a.n}_{a.solver}"
    bc_x, bc_y = P["bc"]

    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))
    os.makedirs(out, exist_ok=True)
    eos = hybrid_eos(K=0.0, gamma=P["gamma"], gamma_th=P["gamma"])
    lo, hi = P["box"]
    g = Grid2D(lo, hi, a.n, lo, hi, a.n, ng=2)

    if a.problem == "rotor":
        prims, Az = magnetic_rotor(g, eos)
    else:
        prims, Az = orszag_tang(g, eos)

    Bxf, Byf = b_from_potential(Az, g.dx, g.dy)
    c = prims_to_cons_2d(prims, g)
    st = State2D(cons={k: c[k] for k in EVOLVED_KEYS}, Bxf=Bxf, Byf=Byf,
                 prims=prims)
    st = sync_state(st, g, eos, bc_x, bc_y)

    def save(tag, t):
        ph = g.phys
        np.savez_compressed(
            os.path.join(out, f"snap_{tag}.npz"), t=t,
            x=g.x[ph[0]].numpy(), y=g.y[ph[1]].numpy(),
            **{k: st.prims[k][ph].numpy()
               for k in ("rho", "p", "vx", "vy", "Bx", "By")},
            divB=div_b(st.Bxf, st.Byf, g.dx, g.dy)[ph].numpy())

    print(f"{a.problem} {a.n}^2  solver={a.solver}  emf={a.emf_mode}"
          f"{' (flux-CT)' if a.no_upwind_emf else ''}  limiter={a.limiter}  "
          f"cfl={a.cfl}  gamma={P['gamma']:.4f}  bc=({bc_x},{bc_y})  "
          f"t_end={tend}", flush=True)
    save("000", 0.0)
    log = open(os.path.join(out, "diag.csv"), "w")
    log.write("step,t,dt,divB_max,divB_l2,sym_err,rho_max,rho_min,p_min,W_max,"
              "hlle_frac_x,hlle_frac_y,mean_iters,c2p_bad\n")

    rec = (EnvelopeRecorder(n_per_call=a.envelope_samples,
                            every=a.envelope_every)
           if a.envelope else None)

    flux_fn = SOLVERS[a.solver]
    harvester = None
    if a.solver == "exact":
        import functools
        from src.physics.exact_flux import exact_flux_batched
        if a.harvest:
            from src.physics.harvest import Harvester
            hk = dict(stride=a.harvest_stride)
            if a.harvest_all:
                hk.update(only_retried=False, max_solved=5_000_000,
                          max_unsolved=5_000_000)
            harvester = Harvester(a.harvest, P["gamma"], **hk)
        flux_fn = functools.partial(exact_flux_batched,
                                    tau_weak=a.tau_weak,
                                    tau_bt=a.tau_bt,
                                    n_retries=a.exact_retries,
                                    max_iter=a.exact_max_iter,
                                    harvester=harvester)
        print(f"  exact flux: tau_weak={a.tau_weak:g}  tau_bt={a.tau_bt:g}  "
              f"retries={a.exact_retries}  max_iter={a.exact_max_iter}"
              + (f"  harvest -> {a.harvest}" if a.harvest else ""))

    t, step, t0 = 0.0, 0, time.time()
    next_snap = 1
    while t < tend - 1e-14:
        dt = min(compute_dt_2d(st.prims, g, eos, a.cfl), tend - t)
        st, diag = rk_step_ct(st, g, eos, dt, scheme="rk3",
                              bc_x=bc_x, bc_y=bc_y,
                              flux_fn=flux_fn,
                              limiter=a.limiter, emf_mode=a.emf_mode,
                              upwind=not a.no_upwind_emf, recorder=rec)
        t += dt
        step += 1

        if step % 10 == 0 or t >= tend - 1e-14:
            ph = g.phys
            d = div_b(st.Bxf, st.Byf, g.dx, g.dy)[ph]
            v2 = (st.prims["vx"][ph]**2 + st.prims["vy"][ph]**2
                  + st.prims["vz"][ph]**2)
            W = float((1.0 / torch.sqrt(torch.clamp(1 - v2, min=1e-16))).max())
            dx_, dy_ = diag["x"]["diag"], diag["y"]["diag"]
            from src.physics.driver2d import cons_to_prims_2d
            _, status = cons_to_prims_2d(
                {**st.cons, "Bx": st.prims["Bx"], "By": st.prims["By"]},
                g, eos, 1e-10, return_status=True)
            c2p_bad = int((~status["converged"][ph]).sum())
            sym = symmetry_error(st, g) if P["sym"] else float("nan")
            log.write(f"{step},{t:.6e},{dt:.3e},{float(d.abs().max()):.3e},"
                      f"{float(d.pow(2).mean().sqrt()):.3e},{sym:.3e},"
                      f"{float(st.prims['rho'][ph].max()):.4f},"
                      f"{float(st.prims['rho'][ph].min()):.4e},"
                      f"{float(st.prims['p'][ph].min()):.4e},{W:.3f},"
                      f"{dx_.get('frac_hlle_fallback', float('nan')):.4f},"
                      f"{dy_.get('frac_hlle_fallback', float('nan')):.4f},"
                      f"{dx_.get('mean_iters', float('nan')):.2f},{c2p_bad}\n")
            log.flush()
            extra = f"sym={sym:.2e} " if P["sym"] else ""
            print(f"  step {step:5d} t={t:.4f} dt={dt:.2e} "
                  f"divB={float(d.abs().max()):.2e} {extra}"
                  f"rho=[{float(st.prims['rho'][ph].min()):.3f},"
                  f"{float(st.prims['rho'][ph].max()):.3f}] c2p_bad={c2p_bad} "
                  f"({time.time()-t0:.0f}s)", flush=True)

        if t >= next_snap * tend / a.nsnap - 1e-14 and next_snap <= a.nsnap:
            save(f"{next_snap:03d}", t)
            next_snap += 1

    save("fin", t)
    log.close()
    if harvester is not None:
        # the final partial buffers were never written before this call
        print("harvest:", harvester.close(), flush=True)
    if rec is not None:
        os.makedirs(os.path.dirname(a.envelope) or ".", exist_ok=True)
        summ = rec.save(a.envelope, problem=a.problem, n=a.n,
                        solver=a.solver, tend=tend, gamma=P["gamma"])
        nrow = sum(r.shape[0] for r in rec.rows)
        print(f"\nmeasured interface envelope ({nrow} sampled states) "
              f"-> {a.envelope}", flush=True)
        print(format_summary(summ), flush=True)
    print(f"done in {time.time()-t0:.0f}s, {step} steps -> {out}", flush=True)


if __name__ == "__main__":
    main()
