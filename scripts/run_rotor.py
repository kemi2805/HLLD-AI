"""Run the magnetic rotor and save snapshots."""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np, torch

from src.physics.eos import hybrid_eos
from src.physics.grid import Grid2D
from src.physics.ct import div_b
from src.physics.driver2d import (compute_dt_2d, prims_to_cons_2d, rk_step_ct,
                                  sync_state)
from src.physics.initial_data2d import b_from_potential, magnetic_rotor
from src.physics.state import EVOLVED_KEYS, State2D
from src.physics.hlld import hlld_flux, hlle_flux, hllc_flux

SOLVERS = {"hlld": hlld_flux, "hlle": hlle_flux, "hllc": hllc_flux}


def symmetry_error(st, g):
    """Rotor invariance under a pi-rotation about z combined with B -> -B.

    Exact for the continuum solution at all times:
        rho(-r)=rho(r), p(-r)=p(r), v(-r)=-v(r), Bx(-r)=+Bx(r), By(-r)=+By(r)
    Checkable to accumulated round-off, and destroyed by any handedness in
    the upwind EMF selector.
    """
    ph = g.phys
    out = {}
    for k, parity in (("rho", +1), ("p", +1), ("vx", -1), ("vy", -1),
                      ("Bx", +1), ("By", +1)):
        a = st.prims[k][ph]
        b = torch.flip(a, dims=(0, 1))
        d = float((a - parity * b).abs().max())
        out[k] = d / max(float(a.abs().max()), 1e-30)
    return max(out.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--tend", type=float, default=0.4)
    ap.add_argument("--cfl", type=float, default=0.25)
    ap.add_argument("--solver", default="hlld", choices=list(SOLVERS))
    ap.add_argument("--emf-mode", default="solver", choices=["solver", "hll"])
    ap.add_argument("--limiter", default="mc")
    ap.add_argument("--out", default="results/rotor")
    ap.add_argument("--nsnap", type=int, default=5)
    a = ap.parse_args()

    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))
    os.makedirs(a.out, exist_ok=True)
    eos = hybrid_eos(K=0.0, gamma=5.0 / 3.0, gamma_th=5.0 / 3.0)
    g = Grid2D(-0.5, 0.5, a.n, -0.5, 0.5, a.n, ng=2)

    prims, Az = magnetic_rotor(g, eos)
    Bxf, Byf = b_from_potential(Az, g.dx, g.dy)
    c = prims_to_cons_2d(prims, g)
    st = State2D(cons={k: c[k] for k in EVOLVED_KEYS}, Bxf=Bxf, Byf=Byf,
                 prims=prims)
    st = sync_state(st, g, eos)

    def save(tag, t):
        ph = g.phys
        np.savez_compressed(
            os.path.join(a.out, f"snap_{tag}.npz"), t=t,
            x=g.x[ph[0]].numpy(), y=g.y[ph[1]].numpy(),
            **{k: st.prims[k][ph].numpy()
               for k in ("rho", "p", "vx", "vy", "Bx", "By")},
            divB=div_b(st.Bxf, st.Byf, g.dx, g.dy)[ph].numpy())

    print(f"rotor {a.n}^2  solver={a.solver}  emf={a.emf_mode}  "
          f"limiter={a.limiter}  cfl={a.cfl}  t_end={a.tend}", flush=True)
    save("000", 0.0)
    log = open(os.path.join(a.out, "diag.csv"), "w")
    log.write("step,t,dt,divB_max,divB_l2,sym_err,rho_max,p_min,W_max,"
              "hlle_frac_x,hlle_frac_y,mean_iters\n")

    t, step, t0 = 0.0, 0, time.time()
    next_snap = 1
    while t < a.tend - 1e-14:
        dt = min(compute_dt_2d(st.prims, g, eos, a.cfl), a.tend - t)
        st, diag = rk_step_ct(st, g, eos, dt, scheme="rk3",
                              flux_fn=SOLVERS[a.solver],
                              limiter=a.limiter, emf_mode=a.emf_mode)
        t += dt
        step += 1

        if step % 10 == 0 or t >= a.tend - 1e-14:
            ph = g.phys
            d = div_b(st.Bxf, st.Byf, g.dx, g.dy)[ph]
            v2 = st.prims["vx"][ph]**2 + st.prims["vy"][ph]**2 + st.prims["vz"][ph]**2
            W = float((1.0 / torch.sqrt(torch.clamp(1 - v2, min=1e-16))).max())
            dx_, dy_ = diag["x"]["diag"], diag["y"]["diag"]
            log.write(f"{step},{t:.6e},{dt:.3e},{float(d.abs().max()):.3e},"
                      f"{float(d.pow(2).mean().sqrt()):.3e},"
                      f"{symmetry_error(st, g):.3e},"
                      f"{float(st.prims['rho'][ph].max()):.4f},"
                      f"{float(st.prims['p'][ph].min()):.4e},{W:.3f},"
                      f"{dx_.get('frac_hlle_fallback', float('nan')):.4f},"
                      f"{dy_.get('frac_hlle_fallback', float('nan')):.4f},"
                      f"{dx_.get('mean_iters', float('nan')):.2f}\n")
            log.flush()
            print(f"  step {step:5d} t={t:.4f} dt={dt:.2e} "
                  f"divB={float(d.abs().max()):.2e} sym={symmetry_error(st,g):.2e} "
                  f"rho_max={float(st.prims['rho'][ph].max()):.2f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

        if t >= next_snap * a.tend / a.nsnap - 1e-14 and next_snap <= a.nsnap:
            save(f"{next_snap:03d}", t)
            next_snap += 1

    save("fin", t)
    log.close()
    print(f"done in {time.time()-t0:.0f}s, {step} steps -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
